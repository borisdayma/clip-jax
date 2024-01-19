# coding=utf-8
# Copyright 2023 The OpenAI Team Authors, The Google Flax Team Authors, The HuggingFace Inc. team, The Craiyon team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import functools
import operator
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import partitioning as nn_partitioning
from flax.linen.linear import DotGeneralT, PrecisionLike
from flax.linen.module import merge_param
from flax.linen.partitioning import ScanIn
from transformers import FlaxGenerationMixin, GenerationConfig

remat = nn_partitioning.remat

Axes = Union[int, Iterable[int]]


# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]
Axes = Union[int, Iterable[int]]

# default initializers
default_kernel_init = nn.initializers.lecun_normal()


# Output types, for compatibility with FlaxGenerationMixin
class BaseOutput(OrderedDict):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        super().__setattr__(item, value)


@flax.struct.dataclass
class EncoderOutput(BaseOutput):
    last_hidden_state: Array


@flax.struct.dataclass
class DecoderOutput(BaseOutput):
    logits: Array
    past_key_values: Array


# Utility functions
def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == "linear":
        return lambda x: x
    elif isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    elif callable(fn_or_string):
        return fn_or_string
    else:
        raise ValueError("don't know how to convert %s to an activation function" % (fn_or_string,))


# TODO: will be used for https://github.com/borisdayma/clip-jax/issues/12
def _interpolate(idxs, values):
    """
    Interpolate values at given indices.

    Args:
        idxs: should be fractional, between 0 and 1
        values: values to interpolate, assumed to be evenly spaced between 0 and 1
    """
    idxs = idxs * (values.shape[0] - 1)
    idxs_floor = jnp.floor(idxs)
    idxs_ceil = jnp.ceil(idxs)
    idxs_frac = idxs - idxs_floor.astype(jnp.float32)
    idxs_floor = idxs_floor.astype(jnp.int32)
    idxs_ceil = idxs_ceil.astype(jnp.int32)
    values_floor = jnp.take(values, idxs_floor, axis=0)
    values_ceil = jnp.take(values, idxs_ceil, axis=0)
    idxs_frac = idxs_frac[..., None]
    return (1 - idxs_frac) * values_floor + idxs_frac * values_ceil


# sincos2d position - Source: https://github.com/google-research/big_vision


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


# Rotary Embeddings


# source: https://github.com/google/flaxformer/blob/main/flaxformer/architectures/perceiver_ar/rotary_embedding.py
def rotate_half(x: Array) -> Array:
    """Helper that splits a tensor at last dim into half and rotate it."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


# source: https://github.com/google/flaxformer/blob/main/flaxformer/architectures/perceiver_ar/rotary_embedding.py
@functools.partial(jax.jit, static_argnums=(4,))
def apply_rotary_embedding(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
    decode: bool = False,
    q_position_offset: Optional[Array] = None,
    rotary_index: Optional[Array] = None,
) -> Tuple[Array, Array]:
    """Helper function to apply Rotary Embeddings, supports Q position offset."""
    if len(k.shape) == 3:
        # for multi query attention
        k = jnp.expand_dims(k, 2)
        multiquery = True
    else:
        multiquery = False

    batch, qlen, qheads, d = q.shape
    kbatch, klen, kheads, kd = k.shape
    assert batch == kbatch, f"{batch} != {kbatch}"
    assert d == kd, f"{d} != {kd}"

    # cos: [len, d]
    # sin: [len, d]
    # rotary_index: [batch]
    # q_position_offset: [batch]

    if decode and qlen == 1 and rotary_index is not None:
        # we check qlen == 1 so that we don't do this when initializing cache.
        qcos = cos[rotary_index, :]
        qsin = sin[rotary_index, :]
        # qcos, qsin: [batch, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qlen, qheads, d), (0, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qlen, qheads, d), (0, 3))
        # qcos, qsin: [batch, qlen, qheads, d]
    else:
        if q_position_offset is None:
            qcos, qsin = cos[:qlen, :], sin[:qlen, :]
        else:
            # If q_position_offset is specified, we'll slice per-example after
            # broadcasting to batch size.
            qcos, qsin = cos, sin

        # qcos, qsin: [qlen, d]
        qcos = jax.lax.broadcast_in_dim(qcos, (batch, qcos.shape[0], qheads, d), (1, 3))
        qsin = jax.lax.broadcast_in_dim(qsin, (batch, qsin.shape[0], qheads, d), (1, 3))
        # qcos, qsin: [batch, qlen, qheads, d]
        if q_position_offset is not None:
            qcos = jax.vmap(functools.partial(jax.lax.dynamic_slice_in_dim, slice_size=qlen, axis=0))(
                qcos, q_position_offset
            )
            qsin = jax.vmap(functools.partial(jax.lax.dynamic_slice_in_dim, slice_size=qlen, axis=0))(
                qsin, q_position_offset
            )

    kcos, ksin = cos[:klen, :], sin[:klen, :]
    # kcos, ksin: [klen, d]
    kcos = jax.lax.broadcast_in_dim(kcos, (batch, klen, kheads, d), (1, 3))
    ksin = jax.lax.broadcast_in_dim(ksin, (batch, klen, kheads, d), (1, 3))
    # kcos, ksin: [batch, klen, kheads, d]

    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    if multiquery:
        out_k = jnp.squeeze(out_k, 2)
    return out_q, out_k


# source: https://github.com/google/flaxformer/blob/main/flaxformer/components/embedding.py
def generate_fixed_pos_embedding(features, length, min_timescale=1.0, max_timescale=10000.0):
    """Generate Sin/Cos for Rotary Embeddings.
    Generates sinusoids at (features//2) different timescales, where the
    timescales form a gemetric series from min_timescale to max_timescale
    (max_timescale is not included, but would be the next element in the series).
    Sinusoids are evaluated at integer positions i in [0, length).
    The outputs are computed as:
      output_sin[i, j] = sin(i / timescale[j])
      output_cos[i, j] = cos(i / timescale[j])
    Finally, the outputs are tiled twice in the features dimension.
    Args:
      features: an integer
      length: an integer
      min_timescale: an optional float
      max_timescale: an optional float
    Returns:
      output_sin: a float32 Tensor with shape [length, features]
      output_cos: a float32 Tensor with shape [length, features]
    """
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, since rounding off to a bfloat16 is
    # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
    # from sin(256).
    sinusoid_inp = jnp.einsum(
        "i , j -> i j", jnp.arange(length), rotational_frequency, precision=jax.lax.Precision.HIGHEST
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def norm(use_rmsnorm):
    """Normalization wrapper"""
    if use_rmsnorm:
        # no bias
        return lambda use_bias, bias_init, *args, **kwargs: nn.RMSNorm(*args, **kwargs)
    else:
        return nn.LayerNorm


class CLIPVisionEmbeddings(nn.Module):
    hidden_size: int
    use_bias: bool
    patch_size: int
    position_embedding_type: str  # "learnt" or "sincos2d"
    position_embedding_shape: Optional[Tuple[int, int, int]]
    position_embedding_factorized: bool
    pool_type: str  # "tok", "gap", "map" per google-research/big_vision
    registers: int
    dtype: Dtype

    @nn.compact
    def __call__(self, pixel_values):
        assert self.position_embedding_type in [
            "learnt",
            "sincos2d",
        ], f"Unknown position embedding type {self.position_embedding_type}"
        if self.position_embedding_shape is not None or self.position_embedding_factorized:
            assert self.position_embedding_type == "learnt", f"Position embedding must be learnt."
        patch_embeds = nn.Conv(
            self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.lecun_normal(), ("conv_height", "conv_width", "input_channels", "embed")
            ),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="patch_embeds",
        )(pixel_values)
        patch_embeds = nn.with_logical_constraint(patch_embeds, ("batch", "height", "width", "embed"))
        batch_size, height, width, channels = patch_embeds.shape
        num_patches = height * width
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, num_patches, channels))
        patch_embeds = nn.with_logical_constraint(patch_embeds, ("batch", "length", "embed"))
        if self.position_embedding_type == "learnt":
            num_positions = num_patches
            position_height, position_width = height, width
            if self.position_embedding_shape is not None:
                num_positions = self.position_embedding_shape[0] * self.position_embedding_shape[1]
                position_height, position_width = self.position_embedding_shape[0], self.position_embedding_shape[1]
            if self.position_embedding_factorized:
                position_embeds_height = self.param(
                    "position_embeds_height",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)), (None, "height", "embed")
                    ),
                    (1, position_height, self.hidden_size),
                )
                position_embeds_width = self.param(
                    "position_embeds_width",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)), (None, "width", "embed")
                    ),
                    (1, position_width, self.hidden_size),
                )
                if position_height != height:
                    # interpolate
                    position_embeds_height = jax.image.resize(position_embeds, (height,), method="linear")
                    position_embeds_height = nn.with_logical_constraint(
                        position_embeds_height, ("batch", "height", "embed")
                    )
                if position_width != width:
                    # interpolate
                    position_embeds_width = jax.image.resize(position_embeds, (width,), method="linear")
                    position_embeds_width = nn.with_logical_constraint(
                        position_embeds_width, ("batch", "width", "embed")
                    )
                # make it 2d
                position_embeds_height = position_embeds[:, :, None, :]
                position_embeds_width = position_embeds[:, None, :, :]
                position_embeds = position_embeds_height + position_embeds_width
                assert position_embeds.shape == (batch_size, height, width, self.hidden_size)
            else:
                position_embeds = self.param(
                    "position_embeds",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)), (None, "vocab", "embed")
                    ),
                    (1, num_positions, self.hidden_size),
                )
                if num_positions != num_patches:
                    position_embeds = jnp.reshape(
                        position_embeds,
                        (1, self.position_embedding_shape[0], self.position_embedding_shape[1], self.hidden_size),
                    )
                    position_embeds = nn.with_logical_constraint(
                        position_embeds, ("batch", "height", "width", "embed")
                    )
                    # interpolate
                    position_embeds = jax.image.resize(position_embeds, (height, width), method="linear")
                    position_embeds = nn.with_logical_constraint(
                        position_embeds, ("batch", "height", "width", "embed")
                    )
                    position_embeds = jnp.reshape(position_embeds, (1, num_patches, self.hidden_size))
        elif self.position_embedding_type == "sincos2d":
            position_embeds = posemb_sincos_2d(height, width, self.hidden_size, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown position embedding type {self.position_embedding_type}")
        embeddings = patch_embeds + position_embeds
        embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.pool_type == "tok":
            cls_token = self.param(
                "cls_token",
                nn.with_logical_partitioning(nn.initializers.zeros_init(), (None, None, "embed")),
                (1, 1, self.hidden_size),
            )
            embeddings = jnp.concatenate([jnp.tile(cls_token, [batch_size, 1, 1]), embeddings], axis=1)
            embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.registers:
            registers = self.param(
                "registers",
                nn.with_logical_partitioning(
                    nn.initializers.normal(1 / np.sqrt(self.hidden_size)), (None, None, "embed")
                ),
                (1, self.registers, self.hidden_size),
            )
            embeddings = jnp.concatenate([embeddings, jnp.tile(registers, [batch_size, 1, 1])], axis=1)
            embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    hidden_size: int
    vocab_size: int
    max_length: int
    position_embedding_type: str  # "learnt" or "rotary"
    dtype: Dtype

    @nn.compact
    def __call__(self, input_ids):
        assert self.position_embedding_type in ["learnt", "rotary"]
        embed_dim = self.hidden_size
        embeddings = nn.Embed(
            self.vocab_size,
            embed_dim,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.normal(1 / np.sqrt(embed_dim)), ("vocab", "embed")
            ),
            name="embeddings",
        )(input_ids.astype("i4"))
        embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.position_embedding_type == "learnt":
            position_embeds = self.param(
                "position_embeds",
                nn.with_logical_partitioning(nn.initializers.normal(1 / np.sqrt(embed_dim)), (None, "vocab", "embed")),
                (1, self.max_length, embed_dim),
            )
            embeddings += position_embeds
            embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        return embeddings


class MultiHeadDotProductAttention(nn.Module):
    """
    Adapted from nn.MultiHeadDotProductAttention + maxtext:
    """

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    float32_logits: bool = False  # compute logits in float32 for stability
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = nn.dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = jax.lax.dot_general
    out_dot_general: DotGeneralT = jax.lax.dot_general
    # custom config
    use_rotary: bool = False
    max_length: Optional[int] = None  # required if use_rotary
    embed_dim_name: str = "embed"
    normalize_qk: bool = False
    kernel_init_out: Optional[Callable[[PRNGKey, Shape, Dtype], Array]] = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, inputs_q: Array, inputs_kv: Array, mask: Optional[Array] = None, deterministic: Optional[bool] = None
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        with jax.profiler.TraceAnnotation("Attention_Block"):
            dense = functools.partial(
                nn.DenseGeneral,
                axis=-1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                features=(self.num_heads, head_dim),
                kernel_init=nn.with_logical_partitioning(self.kernel_init, (self.embed_dim_name, "heads", "kv")),
                bias_init=nn.with_logical_partitioning(self.bias_init, ("kv",)),
                use_bias=self.use_bias,
                precision=self.precision,
                dot_general=self.qkv_dot_general,
            )
            # project inputs_q to multi-headed q/k/v
            # dimensions are then [batch..., length, n_heads, n_features_per_head]
            query, key, value = (
                dense(name="query")(inputs_q),
                dense(name="key")(inputs_kv),
                dense(name="value")(inputs_kv),
            )
            if self.normalize_qk:
                query = nn.with_logical_constraint(query, ("batch", "length", "heads", "kv"))
                key = nn.with_logical_constraint(key, ("batch", "length", "heads", "kv"))
                query = norm(use_rmsnorm=True)(
                    dtype=self.dtype,
                    use_bias=False,
                    use_scale=True,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("kv",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("kv",)),
                    name="norm_query",
                )(query)
                key = norm(use_rmsnorm=True)(
                    dtype=self.dtype,
                    use_bias=False,
                    use_scale=True,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("kv",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("kv",)),
                    name="norm_key",
                )(key)

            # ensure correct sharding
            query = nn.with_logical_constraint(query, ("batch", "length", "heads", "kv"))
            key = nn.with_logical_constraint(key, ("batch", "length", "heads", "kv"))
            value = nn.with_logical_constraint(value, ("batch", "length", "heads", "kv"))

            if self.use_rotary:
                assert self.max_length is not None, "max_length must be specified for rotary embeddings."
                # source: https://github.com/google-research/jestimator/blob/main/jestimator/models/rope/modeling.py
                sin, cos = generate_fixed_pos_embedding(head_dim, self.max_length)
                query, key = apply_rotary_embedding(query, key, cos, sin)
                # convert to correct type
                query = query.astype(self.dtype)
                key = key.astype(self.dtype)

                # ensure sharding
                query = nn.with_logical_constraint(query, ("batch", "length", "heads", "kv"))
                key = nn.with_logical_constraint(key, ("batch", "length", "heads", "kv"))

            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            if self.decode:
                # detect if we're initializing by absence of existing cache data.
                is_initialized = self.has_variable("cache", "cached_key")
                cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
                cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
                cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
                if is_initialized:
                    *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
                    # shape check of cached keys against query input
                    expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                    if expected_shape != query.shape:
                        raise ValueError(
                            "Autoregressive cache shape error, "
                            "expected query shape %s instead got %s." % (expected_shape, query.shape)
                        )
                    # update key, value caches with our new 1d spatial slices
                    cur_index = cache_index.value
                    indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                    key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
                    value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
                    cached_key.value = key
                    cached_value.value = value
                    cache_index.value = cache_index.value + 1
                    # causal mask for cached decoder self-attention:
                    # our single query position should only attend to those key
                    # positions that have already been generated and cached,
                    # not the remaining zero elements.
                    mask = nn.combine_masks(
                        mask,
                        jnp.broadcast_to(jnp.arange(max_length) <= cur_index, tuple(batch_dims) + (1, 1, max_length)),
                    )

            dropout_rng = None
            if self.dropout_rate > 0.0:  # Require `deterministic` only if using dropout.
                m_deterministic = merge_param("deterministic", self.deterministic, deterministic)
                if not m_deterministic:
                    dropout_rng = self.make_rng("dropout")
            else:
                m_deterministic = True

            # Casting logits and softmax computation for float32 for model stability
            if self.float32_logits:
                query = query.astype(jnp.float32)
                key = key.astype(jnp.float32)

            # apply attention
            x = self.attention_fn(
                query,
                key,
                value,
                mask=mask,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=m_deterministic,
                dtype=self.dtype,
                precision=self.precision,
            )  # pytype: disable=wrong-keyword-args
            # back to the original inputs dimensions
            kernel_init_out = self.kernel_init_out if self.kernel_init_out is not None else self.kernel_init
            out = nn.DenseGeneral(
                features=features,
                axis=(-2, -1),
                kernel_init=nn.with_logical_partitioning(kernel_init_out, ("heads", "kv", self.embed_dim_name)),
                bias_init=nn.with_logical_partitioning(self.bias_init, (self.embed_dim_name,)),
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                dot_general=self.out_dot_general,
                name="out",  # type: ignore[call-arg]
            )(x)
            return out


class MAPHead(nn.Module):
    """
    Multihead Attention Pooling
    Adapted from google-reasearch/big_vision
    """

    mlp_dim: int
    num_heads: int
    ln_type: str
    dtype: Any
    use_bias: bool
    force_scale: bool
    use_rmsnorm: bool
    normalize_qk: bool
    activations: Sequence[Union[str, Callable]]
    attention_dropout: float
    mlp_dropout_rate: float
    float32_logits: bool

    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        batch, length, embed_dim = x.shape
        probe = self.param(
            "probe",
            nn.with_logical_partitioning(nn.initializers.xavier_uniform(), (None, None, "embed")),
            (1, 1, embed_dim),
        )
        probe = jnp.tile(probe, [batch, 1, 1])
        probe = nn.with_logical_constraint(probe, ("batch", None, "embed"))
        x = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.use_bias,
            use_rotary=False,
            max_length=None,
            dropout_rate=self.attention_dropout,
            decode=False,
            normalize_qk=self.normalize_qk,
            float32_logits=self.float32_logits,
            name="attention",
        )(inputs_q=probe, inputs_kv=x, mask=None, deterministic=deterministic)
        x = nn.with_logical_constraint(x, ("batch", "length", "embed"))
        y = CLIPMLP(
            mlp_dim=self.mlp_dim,
            ln_type=self.ln_type,
            activations=self.activations,
            mlp_dropout_rate=self.mlp_dropout_rate,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            use_rmsnorm=self.use_rmsnorm,
            dtype=self.dtype,
            name="mlp",
        )(x, deterministic=deterministic)
        y = nn.with_logical_constraint(y, ("batch", "length", "embed"))
        x = x + y
        x = nn.with_logical_constraint(x, ("batch", "length", "embed"))
        return x[:, 0]


class CLIPMLP(nn.Module):
    mlp_dim: int
    ln_type: str  # "preln", "normformer"
    activations: Sequence[Union[str, Callable]] = ("relu",)
    mlp_dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    use_bias: bool = False
    force_scale: bool = False
    use_rmsnorm: bool = True

    @nn.compact
    def __call__(self, inputs, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        assert self.ln_type in ["normformer", "preln"], f"ln_type {self.ln_type} not supported."
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
        with jax.profiler.TraceAnnotation("MLP_Block"):
            embed_dim = inputs.shape[-1]
            if self.ln_type in ["normformer", "preln"]:
                inputs = norm(self.use_rmsnorm)(
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    use_scale=self.force_scale,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                    name="pre_mlp_norm",
                )(inputs)
                inputs = nn.with_logical_constraint(inputs, ("batch", "length", "embed"))
            activations = []
            for idx, act_fn in enumerate(self.activations):
                dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
                x = nn.DenseGeneral(
                    self.mlp_dim,
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "mlp")),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("mlp",)),
                    name=dense_name,
                )(inputs)
                x = nn.with_logical_constraint(x, ("batch", "length", "mlp"))
                x = _convert_to_activation_function(act_fn)(x)
                x = nn.with_logical_constraint(x, ("batch", "length", "mlp"))
                activations.append(x)
            # Take elementwise product of above intermediate activations.
            x = functools.reduce(operator.mul, activations)
            x = nn.with_logical_constraint(x, ("batch", "length", "mlp"))

            # layer norm
            if self.ln_type == "normformer":
                x = norm(self.use_rmsnorm)(
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    use_scale=self.force_scale,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("mlp",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("mlp",)),
                    name="mid_mlp_norm",
                )(x)
                x = nn.with_logical_constraint(x, ("batch", "length", "mlp"))
            # Apply dropout and final dense output projection.
            x = nn.Dropout(rate=self.mlp_dropout_rate, broadcast_dims=(-2,), name="mlp_dropout")(
                x, deterministic=deterministic
            )  # Broadcast along length.
            x = nn.with_logical_constraint(x, ("batch", "length", "mlp"))
            output = nn.DenseGeneral(
                embed_dim,
                dtype=self.dtype,
                use_bias=self.use_bias,
                kernel_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("mlp", "embed")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init, ("embed",)),
                name="wo",
            )(x)
            output = nn.with_logical_constraint(output, ("batch", "length", "embed"))
            return output


class CLIPEncoderLayer(nn.Module):
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    position_embedding_type: str  # "learnt", "rotary" or "sincos2d"
    max_length: int
    use_causal_mask: bool
    mlp_dim: int
    decode: bool
    float32_logits: bool
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    normalize_qk: bool = False
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        deterministic: bool = True,
    ):
        assert self.ln_type in ["normformer", "preln"], f"ln_type {self.ln_type} not supported."
        assert self.position_embedding_type in [
            "learnt",
            "rotary",
            "sincos2d",
        ], f"position_embedding_type {self.position_embedding_type} not supported."

        # Self attention
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        residual = hidden_states
        if self.ln_type in ["preln", "normformer"]:
            hidden_states = norm(self.use_rmsnorm)(
                dtype=self.dtype,
                use_bias=self.use_bias,
                use_scale=self.force_scale,
                scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                name="pre_attention_norm",
            )(hidden_states)
            hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.use_bias,
            use_rotary=(self.position_embedding_type == "rotary"),
            max_length=self.max_length,
            dropout_rate=self.attention_dropout,
            decode=self.decode,
            normalize_qk=self.normalize_qk,
            float32_logits=self.float32_logits,
            name="attention",
        )(inputs_q=hidden_states, inputs_kv=hidden_states, mask=attention_mask, deterministic=deterministic)
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        if self.ln_type == "normformer":
            hidden_states = norm(self.use_rmsnorm)(
                dtype=self.dtype,
                use_bias=self.use_bias,
                use_scale=True,
                scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                name="post_attention_norm",
            )(hidden_states)
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = residual + hidden_states
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))

        # Cross-attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            if self.ln_type in ["preln", "normformer"]:
                hidden_states = norm(self.use_rmsnorm)(
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    use_scale=self.force_scale,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                    name="pre_cross_attention_norm",
                )(hidden_states)
                hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            hidden_states = MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                deterministic=deterministic,
                use_bias=self.use_bias,
                use_rotary=False,  # don't apply on cross-attention
                max_length=self.max_length,
                dropout_rate=self.attention_dropout,
                decode=False,
                normalize_qk=self.normalize_qk,
                float32_logits=self.float32_logits,
                name="cross_attention",
            )(inputs_q=hidden_states, inputs_kv=encoder_hidden_states, deterministic=deterministic)
            hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            if self.ln_type == "normformer":
                hidden_states = norm(self.use_rmsnorm)(
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    use_scale=True,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                    name="post_cross_attention_norm",
                )(hidden_states)
            hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            hidden_states = residual + hidden_states
            hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))

        # MLP
        residual = hidden_states
        hidden_states = CLIPMLP(
            mlp_dim=self.mlp_dim,
            ln_type=self.ln_type,
            activations=self.activations,
            mlp_dropout_rate=self.mlp_dropout_rate,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            use_rmsnorm=self.use_rmsnorm,
            dtype=self.dtype,
            name="mlp",
        )(hidden_states)
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = residual + hidden_states
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        return hidden_states, None


class CLIPEncoder(nn.Module):
    num_layers: int
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    position_embedding_type: str  # "learnt", "rotary"
    max_length: int
    use_causal_mask: bool
    mlp_dim: int
    float32_logits: bool
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    normalize_qk: bool = False
    use_bias: bool = False
    force_scale: bool = False
    decode: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    unroll: int = 100  # unroll scan layers
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        deterministic: bool = True,
    ):
        # gradient checkpointing
        use_scan = True
        initializing = self.is_mutable_collection("params")
        params_spec = 0 if initializing else ScanIn(0)
        layer = (
            nn.remat(
                CLIPEncoderLayer,
                static_argnums=(-1,),
                prevent_cse=not use_scan,
            )
            if self.gradient_checkpointing
            else CLIPEncoderLayer
        )

        hidden_states, _ = nn.scan(
            layer,
            variable_axes={"params": params_spec, "cache": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.num_layers,
            unroll=self.unroll,
            metadata_params={nn.PARTITION_NAME: "layer"},
        )(
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
            max_length=self.max_length,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            float32_logits=self.float32_logits,
            dtype=self.dtype,
            activations=self.activations,
            normalize_qk=self.normalize_qk,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            attention_dropout=self.attention_dropout,
            mlp_dropout_rate=self.mlp_dropout_rate,
            decode=self.decode,
            name="layers",
        )(
            hidden_states, attention_mask, encoder_hidden_states, deterministic
        )

        return dict(
            last_hidden_state=hidden_states,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPTextTransformer(nn.Module):
    hidden_size: int
    vocab_size: int
    max_length: int
    position_embedding_type: str  # "learnt" or "rotary"
    num_layers: int
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    use_causal_mask: bool
    mlp_dim: int
    float32_logits: bool = False
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    normalize_qk: bool = False
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    unroll: int = 100  # unroll scan layers
    gradient_checkpointing: bool = True
    eos_token_id: int = None
    mask_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    masked_pred_prob: float = 0.75  # recommended by Cappa
    is_decoder: bool = False  # for Cappa
    dtype: str = "float32"

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states=None,
        decode=False,
        deterministic: bool = True,
    ):
        if self.is_decoder:
            assert encoder_hidden_states is not None
            assert self.mask_token_id is not None
            assert self.use_causal_mask
        else:
            assert encoder_hidden_states is None
        if self.use_causal_mask:
            assert self.eos_token_id is not None

        # attention mask
        if attention_mask is not None:
            attention_mask = nn.make_attention_mask(attention_mask, attention_mask, dtype=self.dtype)
        if self.use_causal_mask:
            causal_mask = nn.make_causal_mask(input_ids)
            if attention_mask is None:
                attention_mask = causal_mask
            else:
                attention_mask = nn.combine_masks(attention_mask, causal_mask)
        if decode and attention_mask is not None:
            print("Warning: attention_mask is ignored in decode mode.")
            attention_mask = None
        
        # decoder mode
        # src: adapted from google-research/big_vision
        if self.is_decoder and not deterministic:
            # randomly mask tokens in some instances
            if self.masked_pred_prob > 0.0:

                def _add_random_masks(a):
                    # Generate random mask
                    # NOTE: masking all tokens is the best per Cappa
                    return jnp.ones_like(a) * self.mask_token_id

                def where(mask, x, y):
                    mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))
                    return jnp.where(mask, x, y)

                do_masked_pred = (
                    jax.random.uniform(self.make_rng("dropout"), (len(input_ids),)) < self.masked_pred_prob
                )
                input_ids = where(do_masked_pred, _add_random_masks(input_ids), input_ids)
                attention_mask = where(do_masked_pred, jnp.ones_like(attention_mask), attention_mask)

        dtype = jnp.dtype(self.dtype)
        hidden_states = CLIPTextEmbeddings(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            position_embedding_type=self.position_embedding_type,
            dtype=dtype,
            name="embeddings",
        )(input_ids=input_ids)

        encoder_outputs = CLIPEncoder(
            num_layers=self.num_layers,
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
            max_length=self.max_length,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            float32_logits=self.float32_logits,
            dtype=dtype,
            activations=self.activations,
            normalize_qk=self.normalize_qk,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            attention_dropout=self.attention_dropout,
            mlp_dropout_rate=self.mlp_dropout_rate,
            unroll=self.unroll,
            gradient_checkpointing=self.gradient_checkpointing,
            decode=decode,
            name="encoder",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        last_hidden_state = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="final_norm",
        )(last_hidden_state)
        last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        if self.is_decoder:
            # dense to vocab
            last_hidden_state = nn.Dense(
                self.vocab_size,
                dtype=dtype,
                kernel_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed", "vocab")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("vocab",)),
                name="logits",
            )(last_hidden_state)
            last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "vocab"))
            pooled_output = None

        else:
            if not self.use_causal_mask:
                # take features from the BOS embedding instead of EOS (no causal mask)
                pooled_output = last_hidden_state[:, 0, :]
            else:
                # take features from the EOS embedding

                def _get_id_pos(mat, id):
                    return jnp.where(mat == id, 1, 0).argmax(axis=-1)

                pooled_output = last_hidden_state[
                    jnp.arange(last_hidden_state.shape[0]), _get_id_pos(input_ids, self.eos_token_id)
                ]
            pooled_output = nn.with_logical_constraint(pooled_output, ("batch", "embed"))

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPVisionTransformer(nn.Module):
    image_size: int
    hidden_size: int
    patch_size: int
    num_layers: int
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    use_causal_mask: bool
    mlp_dim: int
    float32_logits: bool = False
    position_embedding_type: str = "sincos2d"  # "learnt" or "sincos2d"
    position_embedding_shape: Optional[Tuple[int, int]] = None  # e.g. (16, 16) for 256x256 images with patch 16
    position_embedding_factorized: bool = False
    dtype: str = "float32"
    activations: Sequence[Union[str, Callable]] = ("relu",)
    normalize_qk: bool = False
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    pool_type: str = None  # "tok", "gap", "map", None per google-research/big_vision
    unroll: int = 100  # unroll scan layers
    registers: int = 0  # number of registers per "vision transformers need registers"
    keep_registers: bool = False  # keep registers in the output
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        dtype = jnp.dtype(self.dtype)
        batch, height, width, channels = pixel_values.shape
        if not (height == self.image_size and width == self.image_size and channels == 3):
            print(
                f"Warning: Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        hidden_states = CLIPVisionEmbeddings(
            hidden_size=self.hidden_size,
            use_bias=self.use_bias,
            patch_size=self.patch_size,
            position_embedding_type=self.position_embedding_type,
            position_embedding_shape=self.position_embedding_shape,
            position_embedding_factorized=self.position_embedding_factorized,
            pool_type=self.pool_type,
            registers=self.registers,
            dtype=dtype,
            name="embeddings",
        )(pixel_values)
        hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        max_length = hidden_states.shape[1]
        encoder_outputs = CLIPEncoder(
            num_layers=self.num_layers,
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
            max_length=max_length,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            float32_logits=self.float32_logits,
            dtype=dtype,
            activations=self.activations,
            normalize_qk=self.normalize_qk,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            attention_dropout=self.attention_dropout,
            mlp_dropout_rate=self.mlp_dropout_rate,
            unroll=self.unroll,
            gradient_checkpointing=self.gradient_checkpointing,
            name="encoder",
        )(
            hidden_states=hidden_states,
            deterministic=deterministic,
        )

        # get last hidden state
        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # remove registers
        if self.registers and not self.keep_registers:
            if self.pool_type == "tok":
                print("Warning: removing registers in tok pool mode does not seem necessary.")
            last_hidden_state = last_hidden_state[:, : -self.registers]
            last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # layernorm
        last_hidden_state = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="final_norm",
        )(last_hidden_state)
        last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        if self.pool_type == "tok":
            pooled_output = last_hidden_state[:, 0, :]

        elif self.pool_type == "gap":
            # mean pool - jnp.mean -> leads to large memory consumption
            # pooled_output = jnp.mean(last_hidden_state, axis=1)

            # mean pool - for loop -> this works!
            length = last_hidden_state.shape[1]
            pooled_output = last_hidden_state[:, 0, :]
            for i in range(1, length):
                pooled_output = pooled_output + last_hidden_state[:, i, :]
            pooled_output = pooled_output / length

        elif self.pool_type == "map":
            pooled_output = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                ln_type=self.ln_type,
                use_rmsnorm=self.use_rmsnorm,
                use_bias=self.use_bias,
                force_scale=self.force_scale,
                activations=self.activations,
                normalize_qk=self.normalize_qk,
                attention_dropout=self.attention_dropout,
                mlp_dropout_rate=self.mlp_dropout_rate,
                float32_logits=self.float32_logits,
                dtype=dtype,
            )(last_hidden_state)

        elif self.pool_type is None:
            pooled_output = None

        else:
            raise ValueError(f"pool_type {self.pool_type} not supported.")

        pooled_output = nn.with_logical_constraint(pooled_output, ("batch", "embed"))

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPVisionModelForImageClassification(nn.Module):
    vision_config: Any
    num_labels: int
    dtype: str = "float32"

    def __post_init__(self):
        # add default fields vision_config
        default_fields = dataclasses.fields(CLIPVisionTransformer)
        default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
        default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
        vision_config = {**default_fields, **self.vision_config}
        if self.dtype is not None:
            vision_config["dtype"] = self.dtype
        self.vision_config = vision_config
        return super().__post_init__()

    @nn.compact
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
    ):
        dtype = jnp.dtype(self.dtype)
        outputs = CLIPVisionTransformer(
            **self.vision_config,
            name="vision",
        )(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )

        logits = nn.Dense(
            self.num_labels,
            dtype=dtype,
            kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "classifier")),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("classifier",)),
            name="classifier",
        )(outputs["pooled_output"])

        return dict(logits=logits)

    def init_inputs(config, rng: jax.random.PRNGKey):
        vision_config = config.vision_config
        if isinstance(vision_config, dict):
            vision_config = SimpleNamespace(**vision_config)
        pixel_values = jnp.ones((1, vision_config.image_size, vision_config.image_size, 3), dtype="f4")
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return {"rngs": rngs, "pixel_values": pixel_values}

    def init_weights(self, rng: jax.random.PRNGKey):
        inputs = self.init_inputs(rng)
        return self.init(**inputs)


class CLIPTextModelForFineTuning(nn.Module):
    text_config: Any
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        deterministic: bool = True,
    ):
        text_outputs = CLIPTextTransformer(**self.text_config, dtype=self.dtype)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )

        # return penuultimate layer
        return text_outputs["hidden_states"][-2]


class CLIPModel(nn.Module, FlaxGenerationMixin):
    text_config: Any
    vision_config: Any
    projection_dim: int
    logit_scale_init_value: float = 2.6592
    logit_bias_init_value: float = 0.0
    dtype: str = "float32"

    def __post_init__(self):
        # add default fields text_config
        default_fields = dataclasses.fields(CLIPTextTransformer)
        default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
        default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
        text_config = {**default_fields, **self.text_config}
        if self.dtype is not None:
            text_config["dtype"] = self.dtype
        self.text_config = text_config
        # add default fields vision_config
        default_fields = dataclasses.fields(CLIPVisionTransformer)
        default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
        default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
        vision_config = {**default_fields, **self.vision_config}
        if self.dtype is not None:
            vision_config["dtype"] = self.dtype
        self.vision_config = vision_config
        # set config flags for FlaxGenerationMixin
        self.config = SimpleNamespace(is_encoder_decoder=text_config["is_decoder"])
        return super().__post_init__()

    def setup(self):
        if self.text_config["is_decoder"]:
            assert self.vision_config["pool_type"] is None, "pool_type must be None for decoder mode"
        dtype = jnp.dtype(self.dtype)
        self.logit_scale = self.param(
            "logit_scale",
            nn.with_logical_partitioning(nn.initializers.constant(self.logit_scale_init_value), (None,)),
            (1,),
        )
        self.logit_bias = self.param(
            "logit_bias",
            nn.with_logical_partitioning(nn.initializers.constant(self.logit_bias_init_value), (None,)),
            (1,),
        )
        self.text_model = CLIPTextTransformer(
            **self.text_config,
            name="text",
        )
        self.vision_model = CLIPVisionTransformer(
            **self.vision_config,
            name="vision",
        )
        if (not self.text_config["is_decoder"]) and (
            self.text_config["hidden_size"] != self.vision_config["hidden_size"]
        ):
            self.text_projection = nn.Dense(
                self.projection_dim,
                dtype=dtype,
                use_bias=False,
                kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "embed_proj")),
                name="text_projection",
            )
            self.vision_projection = nn.Dense(
                self.projection_dim,
                dtype=dtype,
                use_bias=False,
                kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "embed_proj")),
                name="vision_projection",
            )
        else:
            self.text_projection = None
            self.vision_projection = None

    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        decode: bool = False,
        deterministic: bool = True,
    ):
        image_features = self.get_image_features(pixel_values, deterministic=deterministic)
        image_embeds, vision_model_output = image_features["image_embeds"], image_features["vision_model_output"]

        is_decoder = self.text_config["is_decoder"]
        if decode:
            assert is_decoder, "decode=True only works for decoder mode"
        text_features = self.get_text_features(
            input_ids,
            attention_mask,
            encoder_hidden_states=vision_model_output["last_hidden_state"] if is_decoder else None,
            decode=decode,
            deterministic=deterministic,
        )
        text_embeds, text_model_output = text_features["text_embeds"], text_features["text_model_output"]

        # temperature scaling
        logit_scale = jnp.exp(self.logit_scale)

        # logit bias is only used in chunked sigmoid loss
        logit_bias = self.logit_bias

        # normalize
        if not is_decoder:
            image_embeds = normalize(image_embeds)
            text_embeds = normalize(text_embeds)
            logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
            logits_per_image = logits_per_text.T
        else:
            logits_per_text = None
            logits_per_image = None

        return dict(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_model_output,
            vision_model_output=vision_model_output,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

    def get_text_features(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        decode: bool = False,
        deterministic: bool = True,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            decode=decode,
            deterministic=deterministic,
        )

        text_embeds = text_outputs["pooled_output"]
        if self.text_projection is not None:
            text_embeds = self.text_projection(text_embeds)

        return {"text_embeds": text_embeds, "text_model_output": text_outputs}

    def get_image_features(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )
        image_embeds = vision_outputs["pooled_output"]
        if self.vision_projection is not None:
            image_embeds = self.vision_projection(image_embeds)

        return {"image_embeds": image_embeds, "vision_model_output": vision_outputs}

    def init_inputs(config, rng: jax.random.PRNGKey):
        text_config = config.text_config
        vision_config = config.vision_config
        if isinstance(text_config, dict):
            text_config = SimpleNamespace(**text_config)
        if isinstance(vision_config, dict):
            vision_config = SimpleNamespace(**vision_config)
        input_ids = jnp.ones((1, text_config.max_length), dtype="i4")
        attention_mask = jnp.ones((1, text_config.max_length), dtype="i4")
        pixel_values = jnp.ones((1, vision_config.image_size, vision_config.image_size, 3), dtype="f4")
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return {"rngs": rngs, "input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}

    def init_weights(self, rng: jax.random.PRNGKey):
        inputs = self.init_inputs(rng)
        return self.init(**inputs)

    # Methods for FlaxGenerationMixin
    def prepare_inputs_for_generation(self, input_ids, max_length, encoder_outputs):
        # initialize cache
        model_inputs = self.init_inputs(jax.random.PRNGKey(0))
        bs = input_ids.shape[0]
        for k in ["input_ids", "attention_mask", "pixel_values"]:
            model_inputs[k] = jnp.repeat(model_inputs[k], bs, axis=0)
        _decode = partial(CLIPModel.__call__, decode=True)
        past_key_values = self.init(**model_inputs, method=_decode)["cache"]
        # for generation compatibility
        # - put batch before scan
        past_key_values = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        # - remove scan dimension from index
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = past_key_values["text"]["encoder"][
            "layers"
        ]["attention"]["cache_index"][0]
        return {"past_key_values": past_key_values, "encoder_outputs": encoder_outputs}

    def encode(self, input_ids, params, return_dict=True):
        res = self.apply(
            {"params": params},
            pixel_values=input_ids,
            deterministic=True,
            method=self.get_image_features,
        )["vision_model_output"]["last_hidden_state"]
        return EncoderOutput(last_hidden_state=res) if return_dict else res

    def decode(self, decoder_input_ids, encoder_outputs, params, past_key_values, return_dict=True):
        # for generation compatibility, apply inverse
        past_key_values = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        scan_dim = past_key_values["text"]["encoder"]["layers"]["attention"]["cached_key"].shape[0]
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = jnp.broadcast_to(
            past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"], (scan_dim,)
        )
        outputs, mutable = self.apply(
            {"params": params, "cache": past_key_values},
            mutable=["cache"],
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            decode=True,
            deterministic=True,
            method=self.get_text_features,
        )
        logits = outputs["text_model_output"]["last_hidden_state"]
        # for generation compatibility
        past_key_values = mutable["cache"]
        past_key_values = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = past_key_values["text"]["encoder"][
            "layers"
        ]["attention"]["cache_index"][0]
        return (
            DecoderOutput(logits=logits, past_key_values=past_key_values) if return_dict else (logits, past_key_values)
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs

    def generate(self, *args, **kwargs):
        assert self.text_config["is_decoder"], "generate() only works for decoder mode"
        generation_config = kwargs.pop("generation_config", None)
        if generation_config is None:
            generation_config = GenerationConfig(
                pad_token_id=self.text_config["pad_token_id"],
                eos_token_id=self.text_config["eos_token_id"],
                decoder_start_token_id=self.text_config["bos_token_id"],
                bos_token_id=self.text_config["bos_token_id"],
                max_length=self.text_config["max_length"],
            )
        return super().generate(*args, generation_config=generation_config, **kwargs)

    @classmethod
    def can_generate(cls):
        return True


def normalize(x, eps=1e-7, safe_norm=True):
    if safe_norm:
        return x * jax.lax.rsqrt(jnp.sum(jax.lax.square(x), axis=-1, keepdims=True) + eps)
    else:
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)
