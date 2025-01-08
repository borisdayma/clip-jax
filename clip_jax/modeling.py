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
from jax.ad_checkpoint import checkpoint_name
from transformers import FlaxGenerationMixin, GenerationConfig

from .maxtext import common_types
from .maxtext.inference_utils import sampling
from .maxtext.layers.models import Transformer
from .utils import with_logical_constraint

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


@flax.struct.dataclass
class DecodeState:
    cur_len: int
    is_sent_finished: Array
    sent_result: Array
    input_ids: Array
    attention_mask: Array
    position_ids: Array
    encoder_hidden_states: Array
    vision_start_ids: Array
    cache: Array


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


# Rotary Embeddings - Source: https://github.com/google/maxtext
class RotaryEmbedding(nn.Module):
    """RoPE

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """

    min_timescale: int = 1
    max_timescale: int = 10_000
    embedding_dims: int = 0

    def setup(self) -> None:
        if self.embedding_dims % 2:
            raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")

    def __call__(
        self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        inputs: jax.Array,
        position: jax.Array,
    ) -> jax.Array:
        """Generates a jax.Array of sinusoids with different frequencies.

        Args:
          inputs: The input sequence on which to apply the Rotary position
            embedding. Since rotary position embeddings are applied to query and
            keys after projection, it is assumed of shape [B, S, N, H].
          position: Optional position jax.Array which denotes the position of each
            token in the sequence. It is of shape [B, S].

        Returns:
          a jax.Array of shape [B, S, N, H] which includes the inputs together with
          the rotary position embedding incorporated in it.
        """
        assert position is not None
        if len(inputs.shape) != 4:
            raise ValueError("Input is assumed to be a rank 4 tensor of shape" "[batch, sequence, heads, dims].")
        if self.embedding_dims != inputs.shape[3]:
            raise ValueError(
                "The embedding dims of the rotary position embedding" "must match the hidden dimension of the inputs."
            )
        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        position = position[:, :, jnp.newaxis, jnp.newaxis]
        sinusoid_inp = position / timescale
        sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
        cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
        first_half, second_half = jnp.split(inputs, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        x_out = jnp.concatenate((first_part, second_part), axis=-1)
        return x_out


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
    position_embedding_shape: Optional[Tuple[int, int]]
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
            assert self.position_embedding_type == "learnt", "Position embedding must be learnt."
        if self.positiion_embedding_type == "learnt":
            assert self.position_embedding_shape is not None, "Position embedding shape must be provided."
        patch_embeds = nn.Conv(
            self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.lecun_normal(),
                ("conv_height", "conv_width", "input_channels", "embed"),
            ),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="patch_embeds",
        )(pixel_values)
        patch_embeds = with_logical_constraint(patch_embeds, ("batch", "height", "width", "embed"))
        batch_size, height, width, channels = patch_embeds.shape
        num_patches = height * width
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, num_patches, channels))
        patch_embeds = with_logical_constraint(patch_embeds, ("batch", "length", "embed"))
        if self.position_embedding_type == "learnt":
            num_positions = self.position_embedding_shape[0] * self.position_embedding_shape[1]
            position_height, position_width = (
                self.position_embedding_shape[0],
                self.position_embedding_shape[1],
            )
            if self.position_embedding_factorized:
                position_embeds_height = self.param(
                    "position_embeds_height",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)),
                        (None, "height", "embed"),
                    ),
                    (1, position_height, self.hidden_size),
                )
                position_embeds_width = self.param(
                    "position_embeds_width",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)),
                        (None, "width", "embed"),
                    ),
                    (1, position_width, self.hidden_size),
                )
                if position_height != height:
                    # interpolate
                    position_embeds_height = jax.image.resize(
                        position_embeds_height, (1, height, self.hidden_size), method="linear"
                    )
                    position_embeds_height = with_logical_constraint(
                        position_embeds_height, ("batch", "height", "embed")
                    )
                if position_width != width:
                    # interpolate
                    position_embeds_width = jax.image.resize(
                        position_embeds_width, (1, width, self.hidden_size), method="linear"
                    )
                    position_embeds_width = with_logical_constraint(position_embeds_width, ("batch", "width", "embed"))
                # make it 2d
                position_embeds_height = position_embeds_height[:, :, None, :]
                position_embeds_width = position_embeds_width[:, None, :, :]
                position_embeds = position_embeds_height + position_embeds_width
                assert (
                    position_embeds.shape
                    == (
                        1,
                        height,
                        width,
                        self.hidden_size,
                    )
                ), f"Position embeds shape: {position_embeds.shape}, expected: (1, {height}, {width}, {self.hidden_size})"
                position_embeds = jnp.reshape(position_embeds, (1, num_patches, self.hidden_size))
            else:
                position_embeds = self.param(
                    "position_embeds",
                    nn.with_logical_partitioning(
                        nn.initializers.normal(1 / np.sqrt(self.hidden_size)),
                        (None, "vocab", "embed"),
                    ),
                    (1, num_positions, self.hidden_size),
                )
                if num_positions != num_patches:
                    position_embeds = jnp.reshape(
                        position_embeds,
                        (
                            1,
                            self.position_embedding_shape[0],
                            self.position_embedding_shape[1],
                            self.hidden_size,
                        ),
                    )
                    position_embeds = with_logical_constraint(position_embeds, ("batch", "height", "width", "embed"))
                    # interpolate
                    position_embeds = jax.image.resize(position_embeds, (height, width), method="linear")
                    position_embeds = with_logical_constraint(position_embeds, ("batch", "height", "width", "embed"))
                    position_embeds = jnp.reshape(position_embeds, (1, num_patches, self.hidden_size))
        elif self.position_embedding_type == "sincos2d":
            position_embeds = posemb_sincos_2d(height, width, self.hidden_size, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown position embedding type {self.position_embedding_type}")
        embeddings = patch_embeds + position_embeds
        embeddings = with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.pool_type == "tok":
            cls_token = self.param(
                "cls_token",
                nn.with_logical_partitioning(nn.initializers.zeros_init(), (None, None, "embed")),
                (1, 1, self.hidden_size),
            )
            embeddings = jnp.concatenate([jnp.tile(cls_token, [batch_size, 1, 1]), embeddings], axis=1)
            embeddings = with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.registers:
            registers = self.param(
                "registers",
                nn.with_logical_partitioning(
                    nn.initializers.normal(1 / np.sqrt(self.hidden_size)),
                    (None, None, "embed"),
                ),
                (1, self.registers, self.hidden_size),
            )
            embeddings = jnp.concatenate([embeddings, jnp.tile(registers, [batch_size, 1, 1])], axis=1)
            embeddings = with_logical_constraint(embeddings, ("batch", "length", "embed"))
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
        embeddings = with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.position_embedding_type == "learnt":
            position_embeds = self.param(
                "position_embeds",
                nn.with_logical_partitioning(
                    nn.initializers.normal(1 / np.sqrt(embed_dim)),
                    (None, "vocab", "embed"),
                ),
                (1, self.max_length, embed_dim),
            )
            embeddings += position_embeds
            embeddings = with_logical_constraint(embeddings, ("batch", "length", "embed"))
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
    embed_dim_name: str = "embed"
    normalize_qk: bool = False
    kernel_init_out: Optional[Callable[[PRNGKey, Shape, Dtype], Array]] = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        position_ids: Optional[Array] = None,
        deterministic: Optional[bool] = None,
        sow_weights: Optional[bool] = False,
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
                query = with_logical_constraint(query, ("batch", "length", "heads", "kv"))
                key = with_logical_constraint(key, ("batch", "length", "heads", "kv"))
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
            query = with_logical_constraint(query, ("batch", "length", "heads", "kv"))
            key = with_logical_constraint(key, ("batch", "length", "heads", "kv"))
            value = with_logical_constraint(value, ("batch", "length", "heads", "kv"))

            if self.use_rotary:
                if position_ids is not None:
                    query_positions = position_ids
                    key_positions = position_ids
                else:
                    query_positions = jnp.arange(query.shape[1])
                    query_positions = jnp.broadcast_to(query_positions[None], query.shape[:2])
                    key_positions = jnp.arange(key.shape[1])
                    key_positions = jnp.broadcast_to(key_positions[None], key.shape[:2])

                key = RotaryEmbedding(embedding_dims=head_dim, name="key_rotary")(inputs=key, position=key_positions)
                query = RotaryEmbedding(embedding_dims=head_dim, name="query_rotary")(
                    inputs=query, position=query_positions
                )
                # convert to correct type
                query = query.astype(self.dtype)
                key = key.astype(self.dtype)

                # ensure sharding
                query = with_logical_constraint(query, ("batch", "length", "heads", "kv"))
                key = with_logical_constraint(key, ("batch", "length", "heads", "kv"))

            # checkpoint policies
            query = checkpoint_name(query, "query_proj")
            key = checkpoint_name(key, "key_proj")
            value = checkpoint_name(value, "value_proj")

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
                    # update key, value caches with our new 1d spatial slices
                    cur_index = cache_index.value
                    indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
                    key = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
                    value = jax.lax.dynamic_update_slice(cached_value.value, value, indices)
                    cached_key.value = key
                    cached_value.value = value
                    num_updated_cache_vectors = query.shape[1]
                    cache_index.value = cache_index.value + num_updated_cache_vectors

                    # causal mask for cached decoder self-attention:
                    # our single query position should only attend to those key
                    # positions that have already been generated and cached,
                    # not the remaining zero elements.
                    causal_mask = jnp.broadcast_to(
                        jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                        tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
                    )
                    if mask is not None:
                        mask = jax.lax.dynamic_slice(mask, (0, 0, cur_index, 0), causal_mask.shape)
                    mask = nn.combine_masks(mask, causal_mask)

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
            # NOTE: we compute both attention weights and logits in float32 with float32_logits
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
                module=self if sow_weights else None,
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

    num_queries: int
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
    def __call__(self, x, mask, deterministic: bool = False):
        batch, length, embed_dim = x.shape
        probe = self.param(
            "probe",
            nn.with_logical_partitioning(nn.initializers.xavier_uniform(), (None, None, "embed")),
            (1, self.num_queries, embed_dim),
        )
        probe = jnp.tile(probe, [batch, self.num_queries, 1])
        probe = with_logical_constraint(probe, ("batch", None, "embed"))
        if mask is not None:
            mask = nn.make_attention_mask(jnp.ones((batch, 1), dtype="i4"), mask, dtype=self.dtype)
        x = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.use_bias,
            use_rotary=False,
            dropout_rate=self.attention_dropout,
            decode=False,
            normalize_qk=self.normalize_qk,
            float32_logits=self.float32_logits,
            kernel_init_out=default_kernel_init,
            name="attention",
        )(inputs_q=probe, inputs_kv=x, mask=mask, deterministic=deterministic)
        x = with_logical_constraint(x, ("batch", "length", "embed"))
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
        y = with_logical_constraint(y, ("batch", "length", "embed"))
        x = x + y
        x = with_logical_constraint(x, ("batch", "length", "embed"))
        return x


class CLIPMLP(nn.Module):
    mlp_dim: int
    ln_type: str  # "preln", "normformer"
    activations: Sequence[Union[str, Callable]] = ("relu",)
    mlp_dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    use_bias: bool = False
    force_scale: bool = False
    use_rmsnorm: bool = True
    kernel_init_out: Optional[Callable[[PRNGKey, Shape, Dtype], Array]] = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        assert self.ln_type in [
            "normformer",
            "preln",
        ], f"ln_type {self.ln_type} not supported."
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
                inputs = with_logical_constraint(inputs, ("batch", "length", "embed"))
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
                x = with_logical_constraint(x, ("batch", "length", "mlp"))
                x = _convert_to_activation_function(act_fn)(x)
                x = with_logical_constraint(x, ("batch", "length", "mlp"))
                activations.append(x)
            # Take elementwise product of above intermediate activations.
            x = functools.reduce(operator.mul, activations)
            x = with_logical_constraint(x, ("batch", "length", "mlp"))

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
                x = with_logical_constraint(x, ("batch", "length", "mlp"))
            # Apply dropout and final dense output projection.
            x = nn.Dropout(rate=self.mlp_dropout_rate, broadcast_dims=(-2,), name="mlp_dropout")(
                x, deterministic=deterministic
            )  # Broadcast along length.
            x = with_logical_constraint(x, ("batch", "length", "mlp"))
            output = nn.DenseGeneral(
                embed_dim,
                dtype=self.dtype,
                use_bias=self.use_bias,
                kernel_init=nn.with_logical_partitioning(self.kernel_init_out, ("mlp", "embed")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                name="wo",
            )(x)
            output = with_logical_constraint(output, ("batch", "length", "embed"))
            return output


class CLIPEncoderLayer(nn.Module):
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    position_embedding_type: str  # "learnt", "rotary" or "sincos2d"
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
        position_ids: Optional[Array] = None,
        deterministic: bool = True,
    ):
        assert self.ln_type in [
            "normformer",
            "preln",
        ], f"ln_type {self.ln_type} not supported."
        assert self.position_embedding_type in [
            "learnt",
            "rotary",
            "sincos2d",
        ], f"position_embedding_type {self.position_embedding_type} not supported."

        # Self attention
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
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
            hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.use_bias,
            use_rotary=(self.position_embedding_type == "rotary"),
            dropout_rate=self.attention_dropout,
            decode=self.decode,
            normalize_qk=self.normalize_qk,
            float32_logits=self.float32_logits,
            name="attention",
        )(
            inputs_q=hidden_states,
            inputs_kv=hidden_states,
            mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        if self.ln_type == "normformer":
            hidden_states = norm(self.use_rmsnorm)(
                dtype=self.dtype,
                use_bias=self.use_bias,
                use_scale=True,
                scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                name="post_attention_norm",
            )(hidden_states)
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = residual + hidden_states
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))

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
                hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            hidden_states = MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dtype=self.dtype,
                deterministic=deterministic,
                use_bias=self.use_bias,
                use_rotary=False,  # don't apply on cross-attention
                dropout_rate=self.attention_dropout,
                decode=False,
                normalize_qk=self.normalize_qk,
                float32_logits=self.float32_logits,
                name="cross_attention",
            )(
                inputs_q=hidden_states,
                inputs_kv=encoder_hidden_states,
                deterministic=deterministic,
            )
            hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            if self.ln_type == "normformer":
                hidden_states = norm(self.use_rmsnorm)(
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    use_scale=True,
                    scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
                    name="post_cross_attention_norm",
                )(hidden_states)
            hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
            hidden_states = residual + hidden_states
            hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))

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
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        hidden_states = residual + hidden_states
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        return hidden_states, None


class CLIPEncoder(nn.Module):
    num_layers: int
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    position_embedding_type: str  # "learnt", "rotary"
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
    remat_policy: str = "none"

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        position_ids=None,
        deterministic=True,
    ):
        # gradient checkpointing
        use_scan = True
        initializing = self.is_mutable_collection("params")
        params_spec = 0 if initializing else ScanIn(0)
        layer = CLIPEncoderLayer
        if self.remat_policy != "none":
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            elif self.remat_policy == "proj":
                policy = jax.checkpoint_policies.save_only_these_names("query_proj", "value_proj", "key_proj")
            elif self.remat_policy == "minimal_offloaded":
                policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(
                    offload_src="device", offload_dst="pinned_host"
                )
            else:
                assert self.remat_policy == "full", "Remat policy needs to be on list of remat policies"
                policy = None
            layer = nn.remat(
                layer,
                prevent_cse=not use_scan,
                policy=policy,
                static_argnums=(-1, -2, -3, -4, -5),
            )

        hidden_states, _ = nn.scan(
            layer,
            variable_axes={"params": params_spec, "cache": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.num_layers,
            unroll=self.unroll,
            metadata_params={nn.PARTITION_NAME: "layers"},
        )(
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
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
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            position_ids,
            deterministic,
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
    remat_policy: str = "none"
    eos_token_id: int = None
    mask_token_id: int = None
    pad_token_id: int = None
    bos_token_id: int = None
    masked_pred_prob: float = 0.75  # recommended by Cappa
    is_decoder: bool = False  # for Cappa
    dtype: str = "float32"
    pool_type: str = None  # "bos", "eos", "map"
    num_queries: int = 1  # used for map

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states=None,
        position_ids=None,
        decode=False,
        deterministic: bool = True,
    ):
        if self.is_decoder:
            assert encoder_hidden_states is not None
            assert self.mask_token_id is not None
            assert self.use_causal_mask
        else:
            assert encoder_hidden_states is None
        if self.use_causal_mask or self.pool_type in ["map", "eos"]:
            assert self.eos_token_id is not None

        # attention mask
        input_attention_mask = attention_mask
        if attention_mask is not None:
            attention_mask = nn.make_attention_mask(attention_mask, attention_mask, dtype=self.dtype)
        if self.use_causal_mask and not decode:
            causal_mask = nn.make_causal_mask(input_ids)
            if attention_mask is None:
                attention_mask = causal_mask
            else:
                attention_mask = nn.combine_masks(attention_mask, causal_mask)

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
            remat_policy=self.remat_policy,
            decode=decode,
            name="encoder",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        last_hidden_state = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="final_norm",
        )(last_hidden_state)
        last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        if self.is_decoder:
            assert self.pool_type is None, "pool_type is ignored in decoder mode"
            # dense to vocab
            last_hidden_state = nn.Dense(
                self.vocab_size,
                dtype=dtype,
                kernel_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed", "vocab")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("vocab",)),
                name="logits",
            )(last_hidden_state)
            last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "vocab"))
            pooled_output = None

        else:
            if self.use_causal_mask:
                assert not self.pool_type == "bos", "pool_type = 'bos' does not make sense with causal mask"
            if self.pool_type == "bos":
                pooled_output = last_hidden_state[:, 0, :]
            elif self.pool_type == "eos":

                def _get_id_pos(mat, id):
                    return jnp.where(mat == id, 1, 0).argmax(axis=-1)

                pooled_output = last_hidden_state[
                    jnp.arange(last_hidden_state.shape[0]),
                    _get_id_pos(input_ids, self.eos_token_id),
                ]
            elif self.pool_type == "map":
                # TODO: should allow custom heads/dim when using as queries for llm
                pooled_output = MAPHead(
                    num_queries=self.num_queries,
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
                )(last_hidden_state, input_attention_mask, deterministic=deterministic)
                if self.num_queries == 1:
                    # used for clip
                    pooled_output = pooled_output[:, 0]

            else:
                pooled_output = None
            if pooled_output is not None:
                pooled_output = with_logical_constraint(pooled_output, ("batch", "embed"))

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
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
    position_embedding_shape: Optional[Tuple[int, int]] = (16, 16)  # e.g. (16, 16) for 256x256 images with patch 16
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
    remat_policy: str = "none"
    num_queries: int = 1  # used for map

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
        hidden_states = with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        encoder_outputs = CLIPEncoder(
            num_layers=self.num_layers,
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
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
            remat_policy=self.remat_policy,
            name="encoder",
        )(
            hidden_states=hidden_states,
            deterministic=deterministic,
        )

        # get last hidden state
        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # remove registers
        if self.registers and not self.keep_registers:
            if self.pool_type is None or self.pool_type == "tok":
                print(f"Warning: removing registers with pool_type = {self.pool_type} does not seem necessary.")
            last_hidden_state = last_hidden_state[:, : -self.registers]
            last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # layernorm
        last_hidden_state = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="final_norm",
        )(last_hidden_state)
        last_hidden_state = with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        if self.pool_type == "tok":
            pooled_output = last_hidden_state[:, 0, :]

        elif self.pool_type == "gap":
            # mean pool - jnp.mean -> was leading to large memory consumption in the past
            pooled_output = jnp.mean(last_hidden_state, axis=1)

        elif self.pool_type == "map":
            pooled_output = MAPHead(
                num_queries=self.num_queries,
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
            )(last_hidden_state, None, deterministic=deterministic)
            if self.num_queries == 1:
                # used for clip
                pooled_output = pooled_output[:, 0]

        elif self.pool_type is None:
            pooled_output = None

        else:
            raise ValueError(f"pool_type {self.pool_type} not supported.")

        if pooled_output is not None:
            pooled_output = with_logical_constraint(pooled_output, ("batch", "embed"))

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPVisionModelForFineTuning(nn.Module):
    vision_config: Any
    dtype: str = "float32"
    maxtext_mesh: Any = None
    maxtext_args: Any = None

    def __post_init__(self):
        # add default fields vision_config
        assert self.maxtext_mesh is None, "maxtext_mesh should not be set for classification"
        assert self.maxtext_args is None, "maxtext_args should not be set for classification"
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
        outputs = CLIPVisionTransformer(
            **self.vision_config,
            name="vision",
        )(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )
        return outputs

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


class CLIPVisionModelForImageClassification(nn.Module):
    vision_config: Any
    num_labels: int
    dtype: str = "float32"
    maxtext_mesh: Any = None
    maxtext_args: Any = None

    def __post_init__(self):
        # add default fields vision_config
        assert self.maxtext_mesh is None, "maxtext_mesh should not be set for classification"
        assert self.maxtext_args is None, "maxtext_args should not be set for classification"
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
            kernel_init=nn.with_logical_partitioning(nn.initializers.zeros, ("embed", "classifier")),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("classifier",)),
            name="vision_projection",
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

    def __post_init__(self):
        # add default fields vision_config
        default_fields = dataclasses.fields(CLIPTextTransformer)
        default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
        default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
        text_config = {**default_fields, **self.text_config}
        if self.dtype is not None:
            text_config["dtype"] = self.dtype
        self.text_config = text_config
        return super().__post_init__()

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
    ):
        text_outputs = CLIPTextTransformer(**self.text_config, name="text")(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
        )
        return text_outputs

    def init_inputs(config, rng: jax.random.PRNGKey):
        text_config = config.text_config
        if isinstance(text_config, dict):
            text_config = SimpleNamespace(**text_config)
        input_ids = jnp.ones((1, text_config.max_length), dtype="i4")
        attention_mask = jnp.ones((1, text_config.max_length), dtype="i4")
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return {"rngs": rngs, "input_ids": input_ids, "attention_mask": attention_mask}

    def init_weights(self, rng: jax.random.PRNGKey):
        inputs = self.init_inputs(rng)
        return self.init(**inputs)


class CLIPModel(nn.Module, FlaxGenerationMixin):
    text_config: Any
    vision_config: Any
    projection_dim: int
    logit_scale_init_value: float = 2.6592
    logit_bias_init_value: float = 0.0
    dtype: str = "float32"
    # for maxtext models
    maxtext_mesh: Any = None
    maxtext_args: Any = None

    def __post_init__(self):
        # add default fields text_config
        if self.maxtext_mesh is None:
            # regular model
            default_fields = dataclasses.fields(CLIPTextTransformer)
            default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
            default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
            text_config = {**default_fields, **self.text_config}
        else:
            # maxtext model, accepts extra args for customization (mesh spec, etc)
            maxtext_args = self.maxtext_args or {}
            text_config = {**self.text_config, **maxtext_args}
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
        self.config = SimpleNamespace(
            is_encoder_decoder=text_config.get("is_decoder") or self.maxtext_mesh is not None
        )
        return super().__post_init__()

    def setup(self):
        if self.config.is_encoder_decoder:
            assert (self.vision_config["pool_type"] is None) or (
                self.vision_config["pool_type"] == "map" and self.vision_config["num_queries"] > 1
            ), "pool_type must be None or map with num_queries > 1 for decoder mode without maxtext"
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
        if self.maxtext_mesh is None:
            self.text_model = CLIPTextTransformer(
                **self.text_config,
                name="text",
            )
        else:
            self.text_model = Transformer(SimpleNamespace(**self.text_config), self.maxtext_mesh, quant=None)
        self.vision_model = CLIPVisionTransformer(
            **self.vision_config,
            name="vision",
        )
        if (not self.text_config.get("is_decoder", True)) and (
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
        position_ids: Optional[jnp.ndarray] = None,
        decode: bool = False,
        deterministic: bool = True,
        vision_start_ids: Optional[Any] = None,
        model_mode: str = common_types.MODEL_MODE_TRAIN,
    ):
        image_features = self.get_image_features(pixel_values, deterministic=deterministic)
        image_embeds, vision_model_output = (
            image_features["image_embeds"],
            image_features["vision_model_output"],
        )

        is_decoder = self.text_config.get("is_decoder", True)
        if decode:
            assert is_decoder, "decode=True only works for decoder mode"
        text_features = self.get_text_features(
            input_ids,
            attention_mask,
            encoder_hidden_states=vision_model_output["last_hidden_state"] if is_decoder else None,
            position_ids=position_ids,
            vision_start_ids=vision_start_ids,
            decode=decode,
            deterministic=deterministic,
            model_mode=model_mode,
        )
        text_embeds, text_model_output = (
            text_features["text_embeds"],
            text_features["text_model_output"],
        )

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
        position_ids: Optional[jnp.ndarray] = None,
        vision_start_ids: Optional[Any] = None,
        decode: bool = False,
        deterministic: bool = True,
        model_mode: str = common_types.MODEL_MODE_TRAIN,
    ):
        if self.maxtext_mesh is None:
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                position_ids=position_ids,
                decode=decode,
                deterministic=deterministic,
            )

            text_embeds = text_outputs["pooled_output"]
            if self.text_projection is not None:
                text_embeds = self.text_projection(text_embeds)
        else:
            text_embeds = None
            last_hidden_state = self.text_model(
                decoder_input_tokens=input_ids,
                decoder_positions=position_ids,
                decoder_segment_ids=attention_mask,
                enable_dropout=not deterministic,
                vision_embeddings=encoder_hidden_states,
                vision_start_ids=vision_start_ids,
                model_mode=model_mode,
            )
            text_outputs = dict(
                last_hidden_state=last_hidden_state,
                pooled_output=None,
            )

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

    def init_inputs(config, rng: jax.random.PRNGKey, batch_size=1, max_length=512):
        text_config = config.text_config
        vision_config = config.vision_config
        if isinstance(text_config, dict):
            text_config = SimpleNamespace(**text_config)
        if isinstance(vision_config, dict):
            vision_config = SimpleNamespace(**vision_config)
        max_len = getattr(text_config, "max_length", max_length)
        input_ids = jnp.ones((batch_size, max_len), dtype="i4")
        pixel_values = jnp.ones(
            (batch_size, vision_config.image_size, vision_config.image_size, 3),
            dtype="f4",
        )
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng, "aqt": params_rng}
        return {
            "rngs": rngs,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": jnp.ones_like(input_ids),
            "position_ids": jnp.ones_like(input_ids),
            "vision_start_ids": 0,
        }

    def init_weights(self, rng: jax.random.PRNGKey):
        inputs = self.init_inputs(rng)
        return self.init(**inputs)

    # Methods for FlaxGenerationMixin
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        max_length,
        encoder_outputs,
        decoder_attention_mask=None,
    ):
        # initialize cache
        model_inputs = self.init_inputs(jax.random.PRNGKey(0))
        bs = decoder_input_ids.shape[0]
        for k in ["input_ids", "attention_mask", "pixel_values"]:
            model_inputs[k] = jnp.repeat(model_inputs[k], bs, axis=0)
        _decode = partial(CLIPModel.__call__, decode=True)
        past_key_values = self.init(**model_inputs, method=_decode)["cache"]
        # extend attention mask
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            position_ids = jnp.where(decoder_attention_mask == 0, 0, position_ids)
            extended_mask = jnp.ones((bs, max_length), dtype="i4")
            decoder_attention_mask = jax.lax.dynamic_update_slice(extended_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.arange(decoder_input_ids.shape[1])
            position_ids = jnp.broadcast_to(position_ids[None], decoder_input_ids.shape[:2])
        # for generation compatibility
        # - put batch before scan
        past_key_values = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        # - remove scan dimension from index
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = past_key_values["text"]["encoder"][
            "layers"
        ]["attention"]["cache_index"][0]
        return {
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": position_ids,
        }

    def encode(self, input_ids, params, return_dict=True):
        res = self.apply(
            {"params": params},
            pixel_values=input_ids,
            deterministic=True,
            method=self.get_image_features,
        )["vision_model_output"]["last_hidden_state"]
        return EncoderOutput(last_hidden_state=res) if return_dict else res

    def decode(
        self,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_position_ids,
        encoder_outputs,
        params,
        past_key_values,
        return_dict=True,
        vision_start_ids=None,
    ):
        # for generation compatibility, apply inverse
        past_key_values = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        scan_dim = past_key_values["text"]["encoder"]["layers"]["attention"]["cached_key"].shape[0]
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = jnp.broadcast_to(
            past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"],
            (scan_dim,),
        )
        outputs, mutable = self.apply(
            {"params": params, "cache": past_key_values},
            mutable=["cache"],
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            vision_start_ids=vision_start_ids,
            decode=True,
            deterministic=True,
            method=self.get_text_features,
        )
        logits = outputs["text_model_output"]["last_hidden_state"]
        # for generation compatibility
        past_key_values = mutable["cache"]
        past_key_values = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1) if x.ndim >= 2 else x, past_key_values)
        past_key_values["text"]["encoder"]["layers"]["attention"]["cache_index"] = past_key_values["text"]["encoder"][
            "layers"
        ]["attention"]["cache_index"][0]
        return (
            DecoderOutput(logits=logits, past_key_values=past_key_values) if return_dict else (logits, past_key_values)
        )

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["decoder_position_ids"] = model_kwargs["decoder_position_ids"][:, -1:] + 1
        return model_kwargs

    def generate(
        self,
        *args,
        # for maxtext only
        decode_sampling_strategy="greedy",
        decode_sampling_nucleus_p=-1,
        decode_sampling_top_k=0,
        decode_sampling_temperature=1.0,
        decode_force_max_length=False,  # for benchmarking
        **kwargs,
    ):
        assert self.text_config.get("is_decoder", True), "generate() only works for decoder mode"
        if self.maxtext_mesh is None:
            generation_config = kwargs.pop("generation_config", None)
            if generation_config is None:
                generation_config = GenerationConfig(
                    pad_token_id=self.text_config["pad_token_id"],
                    eos_token_id=self.text_config["eos_token_id"],
                    decoder_start_token_id=self.text_config["bos_token_id"],
                    bos_token_id=self.text_config["bos_token_id"],
                    max_length=kwargs.get("max_length", self.text_config["max_length"]),
                )
            return super().generate(*args, generation_config=generation_config, **kwargs)
        else:
            # get relevant args
            params = kwargs["params"]
            input_ids = kwargs["input_ids"]
            pixel_values = kwargs["pixel_values"]
            attention_mask = kwargs["attention_mask"]
            vision_start_ids = kwargs["vision_start_ids"]
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                if attention_mask is not None:
                    # NOTE: does not support packed sequences
                    print("Setting decoder position ids based on attention mask")
                    position_ids = attention_mask.cumsum(axis=-1) - 1
                    position_ids = jnp.where(attention_mask == 0, 0, position_ids)
                else:
                    print("Setting decoder position ids based on input tokens")
                    position_ids = jnp.arange(input_ids.shape[1])
                    position_ids = position_ids[None, :]
            rng = kwargs["rng"]
            batch_size = pixel_values.shape[0]
            max_prefill_length = input_ids.shape[1]
            if self.text_config["max_prefill_predict_length"] != max_prefill_length:
                print(
                    f"""Warning: max_prefill_predict_length != input shape: {self.text_config["max_prefill_predict_length"]} != {max_prefill_length}"""
                )
            max_generate = self.text_config["max_target_length"] - max_prefill_length
            pad_token_id = self.text_config["pad_token_id"]
            eos_token_id = self.text_config["eos_token_id"]

            # get encoder outputs
            encoder_hidden_states = self.encode(pixel_values, params, return_dict=False)

            # set initial_state
            state = DecodeState(
                cur_len=0,
                is_sent_finished=jnp.zeros((batch_size,), dtype=jnp.bool_),
                sent_result=jnp.full((batch_size, max_generate), pad_token_id, dtype=jnp.int32),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                encoder_hidden_states=encoder_hidden_states,
                vision_start_ids=vision_start_ids,
                cache=None,
            )

            # decode one step
            def _decode_one_step(
                state,
                model_mode,
            ):
                var_params = {"params": params}
                if state.cache is not None:
                    var_params["cache"] = state.cache
                outputs, mutable = self.apply(
                    var_params,
                    mutable=["cache"],
                    input_ids=state.input_ids,
                    attention_mask=state.attention_mask,
                    position_ids=state.position_ids,
                    encoder_hidden_states=state.encoder_hidden_states,
                    vision_start_ids=state.vision_start_ids,
                    deterministic=True,
                    model_mode=model_mode,
                    method=self.get_text_features,
                )
                logits = outputs["text_model_output"]["last_hidden_state"]

                # sample token
                logits = logits[:, -1:]
                next_token = sampling(
                    logits,
                    rng,
                    decode_sampling_strategy,
                    topk=decode_sampling_top_k,
                    nucleus_topp=decode_sampling_nucleus_p,
                    temperature=decode_sampling_temperature,
                )
                next_token = (
                    next_token * ~state.is_sent_finished[:, None] + pad_token_id * state.is_sent_finished[:, None]
                )
                state = state.replace(
                    cur_len=state.cur_len + 1,
                    is_sent_finished=state.is_sent_finished | (next_token[..., 0] == eos_token_id),
                    sent_result=jax.lax.dynamic_update_slice(state.sent_result, next_token, (0, state.cur_len)),
                    input_ids=next_token,
                    attention_mask=None,
                    position_ids=state.position_ids[:, -1:] + 1,
                    encoder_hidden_states=None,
                    vision_start_ids=None,
                    cache=mutable["cache"],
                )
                return state

            # prefill
            state = _decode_one_step(
                state,
                model_mode=common_types.MODEL_MODE_PREFILL,
            )

            # generate
            def _cond_fn(state):
                max_len_reached = state.cur_len >= max_generate
                if decode_force_max_length:
                    return ~max_len_reached
                all_sent_finished = jnp.all(state.is_sent_finished)
                return ~(all_sent_finished | max_len_reached)

            state = jax.lax.while_loop(
                _cond_fn,
                partial(_decode_one_step, model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE),
                state,
            )
            return state.sent_result

    @classmethod
    def can_generate(cls):
        return True


def normalize(x, eps=1e-6, safe_norm=True):
    if safe_norm:
        return x * jax.lax.rsqrt(jnp.sum(jax.lax.square(x), axis=-1, keepdims=True) + eps)
    else:
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)
