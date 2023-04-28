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
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.linear import DenseGeneral, DotGeneralT, PrecisionLike
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes
from flax.linen.partitioning import remat
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers.modeling_flax_utils import FlaxPreTrainedModel

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


# Adapted from flax.linen.normalization
def _normalize(
    mdl: Module,
    x: Any,
    mean: Any,
    var: Any,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Callable,
    scale_init: Callable,
):
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    if mean is not None:
        mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean if mean is not None else x
    mul = jax.lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param("scale", scale_init, reduced_feature_shape, param_dtype).reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if use_bias:
        bias = mdl.param("bias", bias_init, reduced_feature_shape, param_dtype).reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


nn.LayerNorm


class RMSNorm(nn.Module):
    """RMSNorm, adapted from flax LayerNorm"""

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = nn.initializers.zeros_init
    scale_init: Callable = nn.initializers.ones_init
    reduction_axes: Axes = -1
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x):
        dtype = self.dtype or jnp.result_type(x)
        # promote x to at least float32, this avoids half precision computation
        # but preserves double or complex floating points
        dtype = jnp.promote_types(dtype, jnp.float32)
        x = jnp.asarray(x, dtype)
        # use mean2 instead of variance (not centered)
        var = jnp.mean(jax.lax.square(x), self.reduction_axes)
        mean = None

        if self.axis_name is not None:
            var = jax.lax.pmean(var, axis_name=self.axis_name, axis_index_groups=self.axis_index_groups)

        return _normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


def norm(use_rmsnorm):
    """Normalization wrapper"""
    if use_rmsnorm:
        return RMSNorm
    else:
        return nn.LayerNorm


class CLIPVisionEmbeddings(nn.Module):
    hidden_size: int
    use_bias: bool
    patch_size: int
    dtype: Dtype

    @nn.compact
    def __call__(self, pixel_values):
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
        # learnt position embeddings
        position_embeds = self.param(
            "position_embeds",
            nn.with_logical_partitioning(
                nn.initializers.normal(1 / np.sqrt(self.hidden_size)), (None, "vocab", "embed")
            ),
            (1, num_patches, self.hidden_size),
        )
        embeddings = patch_embeds + position_embeds
        embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    position_embedding_type: str  # "absolute" or "rotary"
    dtype: Dtype

    @nn.compact
    def __call__(self, input_ids):
        assert self.position_embedding_type in ["absolute", "rotary"]
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
        if self.position_embedding_type == "absolute":
            position_embeds = self.param(
                "position_embeds",
                nn.with_logical_partitioning(nn.initializers.normal(1 / np.sqrt(embed_dim)), (None, "vocab", "embed")),
                (1, self.max_position_embeddings, embed_dim),
            )
            embeddings += position_embeds
            embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        return embeddings


class MultiHeadDotProductAttention(Module):
    """
    Adapted from nn.MultiHeadDotProductAttention:
    * - support use_rotary
    """

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros_init()
    use_bias: bool = True
    attention_fn: Callable[..., Array] = dot_product_attention
    decode: bool = False
    qkv_dot_general: DotGeneralT = jax.lax.dot_general
    out_dot_general: DotGeneralT = jax.lax.dot_general
    use_rotary: bool = False
    max_length: Optional[int] = None  # required if use_rotary

    @compact
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

        with jax.profiler.TraceAnnotation("Attention"):
            dense = functools.partial(
                DenseGeneral,
                axis=-1,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                features=(self.num_heads, head_dim),
                kernel_init=nn.with_logical_partitioning(self.kernel_init, ("embed", "heads", "kv")),
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

            if self.use_rotary:
                assert self.max_length is not None, "max_length must be specified for rotary embeddings."
                # source: https://github.com/google-research/jestimator/blob/main/jestimator/models/rope/modeling.py
                sin, cos = generate_fixed_pos_embedding(head_dim, self.max_length)
                query, key = apply_rotary_embedding(query, key, cos, sin)

            # query = nn.with_logical_constraint(query, ("batch", "length", "heads", "kv"))
            # key = nn.with_logical_constraint(key, ("batch", "length", "heads", "kv"))
            # value = nn.with_logical_constraint(value, ("batch", "length", "heads", "kv"))

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
                    mask = combine_masks(
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
            out = DenseGeneral(
                features=features,
                axis=(-2, -1),
                kernel_init=nn.with_logical_partitioning(self.kernel_init, ("heads", "kv", "embed")),
                bias_init=nn.with_logical_partitioning(self.bias_init, ("embed",)),
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                dot_general=self.out_dot_general,
                name="out",  # type: ignore[call-arg]
            )(x)
            return out


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
        with jax.profiler.TraceAnnotation("MLP"):
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
                x = DenseGeneral(
                    self.mlp_dim,
                    dtype=self.dtype,
                    use_bias=self.use_bias,
                    kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "mlp")),
                    bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init, ("mlp",)),
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
            output = DenseGeneral(
                inputs.shape[-1],
                dtype=self.dtype,
                use_bias=self.use_bias,
                kernel_init=nn.with_logical_partitioning(default_kernel_init, ("mlp", "embed")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init, ("embed",)),
                name="wo",
            )(x)
            output = nn.with_logical_constraint(output, ("batch", "length", "embed"))
            return output


class CLIPEncoderLayer(nn.Module):
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    position_embedding_type: str  # "absolute", "rotary"
    max_position_embeddings: int
    use_causal_mask: bool
    mlp_dim: int
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
    ):
        assert self.ln_type in ["normformer", "preln"], f"ln_type {self.ln_type} not supported."
        assert self.position_embedding_type in ["absolute", "rotary"]
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
        # Reshape attention mask for direct use in attention heads
        if attention_mask is not None:
            attention_mask = nn.attention.make_attention_mask(attention_mask, attention_mask, dtype=self.dtype)
        hidden_states = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.use_bias,
            use_rotary=(self.position_embedding_type == "rotary"),
            max_length=self.max_position_embeddings,
            dropout_rate=self.attention_dropout,
            decode=self.use_causal_mask,
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
    position_embedding_type: str  # "absolute", "rotary"
    max_position_embeddings: int
    use_causal_mask: bool
    mlp_dim: int
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    unroll: int = 100  # unroll scan layers
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
    ):
        # gradient checkpointing
        use_scan = True
        layer = (
            remat(
                CLIPEncoderLayer,
                static_argnums=(2,),
                prevent_cse=not use_scan,
            )
            if self.gradient_checkpointing
            else CLIPEncoderLayer
        )

        hidden_states, _ = nn.scan(
            layer,
            variable_axes={"params": 0, "cache": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast),
            length=self.num_layers,
            unroll=self.unroll,
            metadata_params={nn.PARTITION_NAME: "layer"},
        )(
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=self.position_embedding_type,
            max_position_embeddings=self.max_position_embeddings,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            activations=self.activations,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            attention_dropout=self.attention_dropout,
            mlp_dropout_rate=self.mlp_dropout_rate,
            name="layers",
        )(
            hidden_states, attention_mask, deterministic
        )

        return dict(
            last_hidden_state=hidden_states,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPTextTransformer(nn.Module):
    hidden_size: int
    vocab_size: int
    max_position_embeddings: int
    position_embedding_type: str  # "absolute" or "rotary"
    num_layers: int
    use_rmsnorm: bool
    ln_type: str  # "preln", "normformer"
    num_heads: int
    use_causal_mask: bool
    mlp_dim: int
    dtype: Dtype = jnp.float32
    activations: Sequence[Union[str, Callable]] = ("relu",)
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    unroll: int = 100  # unroll scan layers
    gradient_checkpointing: bool = True
    eos_token_id: int = -1
    dtype: str = "float32"

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
    ):
        dtype = getattr(jnp, self.dtype)
        hidden_states = CLIPTextEmbeddings(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
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
            max_position_embeddings=self.max_position_embeddings,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            dtype=dtype,
            activations=self.activations,
            use_bias=self.use_bias,
            force_scale=self.force_scale,
            attention_dropout=self.attention_dropout,
            mlp_dropout_rate=self.mlp_dropout_rate,
            unroll=self.unroll,
            gradient_checkpointing=self.gradient_checkpointing,
            name="encoder",
        )(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
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
    dtype: str = "float32"
    activations: Sequence[Union[str, Callable]] = ("relu",)
    use_bias: bool = False
    force_scale: bool = False
    attention_dropout: float = 0.0
    mlp_dropout_rate: float = 0.0
    unroll: int = 100  # unroll scan layers
    gradient_checkpointing: bool = True

    @nn.compact
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        dtype = getattr(jnp, self.dtype)
        batch, height, width, channels = pixel_values.shape
        assert (
            height == self.image_size and width == self.image_size and channels == 3
        ), f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
        position_embedding_type = "absolute"  # for vision it makes more sense than rotary
        hidden_states = CLIPVisionEmbeddings(
            hidden_size=self.hidden_size,
            use_bias=self.use_bias,
            patch_size=self.patch_size,
            dtype=dtype,
            name="embeddings",
        )(pixel_values)
        hidden_states = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="post_embed_norm",
        )(hidden_states)
        # hidden_states = nn.with_logical_constraint(hidden_states, ("batch", "length", "embed"))
        max_position_embeddings = hidden_states.shape[1]
        encoder_outputs = CLIPEncoder(
            num_layers=self.num_layers,
            use_rmsnorm=self.use_rmsnorm,
            ln_type=self.ln_type,
            num_heads=self.num_heads,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=max_position_embeddings,
            use_causal_mask=self.use_causal_mask,
            mlp_dim=self.mlp_dim,
            dtype=dtype,
            activations=self.activations,
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

        last_hidden_state = encoder_outputs["last_hidden_state"]
        last_hidden_state = nn.with_logical_constraint(last_hidden_state, ("batch", "length", "embed"))

        # average pool
        # TEMP: test sharding issues
        pooled_output = last_hidden_state[:, 0, :]  # this works but it's not a mean
        # pooled_output = last_hidden_state.mean(axis=1)  # this leads to huge memory usage
        # pooled_output = jnp.einsum("ijk->ik", last_hidden_state) / last_hidden_state.shape[1]  # still huge memory
        pooled_output = nn.with_logical_constraint(pooled_output, ("batch", "embed"))

        pooled_output = norm(self.use_rmsnorm)(
            dtype=dtype,
            use_bias=self.use_bias,
            use_scale=self.force_scale,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("embed",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
            name="final_norm",
        )(pooled_output)
        pooled_output = nn.with_logical_constraint(pooled_output, ("batch", "embed"))

        return dict(
            last_hidden_state=last_hidden_state,
            pooled_output=pooled_output,
            # TODO: add hidden states (for down-stream tasks)
        )


class CLIPVisionModelForImageClassification(nn.Module):
    vision_config: Any
    num_labels: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
    ):
        outputs = CLIPVisionTransformer(**self.vision_config, dtype=self.dtype)(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )

        logits = nn.Dense(
            self.num_labels,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed", "classifier")),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("classifier",)),
        )(outputs["pooled_output"])

        return dict(
            logits=logits,
            hidden_states=outputs["hidden_states"],
            attentions=outputs["attentions"],
        )


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


class CLIPModel(nn.Module):
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
        self.text_config = {**default_fields, **self.text_config}
        # add default fields vision_config
        default_fields = dataclasses.fields(CLIPVisionTransformer)
        default_fields = {f.name: f.default for f in default_fields if f.default is not dataclasses.MISSING}
        default_fields = {k: v for k, v in default_fields.items() if k not in ["parent", "name"]}
        self.vision_config = {**default_fields, **self.vision_config}
        return super().__post_init__()

    @nn.compact
    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        deterministic: bool = True,
    ):
        vision_config = unfreeze(self.vision_config)
        text_config = unfreeze(self.text_config)
        if self.dtype is not None:
            vision_config["dtype"] = self.dtype
            text_config["dtype"] = self.dtype
        dtype = getattr(jnp, self.dtype)

        vision_outputs = CLIPVisionTransformer(
            **vision_config,
            name="vision",
        )(
            pixel_values=pixel_values,
            deterministic=deterministic,
        )

        text_outputs = CLIPTextTransformer(
            **text_config,
            name="text",
        )(
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )

        image_embeds = vision_outputs["pooled_output"]
        # TEMP: may be better to go from embed to embed_proj
        image_embeds = nn.Dense(
            self.projection_dim,
            dtype=dtype,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed_proj", "embed")),
            name="vision_projection",
        )(image_embeds)

        text_embeds = text_outputs["pooled_output"]
        # TEMP: may be better to go from embed to embed_proj
        text_embeds = nn.Dense(
            self.projection_dim,
            dtype=dtype,
            use_bias=False,
            kernel_init=nn.with_logical_partitioning(default_kernel_init, ("embed_proj", "embed")),
            name="text_projection",
        )(text_embeds)

        # normalize features
        def normalize(x, eps=1e-7, safe_norm=True):
            if safe_norm:
                return x * jax.lax.rsqrt(jnp.sum(jax.lax.square(x), axis=-1, keepdims=True) + eps)
            else:
                return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        image_embeds = normalize(image_embeds)
        text_embeds = normalize(text_embeds)

        # cosine similarity as logits
        logit_scale = jnp.exp(
            self.param(
                "logit_scale",
                nn.with_logical_partitioning(nn.initializers.constant(self.logit_scale_init_value, dtype), (None,)),
                (1,),
            )
        )

        # logit bias is only used for sigmoid loss
        logit_bias = self.param(
            "logit_bias",
            nn.with_logical_partitioning(nn.initializers.constant(self.logit_bias_init_value, dtype), (None,)),
            (1,),
        )

        # logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        # logits_per_image = logits_per_text.T

        return dict(
            # logits_per_image=logits_per_image,
            # logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            # text_model_output=text_outputs,
            # vision_model_output=vision_outputs,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

    def init_inputs(config, rng: jax.random.PRNGKey):
        text_config = config.text_config
        vision_config = config.vision_config
        if isinstance(text_config, dict):
            text_config = SimpleNamespace(**text_config)
        if isinstance(vision_config, dict):
            vision_config = SimpleNamespace(**vision_config)
        input_ids = jnp.ones((1, text_config.max_position_embeddings), dtype="i4")
        attention_mask = jnp.ones((1, text_config.max_position_embeddings), dtype="i4")
        pixel_values = jnp.ones((1, vision_config.image_size, vision_config.image_size, 3), dtype="f4")
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        return {"rngs": rngs, "input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}
