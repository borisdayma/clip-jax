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

import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.linear import (
    DenseGeneral,
    DotGeneralT,
    PrecisionLike,
)
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes
from flax.linen.partitioning import remat
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import AutoTokenizer
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
    FlaxSequenceClassifierOutput,
)
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from .utils import PretrainedFromWandbMixin

remat = nn_partitioning.remat

Axes = Union[int, Iterable[int]]

logger = logging.get_logger(__name__)


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
default_kernel_init = jax.nn.initializers.lecun_normal()


@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    logits_per_image: jnp.ndarray = None
    logits_per_text: jnp.ndarray = None
    text_embeds: jnp.ndarray = None
    image_embeds: jnp.ndarray = None
    text_model_output: FlaxBaseModelOutputWithPooling = None
    vision_model_output: FlaxBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


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


class RMSNorm(nn.Module):
    """RMSNorm, adapted from flax LayerNorm"""

    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable = jax.nn.initializers.zeros
    scale_init: Callable = jax.nn.initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @nn.compact
    def __call__(self, x):
        if dtype is None:
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
    if use_rmsnorm:
        return RMSNorm
    else:
        return nn.LayerNorm


class FlaxCLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, pixel_values):
        embed_dim = self.config.hidden_size
        patch_size = self.config.patch_size
        patch_embeds = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=self.config.use_bias,
            dtype=self.dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.lecun_normal(), ("conv_height", "conv_width", "input_channels", "embed")
            ),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), ("embed",)),
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
                nn.initializers.normal(1 / np.sqrt(embed_dim)), ("broadcast", "vocab", "embed")
            ),
            (1, num_patches, embed_dim),
        )
        embeddings = patch_embeds + position_embeds
        embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        return embeddings


class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, input_ids):
        embed_dim = self.config.hidden_size
        embeddings = nn.Embed(
            self.config.vocab_size,
            embed_dim,
            embedding_init=nn.with_logical_partitioning(
                jax.nn.initializers.normal(1 / np.sqrt(embed_dim)), ("vocab", "embed")
            ),
        )(input_ids.astype("i4"))
        embeddings = nn.with_logical_constraint(embeddings, ("batch", "length", "embed"))
        if self.config.position_embedding_type == "absolute":
            position_embeds = self.param(
                "position_embeds",
                nn.with_logical_partitioning(
                    jax.nn.initializers.normal(1 / np.sqrt(embed_dim)), (None, "vocab", "embed")
                ),
                (1, self.config.max_position_embeddings, embed_dim),
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
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros_init()
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

        query = nn.with_logical_constraint(query, ("batch", "length", "heads", "kv"))
        key = nn.with_logical_constraint(key, ("batch", "length", "heads", "kv"))
        value = nn.with_logical_constraint(value, ("batch", "length", "heads", "kv"))

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
                    mask, jnp.broadcast_to(jnp.arange(max_length) <= cur_index, tuple(batch_dims) + (1, 1, max_length))
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


class FlaxCLIPMLP(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states):
        h1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
            use_bias=self.config.use_bias,
        )(hidden_states)
        h1 = ACT2FN[self.config.hidden_act](h1)
        if self.config.use_glu:
            h2 = nn.Dense(
                self.config.intermediate_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(0.01),
                use_bias=self.config.use_bias,
            )(hidden_states)
            h1 = h1 * h2
        if self.config.ln_type == "normformer":
            h1 = norm(self.config.use_rmsnorm)(
                epsilon=self.config.layer_norm_eps,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                use_scale=self.config.force_scale,
            )(h1)
        h1 = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
            use_bias=self.config.use_bias,
        )(h1)
        return h1


class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
    ):
        # Self attention
        residual = hidden_states
        if self.config.ln_type in ["preln", "normformer"]:
            hidden_states = norm(self.config.use_rmsnorm)(
                epsilon=self.config.layer_norm_eps,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                use_scale=self.config.force_scale,
                name="pre_attention_layer_norm",
            )(hidden_states)
        hidden_states = MultiHeadDotProductAttention(
            num_heads=self.config.num_attention_heads,
            dtype=self.dtype,
            deterministic=deterministic,
            use_bias=self.config.use_bias,
            use_rotary=(self.config.position_embedding_type == "rotary"),
            max_length=self.config.max_position_embeddings,
            dropout_rate=self.config.attention_dropout,
            use_bias=self.config.use_bias,
            name="attention",
        )(inputs_q=hidden_states, inputs_kv=hidden_states, mask=attention_mask, deterministic=deterministic)
        if self.config.ln_type == "normformer":
            hidden_states = norm(self.config.use_rmsnorm)(
                epsilon=self.config.layer_norm_eps,
                dtype=self.dtype,
                use_bias=self.config.use_bias,
                use_scale=True,
                name="post_attention_layer_norm",
            )(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = norm(self.config.use_rmsnorm)(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            use_scale=self.config.force_scale,
        )(hidden_states)
        hidden_states = FlaxCLIPMLP(self.config, dtype=self.dtype)(hidden_states)
        hidden_states = residual + hidden_states

        if self.config.scan_layers:
            return hidden_states, None
        return hidden_states


class FlaxCLIPEncoder(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # gradient checkpointing
        use_scan = True
        layer = (
            remat(
                FlaxCLIPEncoderLayer,
                static_argnums=(2,),
                prevent_cse=not use_scan,
            )
            if self.config.gradient_checkpointing
            else FlaxCLIPEncoderLayer
        )

        # TODO: stack outputs of hidden states when output_hidden_states=True
        assert not output_hidden_states, "scan does not support output_hidden_states"
        hidden_states, _ = nn.scan(
            layer,
            variable_axes={"params": 0, "cache": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(nn.broadcast, nn.broadcast),
            length=self.config.num_hidden_layers,
            unroll=self.config.unroll_scan,
        )(self.config, dtype=self.dtype, name="scanned")(hidden_states, attention_mask, deterministic)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class FlaxCLIPTextTransformer(nn.Module):
    config: CLIPTextConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = FlaxCLIPTextEmbeddings(self.config, dtype=self.dtype)(
            input_ids=input_ids, position_ids=position_ids
        )

        encoder_outputs = FlaxCLIPEncoder(self.config, dtype=self.dtype)(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = norm(self.config.use_rmsnorm)(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            use_scale=self.config.force_scale,
        )(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the BOS embedding instead of EOS (no causal mask anymore)
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FlaxCLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)(pixel_values)
        hidden_states = norm(self.config.use_rmsnorm)(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            use_scale=self.config.force_scale,
        )(hidden_states)

        encoder_outputs = FlaxCLIPEncoder(self.config, dtype=self.dtype)(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        # average pool
        pooled_output = last_hidden_state.mean(axis=1)
        pooled_output = norm(self.config.use_rmsnorm)(
            epsilon=self.config.layer_norm_eps,
            dtype=self.dtype,
            use_bias=self.config.use_bias,
            use_scale=self.config.force_scale,
        )(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FlaxCLIPTextPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPTextConfig
    module_class: nn.Module = None

    def __init__(
        self,
        config: CLIPTextConfig,
        input_shape=(1, 1),
        seed: int = 0,
        dtype: Dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        input_ids = jnp.zeros(input_shape, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    module_class: nn.Module = None

    def __init__(
        self,
        config: CLIPVisionConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: Dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        pixel_values = jax.random.normal(rng, input_shape)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )


class FlaxCLIPPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPConfig
    module_class: nn.Module = None

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: Dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (
                (1, 1),
                (
                    1,
                    config.vision_config.image_size,
                    config.vision_config.image_size,
                    3,
                ),
            )
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        input_ids = jnp.zeros(input_shape[0], dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape[0])
        attention_mask = jnp.ones_like(input_ids)

        pixel_values = jax.random.normal(rng, input_shape[1])

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, pixel_values, attention_mask, position_ids)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def num_params(self, params=None):
        if params is None:
            params = self.params
        num_params = jax.tree_util.tree_map(lambda param: param.size, flatten_dict(unfreeze(params))).values()
        return sum(list(num_params))

    def unscan(self, params):
        if self.config.use_scan:
            self.config.use_scan = False
            params = flatten_dict(params)
            scanned_keys = [k for k in params.keys() if "scanned" in k]
            for k in scanned_keys:
                v = params[k]
                name_idx = k.index("scanned")
                for i in range(len(v)):
                    new_k = (
                        *k[:name_idx],
                        f"{i}",
                        *k[name_idx + 1 :],
                    )
                    params[new_k] = v[i]
                del params[k]
            params = unflatten_dict(params)
        return params

    def __call__(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(pixel_values, dtype=jnp.float32),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )

    def get_text_features(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, input_ids, attention_mask, position_ids, deterministic):
            text_outputs = module.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
            )
            pooled_output = text_outputs[1]
            text_features = module.text_projection(pooled_output)
            return text_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            method=_get_features,
            rngs=rngs,
        )

    def get_image_features(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train=False,
    ):
        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        def _get_features(module, pixel_values, deterministic):
            vision_outputs = module.vision_model(pixel_values=pixel_values, deterministic=deterministic)
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = module.visual_projection(pooled_output)
            return image_features

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            method=_get_features,
            rngs=rngs,
        )


class FlaxCLIPTextModule(nn.Module):
    config: CLIPTextConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return FlaxCLIPTextTransformer(self.config, dtype=self.dtype)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPTextModel(FlaxCLIPTextPreTrainedModel):
    module_class = FlaxCLIPTextModule


class FlaxCLIPVisionModule(nn.Module):
    config: CLIPVisionConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return FlaxCLIPVisionTransformer(self.config, dtype=self.dtype)(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxCLIPVisionModel(PretrainedFromWandbMixin, FlaxCLIPVisionPreTrainedModel):
    module_class = FlaxCLIPVisionModule


class FlaxCLIPVisionModelForImageClassificationModule(nn.Module):
    config: CLIPVisionConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = FlaxCLIPVisionTransformer(self.config, dtype=self.dtype)(
            pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = nn.Dense(
            self.config.num_labels,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )(outputs.pooler_output)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxCLIPVisionModelForImageClassification(PretrainedFromWandbMixin, FlaxCLIPVisionPreTrainedModel):
    module_class = FlaxCLIPVisionModelForImageClassificationModule


class FlaxCLIPTextModelForFineTuningModule(nn.Module):
    config: CLIPVisionConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
    ):
        text_outputs = FlaxCLIPTextTransformer(self.config, dtype=self.dtype)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_hidden_states=True,
            return_dict=True,
        )

        # return penuultimate layer
        return text_outputs.hidden_states[-2]


class FlaxCLIPTextModelForFineTuning(PretrainedFromWandbMixin, FlaxCLIPTextPreTrainedModel):
    module_class = FlaxCLIPTextModelForFineTuningModule


class FlaxCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        text_config = self.config.text_config
        vision_config = self.config.vision_config
        projection_dim = self.config.projection_dim
        text_embed_dim = text_config.hidden_size
        vision_embed_dim = vision_config.hidden_size

        vision_outputs = FlaxCLIPVisionTransformer(vision_config, dtype=self.dtype)(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            name="vision",
        )

        text_outputs = FlaxCLIPTextTransformer(text_config, dtype=self.dtype)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            name="text",
        )

        image_embeds = vision_outputs[1]
        image_embeds = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
            name="vision_projection",
        )(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
            name="text_projection",
        )(text_embeds)

        # normalize features
        def normalize(x):
            return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

        image_embeds = normalize(image_embeds)
        text_embeds = normalize(text_embeds)

        # cosine similarity as logits
        logit_scale = jnp.exp(
            self.param("logit_scale", jax.nn.initializers.constant(self.config.logit_scale_init_value, self.dtype), [])
        )
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T

        if not return_dict:
            return (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                text_outputs,
                vision_outputs,
            )

        return FlaxCLIPOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FlaxCLIPModel(PretrainedFromWandbMixin, FlaxCLIPPreTrainedModel):
    module_class = FlaxCLIPModule


class AutoTokenizer(PretrainedFromWandbMixin, AutoTokenizer):
    pass
