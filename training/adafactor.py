from typing import Optional, Union, Callable, Any

import jax.numpy as jnp
import optax
from optax._src import base, combine, transform


def scale_by_adafactor(
    min_dim_size_to_factor=32,
    decay_rate=0.8,
    decay_offset=0,
    beta2_cap=0.999,
    clipping_threshold=None,
    momentum=0.9,
    dtype_momentum=jnp.bfloat16,
    eps=1e-30,
):
    """The BigVision variant of Adafactor optimizer."""

    def _decay_rate_pow(i, exponent):
        """Second-order moment decay schedule."""
        t = jnp.array(i, jnp.float32) + 1.0
        return jnp.minimum(beta2_cap, 1.0 - t ** (-exponent))

    scale_by_rms = optax.scale_by_factored_rms(
        factored=True,
        decay_rate=decay_rate,
        step_offset=decay_offset,
        min_dim_size_to_factor=min_dim_size_to_factor,
        epsilon=eps,
        decay_rate_fn=_decay_rate_pow,
    )

    clip = (
        optax.clip_by_block_rms(clipping_threshold)
        if clipping_threshold
        else optax.identity()
    )

    mom = (
        optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum)
        if momentum
        else optax.identity()
    )

    return optax.chain(scale_by_rms, clip, mom)


def adafactorw(
    learning_rate: base.ScalarOrSchedule,
    min_dim_size_to_factor: int = 32,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    beta2_cap: float = 0.999,
    clipping_threshold: Optional[float] = None,
    momentum: float = 0.9,
    dtype_momentum: Any = jnp.bfloat16,
    eps: float = 1e-30,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """Adafactor with weight decay regularization.

    AdafactorW combines Adafactor with decoupled weight decay regularization,
    similar to how AdamW extends Adam. This implementation uses the BigVision
    variant of Adafactor.

    Args:
        learning_rate: A global scaling factor.
        min_dim_size_to_factor: Minimum dimension size to apply factorization.
        decay_rate: Base decay rate for second moment estimators.
        decay_offset: Offset for decay rate schedule.
        beta2_cap: Maximum decay rate for second moment estimator.
        clipping_threshold: Optional gradient clipping threshold.
        momentum: Momentum parameter for first moment estimation.
        dtype_momentum: Data type for momentum accumulator.
        eps: Small constant for numerical stability.
        weight_decay: Strength of the weight decay regularization.
        mask: A tree with same structure as params PyTree or a Callable that returns
            such a pytree. Controls which parameters receive weight decay.

    Returns:
        The corresponding `GradientTransformation`.
    """
    return combine.chain(
        scale_by_adafactor(
            min_dim_size_to_factor=min_dim_size_to_factor,
            decay_rate=decay_rate,
            decay_offset=decay_offset,
            beta2_cap=beta2_cap,
            clipping_threshold=clipping_threshold,
            momentum=momentum,
            dtype_momentum=dtype_momentum,
            eps=eps,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        transform.scale_by_learning_rate(learning_rate),
    )