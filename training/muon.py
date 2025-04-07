# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Muon.

Heavily modified and simplified from optax version contributed by leloy.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""

from typing import Any, NamedTuple
import numpy as np

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics


def newton_schulz_iters(x: jax.Array) -> jax.Array:
    r"""Orthogonalize via Newton-Schulz iteration.

    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be
    empirically effective to keep increasing the slope at zero even beyond the
    point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather
    something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    which turns out not to hurt model performance at all relative to UV^T, where
    USV^T = G is the SVD.
    """
    coeffs = (3.4445, -4.7750, 2.0315)

    def get_shape(g):
        shape1 = [np.prod(g.shape[:-1], dtype=np.int32), g.shape[-1]]
        shape2 = [g.shape[0], np.prod(g.shape[1:], dtype=np.int32)]
        return shape1 if np.abs(np.diff(shape1)) <= np.abs(np.diff(shape2)) else shape2

    orig_shape = x.shape
    x = jnp.reshape(x, get_shape(x))
    m, n = x.shape
    if m > n:
        x = x.T
    x /= jnp.linalg.norm(x) + 1e-7  # Ensure spectral norm is at most 1
    orig_dtype = x.dtype
    x = x.astype(jnp.bfloat16)
    # python loop instead of jax.lax.fori_loop to let jax better handle compilation
    for _ in range(5):
        a = x @ x.T
        b = coeffs[1] * a + coeffs[2] * a @ a
        x = coeffs[0] * x + b @ x
    x = x.astype(orig_dtype)
    if m > n:
        x = x.T
        x *= np.sqrt(m / n)  # follow muon scaling
    return jnp.reshape(x, orig_shape)


class MuonState(NamedTuple):
    """State for the Adam algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates


def scale_by_muon(
    beta: float = 0.95,
    scanned_layers: Any = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
) -> base.GradientTransformation:
    r"""Rescale updates according to the Muon algorithm.

    Muon is a variant of Shampoo that uses the Newton-schulz method to
    orthogonalize the momentum accumulated by the optimizer. Mathematically, it
    does steepest descent under the Schatten-p norm, for some large p. With
    p=infty, it is equivalent to Shampoo without accumulation, or steepest
    descent under the Spectral norm.

    References:
      Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
      <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

      Bernstein et al., `Old Optimizer, New Norm: An Anthology
      <https://arxiv.org/abs/2409.20325>`_, 2024
    """
    mu_dtype = jnp.bfloat16

    def init_fn(params):
        # unbox
        params = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )

        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        return MuonState(count=jnp.zeros([], jnp.int32), mu=mu)

    def update_fn(updates, state, params=None):
        del params
        count_inc = numerics.safe_increment(state.count)

        # unbox
        partitioned_flat, orig_struct = jax.tree.flatten(
            updates, is_leaf=lambda x: isinstance(x, nn.Partitioned)
        )
        updates = jax.tree.map(
            lambda u: u.unbox() if isinstance(u, nn.Partitioned) else u,
            updates,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )

        # nesterov momentum
        f = lambda g, t: g + beta * t
        mu = jax.tree.map(f, updates, state.mu)
        updates = jax.tree.map(f, updates, mu)

        # apply Newton-Schulz orthogonalization
        scanned_layers_ = scanned_layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(
                lambda u: (
                    optax.MaskedNode() if isinstance(u, optax.MaskedNode) else False
                ),
                updates,
            )
        else:
            scanned_layers_ = jax.tree.map(
                lambda s, u: (
                    optax.MaskedNode() if isinstance(u, optax.MaskedNode) else s
                ),
                scanned_layers_,
                updates,
            )
        updates = jax.tree.map(
            lambda u, s: _map_fn(
                lax_map_scanned_layers, lax_map_batch_size, int(s), newton_schulz_iters, u
            ),
            updates,
            scanned_layers_,
        )

        # box
        updates, orig_struct = jax.tree.flatten(updates)
        updates = [
            bu.replace_boxed(g) if isinstance(bu, nn.Partitioned) else g
            for bu, g in zip(partitioned_flat, updates)
        ]
        updates = orig_struct.unflatten(updates)

        mu = otu.tree_cast(mu, mu_dtype)
        return updates, MuonState(count=count_inc, mu=mu)

    return base.GradientTransformation(init_fn, update_fn)


def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Maybe map a fn along multiple leading axes."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        mapped_fn = lambda xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.lax.map(mapped_fn, xs=args, batch_size=bs if bs > 1 else None)
    else:
        mapped_fn = lambda *xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.vmap(mapped_fn)(*args)
