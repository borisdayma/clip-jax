import os
from functools import partial

import jax
import jax.numpy as jnp
from pprint import pprint

from kron import kron, scale_by_kron, get_opt_state_partition_specs


if __name__ == "__main__":
    params = {
        "dense1": jnp.ones((16, 128, 256, 50)),
        "dense2": jnp.ones((4, 64, 50000, 64)),
        "bias": jnp.zeros(4),
    }

    scanned = {"dense1": True, "dense2": False, "bias": False}

    print("Input params shapes:")
    pprint(jax.tree.map(lambda x: x.shape, params), width=120, sort_dicts=False)

    grads = jax.tree.map(jnp.ones_like, params)

    optimizer_kwargs = dict(
        learning_rate=0.001,
        b1=0.9,
        weight_decay=0.01,
        weight_decay_mask=None,
        normalize_grads=True,
        preconditioner_update_probability=1.0,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        mu_dtype=None,
        precond_dtype=None,
        precond_update_precision="tensorfloat32",
        precond_grads_precision=None,
        scanned_layers=scanned,
        lax_map_scanned_layers=True,
        lax_map_batch_size=2,
        merge_small_dims=True,
        target_merged_dim_size=2048,
        partition_grads_into_blocks=True,
        block_size=32,
        buffer_qq=False,
    )

    optimizer = kron(**optimizer_kwargs)

    def test_step(params, grads, opt_state):
        return optimizer.update(grads, opt_state, params)

    opt_state = optimizer.init(params)
    print("Opt state after init:")
    pprint(jax.tree.map(lambda x: x.shape, opt_state), width=120, sort_dicts=False)

    updates, new_state = test_step(params, grads, opt_state)
