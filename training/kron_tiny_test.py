import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding as NS
from jax.experimental.mesh_utils import create_device_mesh
from pprint import pprint

from kron import kron, scale_by_kron, get_opt_state_partition_specs


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


if __name__ == "__main__":
    devices = create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, ("fsdp",))
    print(f"Mesh: {mesh}")

    params = {
        "dense1": jnp.ones((16, 128, 256, 50)),
        "dense2": jnp.ones((4, 64, 50000, 64)),
        "bias": jnp.zeros(4),
    }

    scanned = {"dense1": True, "dense2": False, "bias": False}

    params_sharding = {
        "dense1": NS(mesh, P("fsdp")),  
        "dense2": NS(mesh, P(None, "fsdp")),  
        "bias": NS(mesh, P(None)),  
    }
    params_sharding_specs = jax.tree.map(lambda x: x.spec, params_sharding, is_leaf=lambda x: isinstance(x, NS))

    params = jax.device_put(params, device=params_sharding)
    print("Input params shapes:")
    pprint(jax.tree.map(lambda x: x.shape, params), width=120, sort_dicts=False)
    print("Input params sharding:")
    pprint(jax.tree.map(lambda x: x.sharding, params), width=120, sort_dicts=False)

    grads = jax.tree.map(jnp.ones_like, params)
    grads = jax.device_put(grads, device=params_sharding)
    print("Input grads sharding:")
    pprint(jax.tree.map(lambda x: x.sharding, grads), width=120, sort_dicts=False)

    precond_sharding_specs = P(None, None)
    print("Preconditioner sharding:")
    print(precond_sharding_specs)

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
        params_sharding=params_sharding_specs,
        preconditioner_sharding=precond_sharding_specs,
    )

    optimizer = kron(**optimizer_kwargs)

    with mesh:
        opt_state_sharding_specs = get_opt_state_partition_specs(params, **optimizer_kwargs)
        print("Opt state sharding from helper function:")
        pprint(opt_state_sharding_specs, width=120, sort_dicts=False)
        opt_state_sharding = jax.tree.map(lambda spec: NS(mesh, spec), opt_state_sharding_specs, is_leaf=lambda x: isinstance(x, P))

        # @partial(
        #     jax.jit,
        #     in_shardings=(params_sharding, params_sharding, opt_state_sharding),
        #     out_shardings=(params_sharding, opt_state_sharding),
        # )
        def test_step(params, grads, opt_state):
            return optimizer.update(grads, opt_state, params)

        opt_state = optimizer.init(params)
        print("Opt state after init:")
        pprint(jax.tree.map(lambda x: x.shape, opt_state), width=120, sort_dicts=False)
        print("Opt state sharding after init:")
        pprint(jax.tree.map(lambda x: x.sharding, opt_state), width=120, sort_dicts=False)

        updates, new_state = test_step(params, grads, opt_state)

        print("Output updates sharding:")
        pprint(jax.tree.map(lambda x: x.sharding, updates), width=120, sort_dicts=False)

        print("Output opt state sharding:")
        pprint(
            jax.tree.map(lambda x: x.sharding, new_state), width=120, sort_dicts=False
        )
