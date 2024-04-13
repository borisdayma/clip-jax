import json
from dataclasses import fields

import flax
import fsspec
import jax

from .wandb_utils import maybe_use_artifact


def load_config(config_path):
    """Load config from file."""
    with maybe_use_artifact(config_path, "config.json") as json_path:
        with fsspec.open(json_path, "r") as f:
            config = json.load(f)
    return config


def count_params(pytree):
    return sum([x.size for x in jax.tree_util.tree_leaves(pytree)])


def asdict(model):
    excluded_keys = ["maxtext_mesh", "maxtext_args", "parent", "name"]
    return {f.name: getattr(model, f.name) for f in fields(model) if f.name not in excluded_keys}


# from maxtext
def unbox_logicallypartioned(boxed_pytree):
    """Unboxes the flax.LogicallyPartitioned pieces

    Args:
      boxed_pytree: a pytree that includes LogicallyPartitioned
        leaves.
    Returns:
      a pytree where all all LogicallyPartitioned leaves have been unboxed.
    """
    return jax.tree_util.tree_map(
        lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
        boxed_pytree,
        is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
    )
