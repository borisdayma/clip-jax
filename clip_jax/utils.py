import json

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
