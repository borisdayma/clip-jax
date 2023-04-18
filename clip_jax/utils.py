import json

import fsspec

from .wandb_utils import maybe_use_artifact


def load_config(config_path):
    """Load config from file."""
    with maybe_use_artifact(config_path, "config.json") as json_path:
        with fsspec.open(json_path, "r") as f:
            config = json.load(f)
    return config
