import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py
# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace


def _get_partition_rules():
    return [
        # embeddings
        (
            (
                "text_model",
                "embeddings",
            ),
            P(None, "mp"),
        ),
        (
            (
                "vision_model",
                "embeddings",
                "class_embedding",
            ),
            P(None),
        ),
        (
            (
                "vision_model",
                "embeddings",
                "patch_embedding",
            ),
            P(None, None, None, "mp"),
        ),
        (
            (
                "vision_model",
                "embeddings",
                "position_embedding",
            ),
            P(None, "mp"),
        ),
        (("patch_embedding",), P(None, None, None, "mp")),
        # attention
        (("(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (("out_proj", "kernel"), P("mp", None)),
        # FFN
        (("fc1", "kernel"), P("mp", None)),
        (("fc2", "kernel"), P(None, "mp")),
        (("(bias|scale|logit_scale)",), None),
        # Projection
        (("(text_projection|visual_projection)",), P("mp", None)),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(f"Unmatched -> {k}")
    l = list(result.keys())
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))
