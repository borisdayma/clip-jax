import collections
import re
import threading

import jax
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict

try:
    from jax.sharding import PartitionSpec as P
except:
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
            P("vocab", "embed"),
        ),
        (
            (
                "vision_model",
                "embeddings",
                "patch_embedding",
            ),
            P(None, None, None, "embed"),
        ),
        (
            (
                "vision_model",
                "embeddings",
                "position_embedding",
            ),
            P(None, "embed"),
        ),
        (
            ("patch_embedding",),
            P(None, None, None, "embed"),
        ),
        # attention
        (
            ("(q_proj|k_proj|v_proj)", "kernel"),
            P("embed", "joined_kv"),
        ),
        (
            ("out_proj", "kernel"),
            P("joined_kv", "embed"),
        ),
        # FFN
        (
            ("fc1", "kernel"),
            P("embed", "mlp"),
        ),
        (
            ("fc1_glu", "kernel"),
            P("embed", "mlp"),
        ),
        (
            ("fc2", "kernel"),
            P("mlp", "embed"),
        ),
        (
            ("(bias|scale)",),
            P("embed"),
        ),
        # Projection
        (
            ("(text_projection|visual_projection)",),
            P("embed", "embed_proj"),
        ),
        (
            ("logit_scale",),
            None,
        ),
    ]


# Below section adapted from t5x
def standard_logical_axis_rules(
    activation_partitioning_dims=1,
    parameter_partitioning_dims=1,
    additional_rules=None,
):
    """Default sharding rules in terms of logical axis names.

    Args:
      activation_partitioning_dims: enables 2-D activation sharding when set to 2.
      parameter_partitioning_dims: enables 2-D parameter sharding when set to 2.
      additional_rules: additional rules (a sequence of tuples) that will be
        appended to the standard rules.

    Returns:
      Sequence of logical axis rules
    """

    if activation_partitioning_dims == 1 and parameter_partitioning_dims == 1:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("embed", None),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 1:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("embed", "model"),
        ]
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("image_kv", None),
            ("embed", "data"),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("joined_kv", "model"),
            ("embed", "model"),
            ("embed", "data"),
        ]
    else:
        raise ValueError(
            f"`activation_partitioning_dims` = {activation_partitioning_dims} "
            f"`parameter_partitioning_dims` = {parameter_partitioning_dims} "
            "is not supported."
        )

    if additional_rules:
        rules.extend(additional_rules)

    return rules


class _AxisRules:
    """Dynamic logical axis to mesh axis binding context."""

    def __init__(self):
        self._thread_data = threading.local()

    @property
    def rules(self):
        if not hasattr(self._thread_data, "rules"):
            self._thread_data.rules = ()
        return self._thread_data.rules

    @rules.setter
    def rules(self, value):
        self._thread_data.rules = value


# Global axis binding context.
_axis_rules = _AxisRules()


class _UnassignedAxis:
    """Sentinel class for unassigned logical axis name."""

    def __repr__(self):
        return "UnassignedAxis"

    def __bool__(self):
        return False


_unassigned_axis = _UnassignedAxis()


def _mesh_assignment_free(new_assignment, existing_assignments):
    """Determines if a given mesh axis has already been assigned."""
    new = set(jax.tree_util.tree_leaves(new_assignment))
    existing = set(jax.tree_util.tree_leaves(existing_assignments))
    if existing.intersection(new):
        return False
    return True


def _logical_to_mesh_axes(
    array_dim_names,
    rules=None,
):
    if array_dim_names is None:
        return None
    if rules is None:
        rules = _axis_rules.rules
    axis_name_counts = collections.Counter(array_dim_names)
    dups = tuple(k for k, v in axis_name_counts.items() if v > 1 and k is not None)
    if dups:
        raise ValueError(f"Unsupported: Dimensions {dups} occur more than once in array names.")
    if not isinstance(rules, (tuple, list)):
        raise ValueError("Unknown axis rule specification type.")
    # We assign mesh axes using a priority based ruleset over logical axis names.
    result = [_unassigned_axis] * len(array_dim_names)
    for rule_model_name, rule_mesh_names in rules:
        if rule_model_name in array_dim_names:
            pos = array_dim_names.index(rule_model_name)
            if _mesh_assignment_free(rule_mesh_names, result) and result[pos] == _unassigned_axis:
                result[pos] = rule_mesh_names
    return result


def logical_to_mesh_axes(
    array_dim_names,
    rules=None,
):
    result = _logical_to_mesh_axes(array_dim_names, rules)
    if result is None:
        return None
    # We default to None - ie unsharded along the dimension.
    result = [None if x is _unassigned_axis else x for x in result]
    return P(*result)


def set_partitions(in_dict, use_scan, activation_partitioning_dims=1, parameter_partitioning_dims=1):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    for k, v in result.items():
        if v == _unmatched:
            print(f"Unmatched -> {k}")
    result = jax.tree_map(
        lambda x: logical_to_mesh_axes(
            x,
            standard_logical_axis_rules(
                activation_partitioning_dims=activation_partitioning_dims,
                parameter_partitioning_dims=parameter_partitioning_dims,
            ),
        ),
        result,
    )
    if use_scan:
        # add None dimension to layers
        result = {k: (P(*(None,) + v) if v is not None else None) if "scanned" in k else v for k, v in result.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."

    result = freeze(unflatten_dict(result))
    return result
