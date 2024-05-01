# Below section adapted from t5x
def logical_axis_rules(
    activation_partitioning_dims=1,
    parameter_partitioning_dims=1,
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
            ("cache_batch", "data"),
            ("cache_heads", "model"),
            ("cache_kv", None),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 1:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("embed", "model"),
            ("cache_batch", "data"),
            ("cache_heads", "model"),
            ("cache_kv", None),
        ]
    elif activation_partitioning_dims == 1 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("embed", "data"),
            ("cache_batch", "data"),
            ("cache_heads", "model"),
            ("cache_kv", None),
        ]
    elif activation_partitioning_dims == 2 and parameter_partitioning_dims == 2:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("embed", "model"),
            ("embed", "data"),
            ("cache_batch", "data"),
            ("cache_heads", "model"),
            ("cache_kv", None),
        ]
    # TODO: find better namings, this shards mlp as FSDP and rest is same as 1x1
    elif activation_partitioning_dims == 3 and parameter_partitioning_dims == 3:
        rules = [
            ("batch", "data"),
            ("vocab", "model"),
            ("embed", None),
            ("mlp", "data"),
            ("mlp", "model"),
            ("embed_proj", "model"),
            ("heads", "model"),
            ("kv", None),
            ("cache_batch", "data"),
            ("cache_heads", "model"),
            ("cache_kv", None),
        ]
    else:
        raise ValueError(
            f"`activation_partitioning_dims` = {activation_partitioning_dims} "
            f"`parameter_partitioning_dims` = {parameter_partitioning_dims} "
            "is not supported."
        )
    return rules
