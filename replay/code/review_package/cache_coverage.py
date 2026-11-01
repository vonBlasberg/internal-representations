"""Guard the retained native-byte cache recorder against silent field omissions.

This is an explicit wrapper for a future admitted runtime, not a patch to old
receipts. Unknown fields (including scalar metadata) require review, not guesses
about whether they matter. No tensors are decoded or serialized here.
"""
from .admission import require_cache_coverage
from .reuse import legacy


def guarded_cache_summary(recorder, cache, *, model=None, position_state=None, **limits):
    summary = recorder(cache, model=model, position_state=position_state, **limits)
    maximum_nodes = limits.get("maximum_nodes", 8192)
    maximum_depth = limits.get("maximum_depth", 24)
    nodes, paths = 0, []

    def check(value, recorded, path, depth):
        nonlocal nodes
        import torch
        nodes += 1
        if nodes > maximum_nodes or depth > maximum_depth:
            raise legacy.AnalysisError("Cache coverage exceeds the declared inspection bound")
        if torch.is_tensor(value) or value is None or isinstance(value, (bool, int, float, str, torch.dtype, torch.device, type)):
            return
        if isinstance(value, dict):
            require_cache_coverage([str(k) for k in value], recorded.keys())
            for key, item in value.items():
                check(item, recorded[str(key)], path + '.' + str(key), depth + 1)
        elif isinstance(value, (tuple, list)):
            if len(value) != len(recorded):
                raise legacy.AnalysisError("Cache sequence coverage differs")
            for i, item in enumerate(value):
                check(item, recorded[i], path + f'[{i}]', depth + 1)
        elif hasattr(value, "__dict__"):
            fields = recorded.get("fields", {})
            try:
                require_cache_coverage(vars(value), fields)
            except legacy.AnalysisError as exc:
                raise legacy.AnalysisError(path + ': ' + str(exc)) from exc
            paths.append({"path": path, "type": type(value).__name__, "fields": sorted(vars(value))})
            for key, item in vars(value).items():
                check(item, fields[key], path + '.' + key, depth + 1)
        else:
            raise legacy.AnalysisError("Unsupported cache object: " + path)

    check(cache, summary["cache"], "cache", 0)
    check(position_state, summary["position_state"], "position_state", 0)
    rope = getattr(getattr(model, "model", None), "rope_deltas", None) if model is not None else None
    check(rope, summary["model_rope_deltas"], "model_rope_deltas", 0)
    return {**summary, "field_coverage": {"complete": True, "object_fields": paths,
            "policy": "All instance fields must be recorded; no implicit exemptions"}}
