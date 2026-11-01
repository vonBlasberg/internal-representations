"""Native-byte, bounded evidence receipts; no raw state or cache serialization."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: Any, *, chunk_values: int = 1_048_576) -> str:
    """Hash native scalar bytes, preserving BF16, signed zero, and storage dtype.

    No float conversion is performed. CPU contiguous staging is only a transport
    representation. At most one chunk of values is exposed to NumPy at a time.
    """
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("Native-byte hashing requires a tensor.")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("Only unquantized strided tensors have a defined receipt encoding.")
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    flat = cpu.reshape(-1)
    for start in range(0, flat.numel(), chunk_values):
        chunk = flat[start : start + chunk_values]
        # Bitcast, not cast: BF16 remains its exact two-byte representation.
        native_bytes = chunk.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(native_bytes)
    return digest.hexdigest()


def tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    summary = {
        "kind": "tensor_native_byte_summary",
        "shape": list(tensor.shape),
        "storage_dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "device": str(tensor.device),
        "native_sha256": tensor_sha256(tensor),
    }
    if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        summary["finite"] = bool(torch.isfinite(tensor).all().item())
    return summary


def weight_fingerprint(model: Any) -> dict[str, Any]:
    """Stream every parameter and buffer without retaining a second model."""
    entries = []
    for name, value in sorted(model.state_dict().items(), key=lambda item: item[0]):
        entries.append({
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "native_sha256": tensor_sha256(value),
        })
    return {
        "tensor_count": len(entries),
        "total_values": sum(item["numel"] for item in entries),
        "entries_sha256": canonical_sha256(entries),
    }


class HookRegistry:
    """Identity-sensitive snapshot of every module's forward-hook registry."""

    def __init__(self, model: Any) -> None:
        self._handles: dict[str, dict[int, Any]] = {
            name: dict(module._forward_hooks) for name, module in model.named_modules()
        }

    @property
    def active_count(self) -> int:
        return sum(len(handles) for handles in self._handles.values())

    @staticmethod
    def _active_paths(handles: dict[str, dict[int, Any]]) -> list[str]:
        return sorted(name for name, values in handles.items() if values)

    def assert_restored(self, model: Any) -> dict[str, Any]:
        current = {name: dict(module._forward_hooks) for name, module in model.named_modules()}
        if current.keys() != self._handles.keys():
            raise RuntimeError("The module registry changed during an instrumented trial.")
        for name in self._handles:
            before, after = self._handles[name], current[name]
            if before.keys() != after.keys() or any(before[key] is not after[key] for key in before):
                raise RuntimeError(f"Forward-hook registry was not restored at {name or '<root>'}.")
        return {
            "restored": True,
            "active_hook_count": self.active_count,
            "active_hook_paths": self._active_paths(current),
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "active_hook_count": self.active_count,
            "active_hook_paths": self._active_paths(self._handles),
        }


class _SummaryBudget:
    def __init__(self, maximum_nodes: int) -> None:
        self.remaining = maximum_nodes

    def take(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("Cache summary exceeds its declared node bound.")


def _state_kind(path: str) -> str:
    components = path.replace("[", ".").replace("]", "").split(".")
    kinds = {
        "conv_states": "linear_convolution_state",
        "recurrent_states": "linear_recurrent_state",
        "keys": "full_attention_key_cache",
        "values": "full_attention_value_cache",
        "position_ids": "position_state",
        "indexer_position_ids": "position_state",
        "rope_deltas": "position_state",
    }
    for component in reversed(components):
        if component in kinds:
            return kinds[component]
    return "cache_or_position_metadata"


def summarize_cache(
    cache: Any,
    *,
    model: Any = None,
    position_state: Any = None,
    maximum_nodes: int = 8192,
    maximum_metadata_items: int = 64,
    maximum_depth: int = 24,
) -> dict[str, Any]:
    """Recursively summarize conv, recurrent, K/V, position, and flag state.

    Unknown cache objects fail closed rather than being mistaken for empty caches.
    Tensor values are hashed, never serialized. Metadata containers and depth are
    bounded. The actual cache implementation and coverage are explicit.
    """
    import torch

    budget = _SummaryBudget(maximum_nodes)
    tensor_kinds: set[str] = set()
    tensor_count = 0

    def walk(value: Any, path: str, depth: int) -> Any:
        nonlocal tensor_count
        budget.take()
        if depth > maximum_depth:
            raise ValueError("Cache summary exceeds its declared depth bound.")
        kind = _state_kind(path)
        if torch.is_tensor(value):
            tensor_count += 1
            tensor_kinds.add(kind)
            return {"state_kind": kind, **tensor_summary(value)}
        if value is None or isinstance(value, (bool, int, str)):
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("Unbounded string metadata in cache receipt.")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Nonfinite cache metadata is not a valid receipt.")
            return value
        if isinstance(value, (torch.dtype, torch.device)) or isinstance(value, type):
            return str(value)
        if isinstance(value, dict):
            if len(value) > maximum_metadata_items:
                raise ValueError("Cache metadata mapping exceeds its declared bound.")
            result = {}
            for key in sorted(value, key=lambda item: str(item)):
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise TypeError("Unsupported cache metadata key.")
                result[str(key)] = walk(value[key], f"{path}.{key}", depth + 1)
            return result
        if isinstance(value, (tuple, list)):
            if len(value) > maximum_metadata_items:
                raise ValueError("Cache metadata sequence exceeds its declared bound.")
            return [walk(item, f"{path}[{index}]", depth + 1) for index, item in enumerate(value)]
        if hasattr(value, "__dict__"):
            allowed = {
                "layers", "keys", "values", "conv_states", "recurrent_states",
                "number_of_states", "is_conv_states_initialized",
                "is_recurrent_states_initialized", "has_previous_state",
                "conv_kernel_size", "record_past", "cumulative_length",
                "is_initialized", "is_compileable", "device", "dtype",
                "offloading", "only_non_sliding", "sliding_window",
                "_sliding_window_tensor", "indexer_keys", "indexer_position_ids",
                "is_indexer_initialized", "layer_class_to_replicate",
            }
            fields = {key: item for key, item in vars(value).items() if key in allowed}
            return {
                "object_type": type(value).__name__,
                "fields": walk(fields, path, depth + 1),
            }
        raise TypeError(f"Unsupported cache metadata at {path}: {type(value).__name__}")

    summary = walk(cache, "cache", 0) if cache is not None else None
    if position_state is not None:
        position_summary = walk(position_state, "position_state", 0)
    else:
        position_summary = None
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None) if model is not None else None
    return {
        "receipt_kind": "BOUNDED_RECURSIVE_CACHE_STATE_RECEIPT",
        "cache_implementation": type(cache).__name__ if cache is not None else None,
        "cache": summary,
        "position_state": position_summary,
        "model_rope_deltas": walk(rope_deltas, "rope_deltas", 0) if rope_deltas is not None else None,
        "tensor_count": tensor_count,
        "tensor_state_kinds_observed": sorted(tensor_kinds),
        "raw_tensors_retained": False,
        "maximum_nodes": maximum_nodes,
        "maximum_metadata_items": maximum_metadata_items,
        "maximum_depth": maximum_depth,
    }
