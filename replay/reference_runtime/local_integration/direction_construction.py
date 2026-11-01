"""Response-position construction using Steerability capture and vector objects.

This module never loads a model, reads a case bank, generates a response, selects
a layer or launches a worker. The caller supplies an already admitted model and
explicit tokenized construction pairs. Research use still requires the separate
construction-label check and phase approval in the frozen study plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from steerability.algorithms.core.internals.capture import capture_hidden
from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta
from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.internals.pooling import aggregate_condition_hidden
from steerability.algorithms.state_control.common.steering_vector import SteeringVector


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ResponseTokens:
    prefix_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    selected_response_positions: tuple[int, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prefix_ids + self.response_ids


@dataclass(frozen=True)
class ConstructionPair:
    pair_id: str
    family_id: str
    positive: ResponseTokens
    negative: ResponseTokens
    phase: str = "construction_fit"


@dataclass(frozen=True)
class ConstructionSpec:
    model_id: str
    model_revision: str
    model_type: str
    sites: tuple[tuple[int, str], ...]
    hidden_width: int
    pair_ids: tuple[str, ...]
    construction_sha256: str
    excluded_families: tuple[str, ...]
    pooling: str
    accumulation_dtype: str
    insertion_dtype: str
    max_sequence_length: int
    synthetic_test: bool
    # Explicit values are retained in the header; unsupported alternatives fail.
    location: str = "layer_output"
    direction_dtype: str = "float32"
    intended_operation: str = "additive"


def input_hash(pairs: tuple[ConstructionPair, ...]) -> str:
    return canonical_hash([asdict(pair) for pair in pairs])


def _integer(value: Any) -> bool:
    return type(value) is int


def validate_inputs(spec: ConstructionSpec, pairs: tuple[ConstructionPair, ...]) -> None:
    """Reject undeclared positions, pairing errors and held-out input before capture."""
    if not pairs or tuple(p.pair_id for p in pairs) != spec.pair_ids:
        raise ValueError("Pairs must exactly match the declared nonempty ordered pair IDs.")
    if len(set(spec.pair_ids)) != len(spec.pair_ids) or any(not p for p in spec.pair_ids):
        raise ValueError("Pair IDs must be nonempty and unique.")
    if input_hash(pairs) != spec.construction_sha256:
        raise ValueError("Construction input hash differs from the declared hash.")
    if not all(isinstance(v, str) and v for v in (spec.model_id, spec.model_revision, spec.model_type)):
        raise ValueError("Model identity and revision must be explicit.")
    if len(spec.model_revision) != 40 or any(c not in "0123456789abcdef" for c in spec.model_revision):
        raise ValueError("Model revision must be an exact 40-character commit identity.")
    if not _integer(spec.hidden_width) or spec.hidden_width < 1:
        raise ValueError("Hidden width must be positive.")
    if not _integer(spec.max_sequence_length) or spec.max_sequence_length < 2:
        raise ValueError("The sequence limit must be explicit.")
    if spec.pooling not in ("response_mean", "response_last"):
        raise ValueError("Choose declared response_mean or response_last pooling.")
    if spec.accumulation_dtype not in ("float32", "float64"):
        raise ValueError("Accumulation dtype must be float32 or float64.")
    if spec.insertion_dtype not in ("float32", "bfloat16") or spec.direction_dtype != "float32":
        raise ValueError("Unsupported stored or insertion dtype.")
    if spec.location != "layer_output" or spec.intended_operation != "additive":
        raise ValueError("Only declared raw layer-output additive directions are supported.")
    if type(spec.synthetic_test) is not bool:
        raise ValueError("Synthetic-test status must be explicit.")
    if not spec.sites or len({s[0] for s in spec.sites}) != len(spec.sites):
        raise ValueError("Candidate sites must be nonempty and unique.")
    if any(not _integer(i) or i < 0 or not isinstance(path, str) or not path for i, path in spec.sites):
        raise ValueError("Each candidate needs an exact layer index and module path.")
    for pair in pairs:
        if pair.phase != "construction_fit" or not pair.family_id or pair.family_id in spec.excluded_families:
            raise ValueError("Only construction-fit families may enter the direction fit.")
        if pair.positive.prefix_ids != pair.negative.prefix_ids:
            raise ValueError("Both responses in a pair must have identical teacher-forced prefix IDs.")
        for response in (pair.positive, pair.negative):
            if not response.prefix_ids or not response.response_ids:
                raise ValueError("Both the teacher-forced prefix and response must be nonempty.")
            if len(response.input_ids) > spec.max_sequence_length:
                raise ValueError("Input exceeds the declared length; truncation is not allowed.")
            if any(not _integer(t) or t < 0 for t in response.input_ids):
                raise ValueError("Token IDs must be nonnegative integers.")
            positions = response.selected_response_positions
            if not positions or any(not _integer(i) for i in positions):
                raise ValueError("Response positions must be explicit nonempty integer indices.")
            if tuple(sorted(set(positions))) != positions or min(positions) < 0 or max(positions) >= len(response.response_ids):
                raise ValueError("Response positions must be unique, ordered and within the response.")


def pool_response(hidden: torch.Tensor, response: ResponseTokens, spec: ConstructionSpec) -> torch.Tensor:
    """Pool selected response positions, never prompt, padding or implicit EOS positions."""
    if tuple(hidden.shape) != (1, len(response.input_ids), spec.hidden_width):
        raise ValueError("Captured state shape differs from the declared sequence and width.")
    if not torch.is_floating_point(hidden) or not torch.isfinite(hidden).all():
        raise ValueError("Captured states must be finite floating-point values.")
    selection = torch.zeros((1, len(response.input_ids)), dtype=torch.bool, device=hidden.device)
    for position in response.selected_response_positions:
        selection[0, len(response.prefix_ids) + position] = True
    dtype = getattr(torch, spec.accumulation_dtype)
    return aggregate_condition_hidden(hidden.to(dtype),
        "mean" if spec.pooling == "response_mean" else "last", selection).detach().cpu()


def _hooks(model: Any) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _source_hash(function: Any) -> str:
    return hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()


@contextmanager
def isolated_capture(model: Any):
    """Remove only the new passive Transformers recorders installed by capture.

    Transformers 5.21 intentionally keeps output-capturing hooks installed after
    hidden-state extraction. Our fresh-state protocol requires their removal.
    Preserve all pre-existing hooks and restore the lazy-install flags. Unknown
    new hooks cause an error rather than being silently removed.
    """
    if importlib.metadata.version("transformers") != "5.21.0":
        raise RuntimeError("Output-recorder cleanup is checked for Transformers 5.21.0 only.")
    sentinel = object()
    snapshot = [(m, dict(m._forward_hooks), dict(m._forward_pre_hooks),
        getattr(m, "_output_capturing_hooks_installed", sentinel)) for m in model.modules()]
    counts = {"native_recording_hooks_removed": 0}
    try:
        yield counts
    finally:
        unexpected = False
        for module, forward, pre, flag in snapshot:
            for identifier, hook in list(module._forward_hooks.items()):
                if identifier in forward:
                    continue
                if (getattr(hook, "__module__", None) == "transformers.utils.output_capturing"
                    and getattr(hook, "__qualname__", None) == "install_output_capuring_hook.<locals>.output_capturing_hook"):
                    del module._forward_hooks[identifier]
                    module._forward_hooks_with_kwargs.pop(identifier, None)
                    module._forward_hooks_always_called.pop(identifier, None)
                    counts["native_recording_hooks_removed"] += 1
                else:
                    unexpected = True
            if flag is sentinel:
                if hasattr(module, "_output_capturing_hooks_installed"):
                    delattr(module, "_output_capturing_hooks_installed")
            else:
                module._output_capturing_hooks_installed = flag
            if dict(module._forward_hooks) != forward or dict(module._forward_pre_hooks) != pre:
                unexpected = True
        if unexpected:
            raise RuntimeError("Unexpected hook change during capture; discard this model instance.")


@dataclass
class ConstructedDirections:
    raw: SteeringVector
    normalized: SteeringVector
    record: dict[str, Any]

    def save(self, destination: Path) -> dict[str, Any]:
        """Write both direction forms and pooled inputs to a new folder only."""
        destination = Path(destination)
        destination.mkdir(parents=False, exist_ok=False)
        files = {}
        for form, vector in (("raw", self.raw), ("normalized", self.normalized)):
            for layer, tensor in vector.directions.items():
                name = f"layer-{layer}-{form}.f32le"
                values = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
                blob = values.tobytes(order="C")
                with (destination / name).open("xb") as handle:
                    handle.write(blob)
                files[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                               "shape": list(tensor.shape), "dtype": "float32", "byte_order": "little"}
            # The package owns its interoperable vector serialization.
            name = form + ".svec"
            vector.save(str(destination / name))
            blob = (destination / name).read_bytes()
            files[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        header = {**self.record, "files": files}
        with (destination / "construction.json").open("x") as handle:
            json.dump(header, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return header


def construct_directions(model: Any, pairs: tuple[ConstructionPair, ...], spec: ConstructionSpec) -> ConstructedDirections:
    """Capture two teacher-forced forwards per pair with no generation or cache reuse.

    Steerability supplies raw block capture, response pooling, normalization and
    CAA-compatible vector storage. Study-specific code enforces pairing and
    response positions and averages one positive-minus-negative summary per pair.
    This function is not an execution-permission or behavioral-validation gate.
    """
    validate_inputs(spec, pairs)
    if importlib.metadata.version("steerability") != "0.5.7":
        raise RuntimeError("This integration was checked with Steerability 0.5.7 only.")
    if model.training or _hooks(model):
        raise ValueError("Construction requires an evaluation-mode model with no active hooks.")
    config = model.config
    if (getattr(config, "_name_or_path", None), getattr(config, "_commit_hash", None),
        getattr(config, "model_type", None)) != (spec.model_id, spec.model_revision, spec.model_type):
        raise ValueError("Loaded model identity differs from the construction specification.")
    layout = resolve_model_layout(model)
    for layer, path in spec.sites:
        if layer >= layout.num_layers or layout.layer_names[layer] != path:
            raise ValueError("Candidate layer/module mapping differs from the loaded model.")
    device = model.get_input_embeddings().weight.device
    vocabulary_size = model.get_input_embeddings().weight.shape[0]
    if any(t >= vocabulary_size for p in pairs for r in (p.positive, p.negative) for t in r.input_ids):
        raise ValueError("Construction contains an out-of-vocabulary token ID.")
    capture_dtype = str(model.get_input_embeddings().weight.dtype).removeprefix("torch.")
    if capture_dtype != spec.insertion_dtype:
        raise ValueError("The declared insertion dtype differs from the loaded model dtype.")
    sums = {i: torch.zeros((1, spec.hidden_width), dtype=getattr(torch, spec.accumulation_dtype)) for i, _ in spec.sites}
    rows = []
    removed_recorders = 0
    for pair in pairs:
        summaries = {}
        row = {"pair_id": pair.pair_id, "family_id": pair.family_id, "responses": {}}
        for label, response in (("positive", pair.positive), ("negative", pair.negative)):
            ids = torch.tensor([response.input_ids], dtype=torch.long, device=device)
            with isolated_capture(model) as cleanup, torch.inference_mode():
                states, mask = capture_hidden({"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                    model=model, batch_size=1, location=spec.location)
            removed_recorders += cleanup["native_recording_hooks_removed"]
            if mask is None or tuple(mask.shape) != tuple(ids.shape) or not torch.all(mask == 1):
                raise ValueError("Capture changed the unpadded token layout.")
            if _hooks(model):
                raise RuntimeError("A capture hook remained installed after teacher forcing.")
            summaries[label] = {layer: pool_response(states[layer], response, spec) for layer, _ in spec.sites}
            row["responses"][label] = {"prefix_ids": list(response.prefix_ids),
                "response_ids": list(response.response_ids),
                "selected_response_positions": list(response.selected_response_positions),
                "absolute_positions": [len(response.prefix_ids) + i for i in response.selected_response_positions],
                "pooled_states": {str(i): s.flatten().tolist() for i, s in summaries[label].items()}}
            del states
        for layer in sums:
            sums[layer] += summaries["positive"][layer] - summaries["negative"][layer]
        rows.append(row)
    directions = {layer: (value / len(pairs)).to(torch.float32) for layer, value in sums.items()}
    for value in directions.values():
        if not torch.isfinite(value).all() or not torch.isfinite(value.norm()) or value.norm() == 0:
            raise ValueError("A candidate direction is zero or nonfinite; normalization is undefined.")
    metadata = {**artifact_provenance_meta(model), "location": spec.location, "construction_spec": asdict(spec),
                "construction_spec_sha256": canonical_hash(asdict(spec)),
                "release_authorized": False, "behavioral_qualification": False}
    raw = SteeringVector(model_type=spec.model_type, directions=directions, meta=metadata)
    normalized = raw.normalized()
    raw.validate()
    normalized.validate()
    record = {"artifact_type": "RESPONSE_POSITION_DIRECTION_CONSTRUCTION", "schema_version": 1,
        **metadata, "scientific_finding": False, "synthetic_test": spec.synthetic_test,
        "estimator": "equal_pair_mean_positive_minus_negative", "pair_count": len(pairs),
        "capture_forward_count": 2 * len(pairs), "generation_calls": 0, "batch_size": 1,
        "capture_cache": False, "capture_dtype": capture_dtype, "hooks_after": _hooks(model),
        "native_recording_hooks_removed": removed_recorders,
        "raw_direction_norms": {str(i): float(v.norm()) for i, v in directions.items()},
        "normalization": "separate raw and per-layer L2-unit artifacts; neither is selected for release",
        "pairs": rows, "packages": {p: importlib.metadata.version(p) for p in ("steerability", "torch", "transformers")},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_source_sha256": {"capture": _source_hash(capture_hidden),
            "pooling": _source_hash(aggregate_condition_hidden), "vector": _source_hash(SteeringVector),
            "fingerprint": _source_hash(artifact_provenance_meta)}}
    return ConstructedDirections(raw, normalized, record)
