"""Recorded Steerability generation for an already-loaded, admitted model.

There is no model loader, provider connection, study selection or retry here.
Execution approval, case admission and persistent journals belong to the caller.
Steerability applies the edit; these observers verify and record what it did.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
from pathlib import Path
import sys
import threading
from unittest.mock import patch
import weakref

import torch
from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.state_control.caa.control import CAA
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.evaluation.provider import ProviderOptions, as_inspect_model

from .direction_construction import canonical_hash

# Reuse the preserved native-byte and weight-recording implementation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "negotiation-steering"))
from worker.receipts import summarize_cache, tensor_sha256, weight_fingerprint

_LOCKS = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


@dataclass(frozen=True)
class GenerationContract:
    model_id: str
    model_revision: str
    model_type: str
    dtype: str
    chat_template_sha256: str
    generation_config_sha256: str
    max_input_tokens: int
    max_new_tokens: int
    min_new_tokens: int
    eos_token_ids: tuple[int, ...] | None
    pad_token_id: int
    max_generation_calls: int
    synthetic_test: bool
    attention_implementation: str = "eager"

    def generation_kwargs(self):
        return dict(do_sample=False, num_beams=1, num_return_sequences=1,
            use_cache=True, cache_implementation="dynamic", max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens, eos_token_id=None if self.eos_token_ids is None else list(self.eos_token_ids),
            pad_token_id=self.pad_token_id, return_dict_in_generate=True, output_scores=True,
            output_hidden_states=False, output_attentions=False)


@dataclass(frozen=True)
class EditSpec:
    layer: int
    module_path: str
    coefficient: float
    direction_sha256: str


def _hooks(model):
    return {name: (dict(module._forward_hooks), dict(module._forward_pre_hooks))
            for name, module in model.named_modules()}


def _tensor(output):
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 1:
        raise RuntimeError("A recorded block output must be a batch-one rank-three tensor.")
    return value


class RecordedSteeringPipeline(SteeringPipeline):
    """One serialized generation per declared trial, with exact input binding."""

    def __init__(self, model, tokenizer, contract: GenerationContract, *,
                 edit: EditSpec | None = None, direction: torch.Tensor | None = None,
                 observe_sites: tuple[tuple[int, str], ...] = ()):
        if not isinstance(model, torch.nn.Module) or isinstance(tokenizer, str):
            raise TypeError("Supply already-loaded model and tokenizer objects, not hub names.")
        self.research_model, self.research_tokenizer, self.contract = model, tokenizer, contract
        self.edit, self.observation_sites = edit, observe_sites
        self.records, self.seen_trials = [], set()
        self.active_trial, self.failed, self.generation_calls = None, False, 0
        self.applied_direction = None
        self.versions = {name: importlib.metadata.version(name) for name in
                         ("steerability", "inspect-ai", "transformers", "torch")}
        if {k: self.versions[k] for k in ("steerability", "inspect-ai", "transformers")} != {
                "steerability": "0.5.7", "inspect-ai": "0.3.286", "transformers": "5.21.0"}:
            raise ValueError("Use the pinned package versions checked for this connection.")
        if self.versions["torch"].split("+")[0] != "2.15.0":
            raise ValueError("Use the pinned PyTorch version; GPU build suffixes are recorded separately.")
        self._validate_model()
        layout = resolve_model_layout(model)
        if len({layer for layer, _ in observe_sites}) != len(observe_sites):
            raise ValueError("Observation sites must be unique.")
        for layer, path in observe_sites:
            if type(layer) is not int or layer < 0 or layer >= layout.num_layers or layout.layer_names[layer] != path:
                raise ValueError("Observation index and module path do not match the loaded model.")
        controls = []
        if (edit is None) != (direction is None):
            raise ValueError("An edit and its direction must be supplied together.")
        if edit is not None:
            if (edit.layer, edit.module_path) not in observe_sites:
                raise ValueError("The edited site must be observed.")
            if type(edit.coefficient) not in (int, float) or not math.isfinite(edit.coefficient):
                raise ValueError("The coefficient must be finite.")
            width = model.get_input_embeddings().weight.shape[1]
            if direction.dtype != torch.float32 or tuple(direction.shape) != (1, width) or not torch.isfinite(direction).all():
                raise ValueError("Supply the recorded finite float32 direction with shape [1, hidden width].")
            if tensor_sha256(direction) != edit.direction_sha256:
                raise ValueError("Direction bytes differ from the declared hash.")
            # The preserved native runtime casts BEFORE scaling. CAA scales its stored
            # tensor before casting, so store an explicitly cast copy for application.
            self.applied_direction = direction.detach().clone().to(
                device=model.get_input_embeddings().weight.device, dtype=getattr(torch, contract.dtype))
            if not torch.isfinite(self.applied_direction * edit.coefficient).all():
                raise ValueError("The applied direction overflows the model dtype.")
            controls = [CAA(steering_vector=SteeringVector(model_type=contract.model_type,
                directions={edit.layer: self.applied_direction}), layer_id=edit.layer,
                multiplier=edit.coefficient, token_scope="last_k", last_k=1,
                normalize_vector=False, use_norm_preservation=False)]
        with _REGISTRY_LOCK:
            self.model_lock = _LOCKS.setdefault(model, threading.Lock())
        super().__init__(model=model, tokenizer=tokenizer, controls=controls,
                         device=str(model.get_input_embeddings().weight.device), trust_remote_code=False)
        self.steer()
        self._validate_model()

    def _validate_model(self):
        c, model, tokenizer = self.contract, self.research_model, self.research_tokenizer
        for field in ("max_input_tokens", "max_new_tokens", "max_generation_calls"):
            if type(getattr(c, field)) is not int or getattr(c, field) < 1:
                raise ValueError("Explicit positive token and call limits are required.")
        if c.max_new_tokens > 4096 or type(c.min_new_tokens) is not int or not 0 <= c.min_new_tokens <= c.max_new_tokens:
            raise ValueError("Invalid generation length bounds.")
        if type(c.synthetic_test) is not bool or c.dtype not in ("float32", "bfloat16"):
            raise ValueError("Declare the test status and supported model dtype.")
        if len(c.model_revision) != 40 or any(ch not in "0123456789abcdef" for ch in c.model_revision):
            raise ValueError("An exact model revision is required.")
        if (model.config._name_or_path, getattr(model.config, "_commit_hash", None), model.config.model_type) != (
                c.model_id, c.model_revision, c.model_type):
            raise ValueError("Loaded model identity differs from the contract.")
        if model.training or any(a or b for a, b in _hooks(model).values()):
            raise ValueError("The model must be in evaluation mode with no active hooks.")
        if str(model.get_input_embeddings().weight.dtype) != "torch." + c.dtype:
            raise ValueError("Loaded model precision differs from the contract.")
        if getattr(model.config, "_attn_implementation", None) != c.attention_implementation:
            raise ValueError("Attention implementation differs from the contract.")
        template = tokenizer.chat_template
        if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != c.chat_template_sha256:
            raise ValueError("Chat template differs from the contract.")
        if tokenizer.padding_side != "left" or tokenizer.pad_token_id != c.pad_token_id:
            raise ValueError("Tokenizer padding configuration differs from the contract.")
        if canonical_hash(model.generation_config.to_dict()) != c.generation_config_sha256:
            raise ValueError("Loaded generation defaults changed.")
        vocab = model.get_input_embeddings().weight.shape[0]
        ids = (c.pad_token_id,) + (c.eos_token_ids or ())
        if any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("Invalid special-token IDs.")
        if c.eos_token_ids is not None and (not c.eos_token_ids or len(set(c.eos_token_ids)) != len(c.eos_token_ids)):
            raise ValueError("EOS IDs must be an explicit nonempty unique sequence, or None.")

    @contextmanager
    def trial(self, trial_id: str, expected_input_ids: tuple[int, ...]):
        if self.failed or self.active_trial is not None or not trial_id or trial_id in self.seen_trials:
            raise RuntimeError("Trial reuse, nested trials and retries after failure are forbidden.")
        vocab = self.research_model.get_input_embeddings().weight.shape[0]
        if not expected_input_ids or len(expected_input_ids) > self.contract.max_input_tokens or any(
                type(i) is not int or not 0 <= i < vocab for i in expected_input_ids):
            raise ValueError("Expected input tokens must be explicit, in vocabulary and within the limit.")
        self.active_trial = (trial_id, tuple(expected_input_ids))
        self.seen_trials.add(trial_id)
        previous_count = len(self.records)
        try:
            yield self
            if len(self.records) != previous_count + 1 or self.records[-1]["status"] != "completed":
                raise RuntimeError("The trial did not produce exactly one completed generation record.")
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active_trial = None

    def generate(self, *args, **kwargs):
        if self.failed or self.active_trial is None or self.generation_calls >= self.contract.max_generation_calls:
            raise RuntimeError("Generation requires a new admitted trial within the call limit.")
        trial_id, expected_ids = self.active_trial
        if any(r["trial_id"] == trial_id for r in self.records):
            raise RuntimeError("A trial may dispatch only once.")
        if not self.model_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent access to this recorded model is forbidden.")
        record = {"trial_id": trial_id, "status": "started", "generation_call_count": 0,
            "retry_count": 0, "contract": asdict(self.contract), "package_versions": self.versions,
            "contract_sha256": canonical_hash(asdict(self.contract)),
            "edit": asdict(self.edit) if self.edit else None, "state_observations": [],
            "application_rule": "cast_direction_to_model_dtype_then_scale_then_add_at_last_active_prefill_position_and_each_decode_position",
            "applied_direction_sha256": None if self.applied_direction is None else tensor_sha256(self.applied_direction),
            "synthetic_test": self.contract.synthetic_test, "scientific_evidence": False,
            "research_model_qualified": False, "release_authorized": False}
        self.records.append(record)
        handles, pending = [], {}
        try:
            self._validate_model()
            self._validate_request(kwargs)
            before_hooks = _hooks(self.research_model)
            record["loaded_generation_config"] = self.research_model.generation_config.to_dict()
            record["weights_before"] = weight_fingerprint(self.research_model)
            for layer, path in self.observation_sites:
                def before(module, args, keywords, output, path=path):
                    if path in pending:
                        raise RuntimeError("A block was entered again before its observation completed.")
                    positions = keywords.get("position_ids")
                    pending[path] = (_tensor(output).detach().clone(),
                        None if positions is None else positions.detach().cpu().tolist())
                handles.append(self.research_model.get_submodule(path).register_forward_hook(before, with_kwargs=True))
            original_generate = self.research_model.generate

            def recorded_generate(*positional, **call):
                if positional or record["generation_call_count"]:
                    raise RuntimeError("Only one keyword-only model generation is admitted.")
                ids, mask = call.get("input_ids"), call.get("attention_mask")
                if not torch.is_tensor(ids) or tuple(ids.shape) != (1, len(expected_ids)) or tuple(ids[0].tolist()) != expected_ids:
                    raise ValueError("Rendered input IDs differ from the admitted trial.")
                if not torch.is_tensor(mask) or mask.shape != ids.shape or not torch.all(mask == 1):
                    raise ValueError("Only an unpadded batch-one input is admitted.")
                required = self.contract.generation_kwargs()
                if any(call.get(k) != v for k, v in required.items()) or set(call) - set(required) - {"input_ids", "attention_mask"}:
                    raise ValueError("The package changed the effective generation arguments.")
                record.update(input_token_ids=list(expected_ids), attention_mask=mask[0].cpu().tolist(),
                              generation_call=required, generation_call_count=1)
                self.generation_calls += 1
                after_handles = []
                try:
                    for layer, path in self.observation_sites:
                        def after(module, args, output, layer=layer, path=path):
                            before, positions = pending.pop(path)
                            actual = _tensor(output).detach()
                            expected = before.clone()
                            if self.edit and layer == self.edit.layer:
                                expected[:, -1, :] += self.applied_direction * self.edit.coefficient
                            if not torch.equal(actual, expected):
                                raise RuntimeError("The realized state edit differs from the declared operation.")
                            delta = actual.float() - before.float()
                            record["state_observations"].append(dict(layer=layer, module_path=path,
                                forward_index=sum(r["layer"] == layer for r in record["state_observations"]),
                                shape=list(actual.shape), dtype=str(actual.dtype), position_ids=positions,
                                before_sha256=tensor_sha256(before), after_sha256=tensor_sha256(actual),
                                changed_batch_token_indices=(delta != 0).any(-1).nonzero().tolist(),
                                realized_l2=float(delta.norm()), exact_expected_state=True))
                        after_handles.append(self.research_model.get_submodule(path).register_forward_hook(after))
                    output = original_generate(**call)
                    generated = output.sequences[0, len(expected_ids):]
                    if not torch.equal(output.sequences[:, :len(expected_ids)], ids) or len(output.scores) != generated.numel():
                        raise RuntimeError("Generated sequence, prefix and score records disagree.")
                    record["generated_token_ids"] = generated.cpu().tolist()
                    record["processed_logit_steps"] = [dict(step=i, selected_token_id=int(generated[i]),
                        shape=list(score.shape), storage_dtype=str(score.dtype), native_sha256=tensor_sha256(score))
                        for i, score in enumerate(output.scores)]
                    record["terminal_reason"] = "eos" if generated.numel() and int(generated[-1]) in (self.contract.eos_token_ids or ()) else "max_new_tokens"
                    record["cache_state"] = summarize_cache(getattr(output, "past_key_values", None), model=self.research_model)
                    sequences = output.sequences
                    output.scores = None
                    return sequences  # Steerability's HF session expects a tensor.
                finally:
                    for handle in after_handles:
                        handle.remove()

            request = dict(kwargs)
            for key in ("temperature", "top_p", "seed_scope"):
                request.pop(key, None)
            request.update(self.contract.generation_kwargs())
            if request.get("messages") is not None:
                request["chat_template_kwargs"] = {"enable_thinking": False}
            with patch.object(self.research_model, "generate", new=recorded_generate), torch.inference_mode():
                result = super().generate(*args, **request)
            if pending or record["generation_call_count"] != 1:
                raise RuntimeError("Incomplete generation or state observations.")
            steps = len(record["generated_token_ids"])
            if any(sum(r["layer"] == layer for r in record["state_observations"]) != steps for layer, _ in self.observation_sites):
                raise RuntimeError("The recorded edit schedule does not cover every generation step.")
            record["status"] = "completed"
            return result
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.failed = True
            raise
        finally:
            for handle in handles:
                handle.remove()
            try:
                if "weights_before" in record:
                    record["weights_after"] = weight_fingerprint(self.research_model)
                    record["weights_unchanged"] = record["weights_before"] == record["weights_after"]
                    record["hooks_restored"] = _hooks(self.research_model) == before_hooks
                    if not record["weights_unchanged"] or not record["hooks_restored"]:
                        record["status"], self.failed = "failed", True
                        raise RuntimeError("Weights or hook registrations were not restored.")
            finally:
                self.model_lock.release()

    def _validate_request(self, kwargs):
        allowed = {"input_ids", "attention_mask", "messages", "text", "return_output", "runtime_kwargs",
                   "chat_template_kwargs", "temperature", "top_p", "seed_scope"} | set(self.contract.generation_kwargs())
        if set(kwargs) - allowed or kwargs.get("runtime_kwargs"):
            raise ValueError("Undeclared generation settings or runtime controls are forbidden.")
        if kwargs.get("temperature", 0) != 0 or kwargs.get("top_p", 1) != 1:
            raise ValueError("Only the declared greedy generation is admitted.")
        if kwargs.get("chat_template_kwargs", {"enable_thinking": False}) != {"enable_thinking": False}:
            raise ValueError("Thinking mode must remain disabled.")
        for key, value in self.contract.generation_kwargs().items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError("Request conflicts with declared generation setting: " + key)

    def inspect_model(self):
        return as_inspect_model(self, options=ProviderOptions(max_batch_size=1,
            default_max_tokens=self.contract.max_new_tokens, reasoning_tags=None,
            on_unsupported_param="raise", chat_template_kwargs={"enable_thinking": False}),
            model_name=self.contract.model_id)
