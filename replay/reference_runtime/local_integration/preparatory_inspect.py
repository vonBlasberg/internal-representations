"""Use the existing Steerability/Inspect connection for a prepared phase.

Model and direction loading are injected by an approved worker. This module has
no provider or model-download entry point and performs no work on import.
"""
from dataclasses import asdict
import gc
import hashlib
from pathlib import Path

from inspect_ai import Task, eval as evaluate
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from .preparatory_phases import identity, save_new
from .research_pipeline import EditSpec, RecordedSteeringPipeline


class InspectPhaseGenerator:
    """One fresh admitted model per trial; no retries or shared inference state."""
    def __init__(self, plan, contract, *, model_factory, direction_loader):
        if identity(asdict(contract)) != identity(plan["generation_contract"]):
            raise ValueError("Runtime contract differs from the prepared phase.")
        self.plan, self.contract = plan, contract
        self.model_factory, self.direction_loader = model_factory, direction_loader
        self.failed = False
        self.seen = set()
        self.next_index = 0

    def __call__(self, trial, directory):
        if self.failed or trial["trial_id"] in self.seen:
            raise RuntimeError("This connection stops after failure and cannot repeat a trial.")
        if self.next_index >= len(self.plan["trials"]) or identity(trial) != identity(self.plan["trials"][self.next_index]):
            raise ValueError("Trial order or content differs from the prepared phase.")
        self.seen.add(trial["trial_id"])
        self.next_index += 1
        directory = Path(directory)
        pipeline = None
        model = tokenizer = direction = None
        try:
            model, tokenizer = self.model_factory(directory)
            expected = tuple(tokenizer.apply_chat_template(trial["messages"], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False))
            save_new(directory / "input.json", {"input_token_ids": list(expected),
                "input_token_ids_sha256": identity(expected), "messages_sha256": trial["messages_sha256"]})
            edit = EditSpec(**trial["edit"]) if trial["edit"] else None
            direction = self.direction_loader(trial["edit"], directory) if edit else None
            observed = ((edit.layer, edit.module_path),) if edit else ()
            pipeline = RecordedSteeringPipeline(model, tokenizer, self.contract,
                edit=edit, direction=direction, observe_sites=observed)
            messages = [ChatMessageSystem(content=trial["messages"][0]["content"]),
                        ChatMessageUser(content=trial["messages"][1]["content"])]
            metadata = {"run_id": self.plan["run_id"], "phase": self.plan["phase"],
                "plan_sha256": identity(self.plan), "fixture_only": self.plan["fixture_only"],
                "model_id": self.plan["model_id"], "model_revision": self.plan["model_revision"],
                **{k: trial[k] for k in ("trial_id", "case_id", "family_id", "arm_id", "kind", "edit", "messages_sha256", "case_content_sha256", "output_contract_version")}}
            task = Task(name="preparatory-" + self.plan["phase"],
                dataset=[Sample(id=trial["trial_id"], input=messages, metadata={"preparatory_trial": metadata})],
                solver=generate())
            with pipeline.trial(trial["trial_id"], expected):
                logs = evaluate(task, model=pipeline.inspect_model(), log_dir=str(directory / "inspect"),
                    log_format="json", display="none", notification=False, ctl_server=False, acp_server=False,
                    checkpoint=False, epochs=1, retry_on_error=0, max_retries=0, max_samples=1,
                    max_tasks=1, max_connections=1, cache=False, fail_on_error=False, score_on_error=False,
                    sample_shuffle=False, max_tokens=self.contract.max_new_tokens, temperature=0)
                if len(logs) != 1:
                    raise RuntimeError("Expected one Inspect log.")
                saved = read_eval_log(logs[0].location)
                if saved.status != "success" or len(saved.samples or []) != 1 or saved.samples[0].error:
                    raise RuntimeError("Inspect reported a failed or incomplete sample.")
                sample = saved.samples[0]
                if sample.id != trial["trial_id"] or sample.metadata["preparatory_trial"] != metadata:
                    raise RuntimeError("Inspect trial metadata changed.")
                receipt = pipeline.records[-1]
                raw = sample.output.completion
                if raw != tokenizer.decode(receipt["generated_token_ids"], skip_special_tokens=True):
                    raise RuntimeError("Inspect text differs from the recorded generated tokens.")
                path = Path(logs[0].location)
                return {"raw_response": raw, "generated_token_ids": receipt["generated_token_ids"],
                    "source_log_path": str(path.relative_to(directory)),
                    "source_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_sample_id": sample.id, "metadata": metadata,
                    "execution_record": "state_execution.json"}
        except Exception:
            self.failed = True
            raise
        finally:
            if pipeline is not None:
                save_new(directory / "state_execution.json", {"records": pipeline.records})
            del pipeline, model, tokenizer, direction
            gc.collect()
