"""Exercise real Steerability/Inspect interfaces with authored responses, never a model.

No loader, credentials, network endpoint, direction construction or live-run mode
exists in this module. Character codes are synthetic tokens, not Qwen tokens.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import importlib.metadata
import socket
from .reuse import legacy
from .workflow import Study, evaluate_saved


@contextmanager
def no_network():
    def deny(*args, **kwargs):
        raise RuntimeError("Network disabled in package software checks")
    with patch.object(socket.socket, "connect", deny), patch.object(socket.socket, "connect_ex", deny), patch.object(socket, "create_connection", deny):
        yield


def candidate_configurations():
    from steerability.algorithms.core.specs import ControlSpec
    from steerability.algorithms.core.sweeps import expand_configurations
    from steerability.algorithms.state_control.caa.control import CAA
    config = ControlSpec(control_cls=CAA, name="internal", vars={"layer_id": [14, 15], "multiplier": [0.5, 1.0, 2.0]},
                         params={"token_scope": "last_k", "last_k": 1, "normalize_vector": False, "use_norm_preservation": False})
    points = list(expand_configurations({"internal": [config]}, base_model_name_or_path="SYNTHETIC_NO_MODEL"))
    return [{"config_id": p.config_id, "params": p.params, "model_execution": False} for p in points]


class CharacterTokenizer:
    chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]
    def decode(self, ids, **kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            return [self.decode(row) for row in ids]
        return ''.join(chr(i - 2) for i in ids if i > 1)


class AuthoredResponsePipeline:
    _is_steered = True
    supports_batching = False
    supports_concurrent_generate = False
    controls = ()
    def __init__(self, outputs):
        self.tokenizer = CharacterTokenizer()
        self.outputs = dict(outputs)
        self.calls = []
    def generate(self, *, messages=None, text=None, **kwargs):
        import torch
        from steerability.algorithms.core.output import Output
        if text is not None or not messages:
            raise RuntimeError("Only the declared synthetic messages are supported")
        single = isinstance(messages[0], dict)
        conversations = [messages] if single else messages
        results = []
        for conversation in conversations:
            key = legacy.canonical(conversation)
            if key not in self.outputs:
                raise RuntimeError("Undeclared input or repeated synthetic attempt")
            answer = self.outputs.pop(key)
            self.calls.append({"input": conversation, "response": answer, "settings": kwargs})
            results.append(Output(output_ids=torch.tensor([self.tokenizer.encode(answer)], dtype=torch.long),
                adapted_input_ids=torch.tensor([self.tokenizer.encode(key)], dtype=torch.long), finish_reasons=("eos",)))
        return results[0] if single else results


def run_synthetic_inspect(study, scorer, answers, directory):
    """Use the unchanged provider and Inspect logger/scorer. All answers are authored."""
    if study.fixture_only is not True:
        raise ValueError("Synthetic studies only")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    versions = {p: importlib.metadata.version(p) for p in ("steerability", "inspect-ai")}
    if versions != {"steerability": "0.5.7", "inspect-ai": "0.3.286"}:
        raise ValueError("Use the reviewed pinned integration environment")
    import inspect_ai._util.appdirs as inspect_appdirs
    # Inspect has no data-directory environment switch on this pinned macOS
    # version. Isolate its application data/cache in this synthetic check only.
    # This does not alter scoring, provider, serializer or recorded model inputs.
    with no_network(), patch.object(inspect_appdirs, "user_data_path", lambda _: directory / 'local-data'), patch.object(inspect_appdirs, "user_cache_path", lambda _: directory / 'local-cache'):
        from inspect_ai import Task, eval as evaluate
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ChatMessageUser
        from inspect_ai.scorer import Score, scorer as inspect_scorer
        from inspect_ai.solver import generate
        from inspect_ai.analysis import samples_df
        from steerability.evaluation.provider import as_inspect_model, ProviderOptions
        samples, outputs = [], {}
        for case in study.cases:
            for condition in study.conditions:
                tid = case["case_id"] + ':' + condition
                content = 'SYNTHETIC SOFTWARE TEST ONLY\n' + tid + '\n' + legacy.canonical(case)
                messages = [{"role": "user", "content": content}]
                outputs[legacy.canonical(messages)] = answers[(case["case_id"], condition)]
                samples.append(Sample(id=tid, input=[ChatMessageUser(content=content)],
                    metadata={"fixture_only": True, "case": case, "condition": condition,
                              "synthetic_input": messages, "scorer": scorer.name}))
        pipeline = AuthoredResponsePipeline(outputs)
        model = as_inspect_model(pipeline, model_name='synthetic-review-only', options=ProviderOptions(max_batch_size=1, default_max_tokens=2048))
        @inspect_scorer(metrics=[])
        def fixed_score():
            async def score(state, target):
                result = scorer.score(state.metadata["case"], state.output.completion)
                return Score(value=int(result["valid"]), answer=state.output.completion,
                             metadata={"scoring": result, "fixture_only": True})
            return score
        task = Task(dataset=samples, solver=generate(), scorer=fixed_score(), name='synthetic_' + scorer.name,
                    metadata={"fixture_only": True, "model_execution": False})
        logs = evaluate(task, model=model, log_dir=str(directory / 'inspect'), log_format='json', display='none',
            notification=False, ctl_server=False, acp_server=False, checkpoint=False, epochs=1, retry_on_error=0,
            max_retries=0, max_samples=1, max_tasks=1, max_connections=1, cache=False, fail_on_error=True,
            score_on_error=False, sample_shuffle=False, max_tokens=2048, temperature=0)
        if len(logs) != 1 or logs[0].status != 'success' or len(logs[0].samples or []) != len(samples):
            raise RuntimeError("Synthetic Inspect delivery was incomplete")
        if len(pipeline.calls) != len(samples) or pipeline.outputs:
            raise RuntimeError("Repeated or missing synthetic attempts")
        rows = []
        for sample in logs[0].samples:
            if sample.error:
                raise RuntimeError("Synthetic Inspect sample failed")
            meta = sample.metadata
            expected = answers[(meta["case"]["case_id"], meta["condition"])]
            if sample.output.completion != expected:
                raise RuntimeError("Recorded response differs from authored bytes")
            rows.append({"case_id": meta["case"]["case_id"], "condition": meta["condition"],
                "execution_status": "COMPLETED", "raw_response": sample.output.completion,
                "source": {"log": str(Path(logs[0].location).relative_to(directory)), "sample_id": sample.id}})
        result = evaluate_saved(study, scorer, rows)
        frame = samples_df([logs[0].location], quiet=True)
        if len(frame) != len(rows):
            raise RuntimeError("Inspect dataframe dropped samples")
        # Authored, non-sensitive synthetic rows only. No real log is exported here.
        frame.to_json(directory / 'inspect_samples.json', orient='records', indent=2)
        from inspect_ai.log._bundle import bundle_log_dir
        bundle_log_dir(log_dir=str(directory / 'inspect'), output_dir=str(directory / 'inspect-view'), overwrite=False)
        result["integration"] = {"versions": versions, "inspect_samples": len(frame),
            "provider": "Steerability as_inspect_model", "actual_model_loaded": False,
            "synthetic_dispatches": len(pipeline.calls), "retry_count": 0, "network_disabled": True,
            "input_path": "synthetic character tokenizer", "real_runtime_qualified": False}
        result["integration"]["inspect_view_bundle"] = str(directory / 'inspect-view' / 'index.html')
        return result
