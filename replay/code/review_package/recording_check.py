"""Model-free end-to-end check of the prospective recording adapter.

Uses actual Steerability and Inspect with authored strings and character codes.
No research tokenizer, weights, direction, provider service or GPU is accessed.
"""
import argparse
import importlib.metadata
from pathlib import Path
from unittest.mock import patch
from .reuse import legacy
from .fixtures import comparison_parts
from .recording import prepare_plan, inspect_sample, collect_saved_inspect, save_new
from .inspect_bridge import no_network, AuthoredResponsePipeline
from .workflow import NegotiationScorer
from .comparison import analyze_saved
from .cli import write_new
from .outcomes import add_agreement_outcomes


def run_recording_check(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    versions = {p: importlib.metadata.version(p) for p in ("steerability", "inspect-ai")}
    if versions != {"steerability": "0.5.7", "inspect-ai": "0.3.286"}:
        raise RuntimeError("Use the existing pinned environment")
    manifest, cases, trials = comparison_parts()
    cases = [cases[0]]
    trials = [r for r in trials if r["case_id"] == cases[0]["case_id"]]
    manifest["planned_trials"] = [{k: r[k] for k in ("trial_id", "case_id", "condition")} for r in trials]
    # A malformed decision tests the same scoring guard in Inspect and analysis.
    trials[-1]["raw_response"] = '{"decision":[],"package":{}}'
    plan = prepare_plan(manifest, cases, trials,
        tokenize=lambda messages: [ord(c) + 2 for c in legacy.canonical(messages)],
        tokenizer_record={"name": "SYNTHETIC_CHARACTER_CODES_NOT_MODEL_TOKENS", "model_loaded": False})
    plan_path = directory / "original-plan.json"
    save_new(plan_path, plan)
    pin = legacy.sha(plan_path)
    save_new(directory / "plan-frozen-before-dispatch.json", {"plan_sha256": pin,
        "fixture_only": True, "model_execution": False, "execution_authorized": False})
    deliveries, scores = {}, {}
    import inspect_ai._util.appdirs as appdirs
    with no_network(), patch.object(appdirs, "user_data_path", lambda _: directory / 'local-data'), patch.object(appdirs, "user_cache_path", lambda _: directory / 'local-cache'):
        from inspect_ai import Task, eval as evaluate
        from inspect_ai.log import read_eval_log
        from inspect_ai.scorer import Score, scorer as inspect_scorer
        from inspect_ai.solver import generate
        from steerability.evaluation.provider import as_inspect_model, ProviderOptions
        scoring = NegotiationScorer()
        @inspect_scorer(metrics=[])
        def check_score():
            async def score(state, target):
                value = scoring.score(cases[0], state.output.completion)
                return Score(value=int(value["valid"]), answer=state.output.completion,
                    metadata={"scoring": value, "fixture_only": True})
            return score
        for original, trial in zip(plan["trials"], trials):
            messages = original["execution_input"]["messages"]
            pipeline = AuthoredResponsePipeline({legacy.canonical(messages): trial["raw_response"]})
            model = as_inspect_model(pipeline, model_name=manifest["model_id"],
                options=ProviderOptions(max_batch_size=1, default_max_tokens=2048))
            task = Task(name="synthetic-recording-check", dataset=[inspect_sample(original, pin)],
                solver=generate(), scorer=check_score(),
                metadata={"fixture_only": True, "model_execution": False, "authored_responses": True})
            logs = evaluate(task, model=model, log_dir=str(directory / "inspect-originals"),
                log_format="json", display="none", notification=False, ctl_server=False, acp_server=False,
                checkpoint=False, epochs=1, retry_on_error=0, max_retries=0, max_samples=1, max_tasks=1,
                max_connections=1, cache=False, fail_on_error=True, score_on_error=False,
                sample_shuffle=False, max_tokens=2048, temperature=0)
            if len(logs) != 1 or logs[0].status != "success" or len(pipeline.calls) != 1 or pipeline.outputs:
                raise RuntimeError("Incomplete or repeated synthetic attempt")
            saved = read_eval_log(logs[0].location)
            sample = saved.samples[0]
            if sample.output.completion != trial["raw_response"] or pipeline.calls[0]["input"] != messages:
                raise RuntimeError("Provider changed the input or authored response")
            if pipeline.tokenizer.encode(legacy.canonical(pipeline.calls[0]["input"])) != original["execution_input"]["input_ids"]:
                raise RuntimeError("Provider input tokens differ from the original record")
            scores[trial["trial_id"]] = next(iter(sample.scores.values())).metadata["scoring"]
            deliveries[trial["trial_id"]] = Path(logs[0].location)
    outer = {"run_id": manifest["run_id"], "status": "completed", "retry_count": 0,
        "attempt_count": len(deliveries), "trial_ids": [r["trial_id"] for r in trials],
        "fixture_only": True, "model_execution": False}
    path = collect_saved_inspect(directory / "delivery", manifest, cases, plan_path, pin, deliveries, outer)
    analysis = analyze_saved(path)
    for row in analysis["all_trials"]:
        expected = scores[row["trial_id"]]["detail"]
        if row["score"] != add_agreement_outcomes(expected, cases[0]):
            raise RuntimeError("Inspect and independently recalculated scores differ")
    write_new(directory / "report", analysis)
    result = {"status": "PASSED", "fixture_only": True, "evidence_label": "SYNTHETIC_TEST_ONLY",
        "empirical_evidence": False, "model_execution": False, "model_loads": 0, "provider_service_calls": 0,
        "authored_dispatches": len(deliveries), "conditions": [r["condition"] for r in trials],
        "invalid_responses_retained": sum(not v["valid"] for v in scores.values()),
        "versions": versions, "plan_sha256": pin, "network_disabled": True,
        "canonical_messages_checked_at_provider": True, "synthetic_input_tokens_checked": True,
        "inspect_scores_equal_recalculated_scores": True, "original_logs_preserved": True,
        "real_runtime_qualified": False, "deployment": False}
    save_new(directory / "checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(legacy.canonical(run_recording_check(args.output)))


if __name__ == "__main__":
    main()
