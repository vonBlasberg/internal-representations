"""Bind a saved comparison to original inputs and all attempts. Never qualify a GPU route.

The plan hash must come from a separately reviewed pre-execution record. Hashing
an analyst-created plan after seeing results does not establish that provenance.
"""
import hashlib
from pathlib import Path
from .reuse import legacy, binding, protocol
from .comparison import validate_comparison, bind_all_samples
from .inputs import render_messages, validate_messages
from .case_validation import require_prompt_review


def identity(value):
    return hashlib.sha256(legacy.canonical(value).encode()).hexdigest()


def verify_inputs_and_attempts(manifest_path, plan_path, expected_plan_sha256):
    root = Path(manifest_path).resolve().parent
    if legacy.sha(Path(plan_path)) != expected_plan_sha256:
        raise legacy.AnalysisError("Execution plan differs from the separately pinned original")
    plan = legacy.load_json(Path(plan_path))
    manifest, cases, trials, hashes = legacy.read_bundle(Path(manifest_path))
    spec, digest = protocol()
    validate_comparison(manifest, cases, trials, spec, digest, allow_partial=False)
    bind_all_samples(manifest, trials, root)
    if (plan.get("artifact_type") != "ORIGINAL_INPUT_AND_ATTEMPT_PLAN" or
            plan.get("run_id") != manifest["run_id"] or plan.get("fixture_only") is not manifest["fixture_only"]):
        raise legacy.AnalysisError("Original plan identity differs")
    if {c["case_id"]: identity(c) for c in cases} != plan.get("case_sha256"):
        raise legacy.AnalysisError("Case content differs from original execution inputs")
    if not manifest['fixture_only']:
        require_prompt_review(plan.get('prompt_review'), cases, spec)
    expected = plan.get("trials", [])
    if len({r["trial_id"] for r in expected}) != len(expected):
        raise legacy.AnalysisError("Duplicate logical trial in original plan")
    expected = {r["trial_id"]: r for r in expected}
    if set(expected) != {r["trial_id"] for r in trials}:
        raise legacy.AnalysisError("Original plan and delivered trial inventory differ")
    logs = {r["source_log_path"] for r in trials}
    if logs != set(plan.get("source_logs", [])) or len(logs) != len(plan.get("source_logs", [])):
        raise legacy.AnalysisError("Original source-log inventory differs")
    documents = {}
    case_map = {case["case_id"]: case for case in cases}
    for trial in trials:
        original = expected[trial["trial_id"]]
        for key in binding.METADATA_FIELDS:
            if original.get(key) != trial.get(key) or type(original.get(key)) is not type(trial.get(key)):
                raise legacy.AnalysisError("Original execution setting differs: " + key)
        if not manifest['fixture_only'] or 'source_sample_id' in original or 'source_log_path' in original:
            if (original.get('source_log_path') != trial['source_log_path']
                    or 'source_sample_id' not in original
                    or binding._id(original['source_sample_id']) != binding._id(trial['source_sample_id'])):
                raise legacy.AnalysisError('Physical source sample differs from its frozen per-trial identity')
        path = legacy._input_path(root, trial["source_log_path"])
        if path not in documents:
            doc = binding._read_exact_source(path, trial["source_log_sha256"])
            documents[path] = binding._source_index(doc, manifest["fixture_only"])
        kind, index = documents[path]
        sample = index[binding._id(trial["source_sample_id"])]
        if not manifest['fixture_only'] or kind == 'inspect_json_v2':
            if (sample.get('metadata') or {}).get('original_plan_sha256') != expected_plan_sha256:
                raise legacy.AnalysisError('Source sample does not bind the separately pinned original plan')
        actual = (sample.get("metadata") or {}).get("execution_input")
        wanted = original.get("execution_input")
        if not isinstance(wanted, dict) or not isinstance(actual, dict):
            raise legacy.AnalysisError("Original execution inputs are missing")
        messages = validate_messages(wanted.get("messages"))
        validate_messages(actual.get("messages"))
        recorded_messages = validate_messages(sample.get("input"), inspect_serialized=True)
        if (messages != render_messages(case_map[trial["case_id"]], trial["condition"], spec)
                or actual != wanted or recorded_messages != messages):
            raise legacy.AnalysisError("Original model-visible messages differ or are absent")
        tokens = wanted.get("input_ids")
        if (not isinstance(tokens, list) or not tokens or any(type(i) is not int or i < 0 for i in tokens)
                or wanted.get("input_ids_sha256") != identity(tokens)):
            raise legacy.AnalysisError("Original token IDs are missing or inconsistent")
        if wanted.get("case_sha256") != plan["case_sha256"][trial["case_id"]]:
            raise legacy.AnalysisError("Original input record does not bind the case")
    outer_ref = manifest.get("outer_record") or {}
    outer_path = legacy._input_path(root, outer_ref.get("path", ""))
    outer = binding._read_exact_source(outer_path, outer_ref.get("sha256"))
    if (outer.get("run_id") != manifest["run_id"] or outer.get("status") != "completed"
            or outer.get("retry_count") != 0 or type(outer.get("retry_count")) is not int
            or outer.get("attempt_count") != len(trials) or type(outer.get("attempt_count")) is not int
            or outer.get("trial_ids") != list(expected)):
        raise legacy.AnalysisError("Outer completion, attempts, order or retry record differs")
    return {"status": "INPUTS_AND_ATTEMPTS_VERIFIED", "fixture_only": manifest["fixture_only"],
        "original_plan_sha256": expected_plan_sha256, "outer_record_sha256": legacy.sha(outer_path),
        "input_sha256": hashes, "attempt_count": len(trials),
        "runtime_qualification_verified": False, "case_independence_verified": False,
        "research_result_admitted": False,
        "remaining": ["Independently verify applied edits, restoration, weights and runtime records",
                      "Review case-bank provenance, exposure, rights and precision",
                      "Confirm this plan hash was frozen before execution"]}


def require_cache_coverage(observed_fields, recorded_fields):
    """Refuse silent omission of an unfamiliar cache field in a future route."""
    omitted = set(observed_fields) - set(recorded_fields)
    if omitted:
        raise legacy.AnalysisError("Unrecorded cache fields: " + ", ".join(sorted(omitted)))
    return {"coverage_complete": True, "fields": sorted(observed_fields)}
