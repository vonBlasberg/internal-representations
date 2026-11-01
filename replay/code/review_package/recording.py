"""Prepare Inspect inputs and collect saved logs. No model loading or execution.

The caller must preserve the plan hash before execution. This module does not
approve a study, certify runtime mechanics, or rewrite historical records.
"""
from copy import deepcopy
from pathlib import Path
from .reuse import legacy, binding, protocol
from .admission import identity, verify_inputs_and_attempts
from .inputs import render_messages, validate_messages
from .case_validation import validate_setup, require_prompt_review, make_prompt_review


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(legacy.canonical(value) + "\n")


def prepare_plan(manifest, cases, trials, *, tokenize, tokenizer_record, prompt_review=None, for_review=False):
    """Call an injected tokenizer, never a model; return records for owner review."""
    spec, digest = protocol()
    validate_setup(manifest, cases, spec, digest)
    if not manifest['fixture_only']:
        if for_review:
            if prompt_review != make_prompt_review(cases, spec):
                raise legacy.AnalysisError("Preparation requires the exact pending input review")
        else:
            require_prompt_review(prompt_review, cases, spec)
    if not isinstance(tokenizer_record, dict) or not tokenizer_record:
        raise legacy.AnalysisError("An explicit tokenizer record is required")
    case_map = {c["case_id"]: c for c in cases}
    if len(case_map) != len(cases) or not case_map:
        raise legacy.AnalysisError("Unique cases are required")
    required = {(c, condition) for c in case_map for condition in spec["conditions"]}
    if len(trials) != len(required) or {(r["case_id"], r["condition"]) for r in trials} != required:
        raise legacy.AnalysisError("Incomplete or duplicated condition plan")
    if len({r["trial_id"] for r in trials}) != len(trials):
        raise legacy.AnalysisError("Duplicate trial identity")
    if [{k: r[k] for k in ("trial_id", "case_id", "condition")} for r in trials] != manifest["planned_trials"]:
        raise legacy.AnalysisError("Planned trial order differs")
    planned, paths = [], []
    for number, trial in enumerate(trials):
        for key in ("run_id", "phase", "model_id", "model_revision", "manifest_sha256", "fixture_only"):
            if trial[key] != manifest[key] or type(trial[key]) is not type(manifest[key]):
                raise legacy.AnalysisError("Trial setting differs: " + key)
        for key, expected in spec["conditions"][trial["condition"]].items():
            if trial[key] != expected or type(trial[key]) is not type(expected):
                raise legacy.AnalysisError("Fixed intervention differs: " + key)
        case = case_map[trial["case_id"]]
        messages = render_messages(case, trial["condition"], spec)
        tokens = tokenize(deepcopy(messages))
        if not isinstance(tokens, list) or not tokens or any(type(i) is not int or i < 0 for i in tokens):
            raise legacy.AnalysisError("Tokenizer must return explicit nonnegative integer IDs")
        path = f"source-{number:04d}.json"
        paths.append(path)
        planned.append({**binding.source_metadata(trial), "source_log_path": path,
            "source_sample_id": trial["trial_id"], "execution_input": {
                "case_sha256": identity(case), "messages": messages, "input_ids": tokens,
                "input_ids_sha256": identity(tokens), "tokenizer": deepcopy(tokenizer_record)}})
    result = {"artifact_type": "ORIGINAL_INPUT_AND_ATTEMPT_PLAN", "fixture_only": manifest["fixture_only"],
        "run_id": manifest["run_id"], "case_sha256": {c: identity(v) for c, v in case_map.items()},
        "trials": planned, "source_logs": paths}
    if not manifest['fixture_only']:
        result['prompt_review'] = deepcopy(prompt_review)
        if for_review:
            result['preparation_only'] = True
    return result


def inspect_sample(original, plan_sha256):
    """Attach both required metadata records BEFORE Inspect generation."""
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    messages = validate_messages(original["execution_input"]["messages"])
    types = {"system": ChatMessageSystem, "user": ChatMessageUser}
    return Sample(id=original["source_sample_id"], input=[types[m["role"]](content=m["content"]) for m in messages],
        metadata={"behavioral_trial": binding.source_metadata(original),
                  "execution_input": deepcopy(original["execution_input"]), "original_plan_sha256": plan_sha256})


def collect_saved_inspect(directory, manifest, cases, original_plan_path, expected_plan_sha256, deliveries, outer_record):
    """Copy original serialized logs unchanged and derive rows from them, not a result table."""
    if legacy.sha(Path(original_plan_path)) != expected_plan_sha256:
        raise legacy.AnalysisError("Original plan pin differs")
    plan = legacy.load_json(Path(original_plan_path))
    if set(deliveries) != {r["trial_id"] for r in plan["trials"]}:
        raise legacy.AnalysisError("Missing or extra delivered attempts")
    resolved = [Path(p).resolve() for p in deliveries.values()]
    if len(set(resolved)) != len(resolved):
        raise legacy.AnalysisError("One physical log cannot supply multiple attempts")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # Preserve a copied plan, never replace the separately pinned original.
    with (directory / "original-plan.json").open("xb") as handle:
        handle.write(Path(original_plan_path).read_bytes())
    rows = []
    for original in plan["trials"]:
        source = Path(deliveries[original["trial_id"]])
        source_hash = legacy.sha(source)
        document = binding._read_exact_source(source, source_hash)
        kind, index = binding._source_index(document, manifest["fixture_only"])
        if kind != "inspect_json_v2" or set(index) != {binding._id(original["source_sample_id"])}:
            raise legacy.AnalysisError("One original Inspect sample per planned attempt is required")
        sample = index[binding._id(original["source_sample_id"])]
        binding._settings_match(sample, original)
        if ((sample.get("metadata") or {}).get("original_plan_sha256") != expected_plan_sha256
                or sample["metadata"].get("execution_input") != original["execution_input"]
                or validate_messages(sample.get("input"), inspect_serialized=True) != original["execution_input"]["messages"]):
            raise legacy.AnalysisError("Inspect input or original plan reference differs")
        status, raw, partial = binding._response(document, sample, kind, original)
        if partial:
            raise legacy.AnalysisError("Partial failure output requires separate diagnosis; preserve its source log")
        relative = Path(original["source_log_path"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in (".", ".."):
            raise legacy.AnalysisError("New source logs must use a plain filename inside the delivery directory")
        target = directory / relative
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
        if legacy.sha(target) != source_hash:
            raise legacy.AnalysisError("Source changed during collection")
        rows.append({"schema_version": 1, **binding.source_metadata(original),
                     "execution_status": status, "raw_response": raw,
                     "source_log_path": original["source_log_path"], "source_log_sha256": source_hash,
                     "source_sample_id": original["source_sample_id"]})
    output_manifest = deepcopy(manifest)
    for name, value in (("cases", cases), ("trials", rows), ("outer", outer_record)):
        path = directory / (name + ".json")
        save_new(path, value)
        ref = {"path": path.name, "sha256": legacy.sha(path)}
        if name == "outer":
            output_manifest["outer_record"] = ref
        else:
            output_manifest["files"][name] = ref
    path = directory / "manifest.json"
    save_new(path, output_manifest)
    report = verify_inputs_and_attempts(path, directory / "original-plan.json", expected_plan_sha256)
    save_new(directory / "input-verification.json", report)
    return path
