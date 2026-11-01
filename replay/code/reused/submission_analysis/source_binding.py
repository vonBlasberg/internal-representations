"""Bind analysis rows to exact saved log samples. Never execute a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .analyze import (AnalysisError, _input_path, _unique_object, _validate,
    canonical, load_json, pinned_sources, read_bundle, scorer)

METADATA_FIELDS = ("run_id", "phase", "trial_id", "case_id", "condition", "model_id",
    "model_revision", "manifest_sha256", "site", "coefficient", "direction_sha256", "fixture_only")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def bundle_identity(manifest, cases, trials):
    return digest(canonical({"manifest": manifest, "cases": cases, "trials": trials}).encode("utf-8"))


def source_metadata(trial):
    """Fixed settings for Inspect Sample.metadata['behavioral_trial'].

    These fields are recorded before generation. Execution status and response
    are read from the eventual source sample, not asserted in this metadata.
    """
    return {key: trial[key] for key in METADATA_FIELDS}


def _read_exact_source(path, expected_hash):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise AnalysisError("Source log exceeds 32 MiB; split logs before execution.")
    data = path.read_bytes()
    if len(data) > 32 * 1024 * 1024 or digest(data) != expected_hash:
        raise AnalysisError("Source log bytes differ from their declared hash.")
    def invalid(value):
        raise AnalysisError("Non-finite source JSON number: " + value)
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=invalid)


def _id(value):
    if type(value) not in (str, int) or value == "":
        raise AnalysisError("Source sample IDs must be nonempty strings or integers.")
    return type(value).__name__, value


def _source_index(document, fixture_only):
    if not isinstance(document, dict):
        raise AnalysisError("Source log must be a JSON object.")
    if document.get("artifact_type") == "INVENTED_EXECUTION_LOG":
        if not fixture_only or document.get("fixture_only") is not True or document.get("model_execution") is not False or document.get("schema_version") != 1:
            raise AnalysisError("Invented source records cannot verify real model results.")
        kind, field = "invented_fixture", "sample_id"
    elif document.get("version") == 2 and isinstance(document.get("eval"), dict):
        if document.get("invalidated") is not False or document.get("status") not in {"started", "success", "cancelled", "error"}:
            raise AnalysisError("Inspect source is invalidated or has an unknown status.")
        packages = document["eval"].get("packages")
        if not isinstance(packages, dict) or packages.get("inspect_ai") != "0.3.286":
            raise AnalysisError("Source adapter supports the pinned Inspect AI version only.")
        kind, field = "inspect_json_v2", "id"
    else:
        raise AnalysisError("Unsupported source format; no source binding may be inferred.")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise AnalysisError("Source log has no explicit sample records.")
    index = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise AnalysisError("Source sample must be a JSON object.")
        key = _id(sample.get(field))
        if key in index or (kind == "inspect_json_v2" and sample.get("epoch") != 1):
            raise AnalysisError("Duplicate or repeated-epoch source samples are ambiguous.")
        index[key] = sample
    return kind, index


def _text_output(output):
    choices = output.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise AnalysisError("A completed sample must contain exactly one response choice.")
    if not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict):
        raise AnalysisError("Source response choice lacks an assistant message.")
    message = choices[0]["message"]
    if message.get("role") != "assistant" or message.get("tool_calls"):
        raise AnalysisError("Only the declared text response is supported, not tool calls.")
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str) for part in content):
        # Inspect ChatMessage.text joins text parts with newline. Do not strip,
        # canonicalize JSON, or discard reasoning/multimodal content silently.
        text = "\n".join(part["text"] for part in content)
    else:
        raise AnalysisError("Unsupported source content; retain the source and review its adapter.")
    if output.get("completion") != text:
        raise AnalysisError("Inspect completion and its assistant message disagree.")
    return text


def _settings_match(sample, trial):
    metadata = sample.get("metadata")
    declared = metadata.get("behavioral_trial") if isinstance(metadata, dict) else None
    if not isinstance(declared, dict):
        raise AnalysisError("Source sample lacks the recorded behavioral-trial settings.")
    for key in METADATA_FIELDS:
        if key not in declared:
            raise AnalysisError("Source sample lacks setting: " + key)
        value, expected = declared[key], trial[key]
        if key == "coefficient" and value is not None:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise AnalysisError("Invalid source coefficient.")
        elif type(value) is not type(expected):
            raise AnalysisError("Source setting type differs: " + key)
        if value != expected:
            raise AnalysisError("Source setting differs: " + key)


def _response(document, sample, kind, trial):
    if kind == "invented_fixture":
        return sample["execution_status"], sample["response"], False
    permitted_names = {trial["model_id"], "steerability/" + trial["model_id"]}
    if document["eval"].get("model") not in permitted_names:
        raise AnalysisError("Inspect model identity differs from the analysis row.")
    output = sample.get("output") or {}
    if not isinstance(output, dict):
        raise AnalysisError("Malformed Inspect output.")
    if output.get("model") and output["model"] not in permitted_names:
        raise AnalysisError("Inspect output model differs from the declared model.")
    if sample.get("error") is not None:
        return "TECHNICAL_FAILURE", None, bool(output.get("choices") or output.get("completion"))
    if output.get("choices"):
        return "COMPLETED", _text_output(output), False
    if output.get("completion"):
        raise AnalysisError("Inspect has completion text without a response choice.")
    if document["status"] != "success":
        return "INDETERMINATE", None, False
    raise AnalysisError("Successful Inspect log has a sample without an output or failure.")


def verify_source_bindings(manifest, cases, trials, root):
    """Check all declared trials, not just successful or plotted examples."""
    _validate(manifest, cases, trials, pinned_sources(), scorer())
    root = Path(root).resolve()
    sources, used, records = {}, set(), []
    for trial in trials:
        path = _input_path(root, trial["source_log_path"])
        resolved = str(path)
        expected_hash = trial["source_log_sha256"]
        if resolved not in sources:
            document = _read_exact_source(path, expected_hash)
            kind, index = _source_index(document, manifest["fixture_only"])
            sources[resolved] = document, kind, index, expected_hash
        document, kind, index, actual_hash = sources[resolved]
        if actual_hash != expected_hash:
            raise AnalysisError("Rows declare different hashes for the same source file.")
        sample_key = _id(trial["source_sample_id"])
        reference = resolved, sample_key
        if reference in used:
            raise AnalysisError("One physical source sample cannot supply multiple trials.")
        used.add(reference)
        if sample_key not in index:
            raise AnalysisError("Declared source sample is missing: " + str(trial["source_sample_id"]))
        sample = index[sample_key]
        _settings_match(sample, trial)
        status, response, partial = _response(document, sample, kind, trial)
        if status != trial["execution_status"] or response != trial["raw_response"]:
            raise AnalysisError("Analysis status or raw response differs from the original source sample: " + trial["trial_id"])
        if status not in {"COMPLETED", "TECHNICAL_FAILURE", "INDETERMINATE"}:
            raise AnalysisError("Unknown source execution status.")
        if (status == "COMPLETED" and not isinstance(response, str)) or (status != "COMPLETED" and response is not None):
            raise AnalysisError("Source response conflicts with its execution status.")
        records.append({"trial_id": trial["trial_id"], "source_log_path": trial["source_log_path"],
            "source_log_sha256": actual_hash, "source_sample_id": trial["source_sample_id"],
            "source_kind": kind, "execution_status": status,
            "response_sha256": digest(response.encode("utf-8")) if response is not None else None,
            "partial_output_preserved_in_source": partial, "settings_match": True})
    return {"artifact_type": "SOURCE_SAMPLE_BINDING_CHECK", "schema_version": 1, "status": "PASS",
        "run_id": manifest["run_id"], "phase": manifest["phase"], "fixture_only": manifest["fixture_only"],
        "analysis_input_sha256": bundle_identity(manifest, cases, trials),
        "trial_count": len(trials), "source_file_count": len(sources), "records": records,
        "model_execution": False, "qualification_verified": False, "release_approved": False,
        "meaning": "Exact source-sample, response, execution-status and declared-setting agreement. Not independent proof of model execution, applied edits, or release permission."}


def require_matching_binding(binding, manifest, cases, trials):
    if binding.get("artifact_type") != "SOURCE_SAMPLE_BINDING_CHECK" or binding.get("status") != "PASS" or binding.get("analysis_input_sha256") != bundle_identity(manifest, cases, trials) or binding.get("trial_count") != len(trials):
        raise AnalysisError("Source binding does not cover these exact analysis inputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real-saved-outputs", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the existing audit; choose a new output path.")
    try:
        header = load_json(args.manifest)
        if header.get("fixture_only") is not True and not args.allow_real_saved_outputs:
            raise AnalysisError("Real source auditing requires --allow-real-saved-outputs before opening source logs.")
        manifest, cases, trials, _ = read_bundle(args.manifest)
        if manifest.get("fixture_only") is not True and not args.allow_real_saved_outputs:
            raise AnalysisError("Real source auditing was not authorized.")
        result = verify_source_bindings(manifest, cases, trials, args.manifest.resolve().parent)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": result["status"], "trial_count": result["trial_count"], "model_execution": False}))
    except (AnalysisError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, "Source audit refused: " + str(error) + "\n")


if __name__ == "__main__":
    main()
