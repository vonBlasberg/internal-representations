"""Authored software fixtures only. Never use these as an independent case bank."""
from copy import deepcopy
from pathlib import Path
from .reuse import legacy, binding, protocol
from .admission import identity
from .inputs import render_messages
from submission_analysis.fixtures import example_bundle


def comparison_parts():
    manifest, cases, trials = example_bundle()
    spec, digest = protocol()
    manifest.update({"phase": "untouched", "model_id": spec["model_id"], "model_revision": spec["model_revision"],
                     "protocol_sha256": digest, "prompt_text": spec["prompt_text"]})
    for case in cases:
        case.update(partition="untouched", output_contract_version=2)
        case["system_text"], case["user_text"] = legacy.scorer().render_prompt(case)
        case["case_content_sha256"] = legacy.scorer().case_content_sha256(case)
        base = next(r for r in trials if r["case_id"] == case["case_id"] and r["condition"] == "unchanged")
        extra = deepcopy(base)
        extra.update(trial_id=case["case_id"] + '-permuted', condition='permuted', source_sample_id=case["case_id"] + '-permuted')
        trials.append(extra)
        manifest["planned_trials"].append({k: extra[k] for k in ("trial_id", "case_id", "condition")})
    for row in trials:
        row.update({k: manifest[k] for k in ("phase", "model_id", "model_revision")})
        row.update(spec["conditions"][row["condition"]])
    return manifest, cases, trials


def write_fixture(directory, parts=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest, cases, trials = deepcopy(parts or comparison_parts())
    case_map = {c["case_id"]: c for c in cases}
    samples, planned = [], []
    spec, _ = protocol()
    for row in trials:
        case = case_map[row["case_id"]]
        messages = render_messages(case, row["condition"], spec)
        # Deliberately fake token IDs. No research tokenizer or model is loaded.
        ids = [ord(c) + 2 for c in legacy.canonical(messages)]
        inp = {"case_sha256": identity(case), "messages": messages, "input_ids": ids,
               "input_ids_sha256": identity(ids), "tokenizer": "SYNTHETIC_CHARACTER_CODES_NOT_MODEL_TOKENS"}
        metadata = binding.source_metadata(row)
        planned.append({**metadata, "execution_input": inp})
        samples.append({"sample_id": row["source_sample_id"], "input": messages, "response": row["raw_response"],
                        "execution_status": row["execution_status"],
                        "metadata": {"behavioral_trial": metadata, "execution_input": inp}})
    plan = {"artifact_type": "ORIGINAL_INPUT_AND_ATTEMPT_PLAN", "fixture_only": True,
        "run_id": manifest["run_id"], "case_sha256": {c["case_id"]: identity(c) for c in cases},
        "trials": planned, "source_logs": ["invented-source.json"]}
    values = {"original-plan.json": plan, "invented-source.json": {"artifact_type": "INVENTED_EXECUTION_LOG",
        "schema_version": 1, "fixture_only": True, "model_execution": False, "samples": samples},
        "outer.json": {"run_id": manifest["run_id"], "status": "completed", "retry_count": 0,
                       "attempt_count": len(trials), "trial_ids": [r["trial_id"] for r in planned]}}
    for name, value in values.items():
        (directory / name).write_text(legacy.canonical(value) + '\n')
    for row in trials:
        row.update(source_log_path="invented-source.json", source_log_sha256=legacy.sha(directory / "invented-source.json"))
    for name, value in (("cases", cases), ("trials", trials)):
        path = directory / (name + '.json')
        path.write_text(legacy.canonical(value) + '\n')
        manifest["files"][name] = {"path": path.name, "sha256": legacy.sha(path)}
    manifest["outer_record"] = {"path": "outer.json", "sha256": legacy.sha(directory / "outer.json")}
    (directory / 'manifest.json').write_text(legacy.canonical(manifest) + '\n')
    return directory / 'manifest.json'
