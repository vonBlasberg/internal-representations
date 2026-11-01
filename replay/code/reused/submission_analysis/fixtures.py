"""Invented analysis tests. No scientific material or model output is loaded."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from local_integration.fixtures import authored_case, BASE, EDITED, PROMPTED, REJECTED
from .analyze import PINS, canonical, pinned_sources, scorer, sha


def example_bundle():
    scoring = scorer()
    conditions = pinned_sources()["conditions"]
    cases = []
    for family, variants in (("fixture-a", ("base", "paraphrase")), ("fixture-b", ("base",))):
        for variant in variants:
            case = authored_case()
            case.update({"case_id": f"{family}-{variant}", "family_id": family,
                "variant_id": variant, "variant_of": f"{family}-base", "variant_kind": variant,
                "prompt_style": variant, "partition": "development"})
            case["system_text"], case["user_text"] = scoring.render_prompt(case)
            case["case_content_sha256"] = scoring.case_content_sha256(case)
            cases.append(case)
    manifest_hash = hashlib.sha256(b"invented-execution-plan-only").hexdigest()
    manifest = {"artifact_type": "SAVED_BEHAVIORAL_ANALYSIS_INPUT", "schema_version": 1,
        "run_id": "analysis-software-test-only", "phase": "development", "fixture_only": True,
        "model_id": "no-model-invented-answers", "model_revision": "0" * 40,
        "manifest_sha256": manifest_hash,
        "analysis_contract_sha256": PINS["study/manifests/behavioral_analysis_contract.json"],
        "planned_trials": [], "files": {}}
    trials = []
    for case in cases:
        for condition in conditions:
            trial_id = f"{case['case_id']}-{condition}"
            package = BASE
            if condition == "internal":
                package = EDITED if case["family_id"] == "fixture-a" else REJECTED
            elif condition == "prompt":
                package = PROMPTED
            elif condition == "reverse":
                package = REJECTED
            elif condition == "random_2":
                package = EDITED
            edited = condition not in {"unchanged", "prompt"}
            trials.append({"schema_version": 1, "trial_id": trial_id, "case_id": case["case_id"],
                "condition": condition, "execution_status": "COMPLETED",
                "raw_response": canonical({"decision": "offer", "package": deepcopy(package)}),
                "source_log_path": "invented-source.json", "source_log_sha256": "0" * 64,
                "source_sample_id": trial_id, "site": "fixture-only-no-model" if edited else None,
                "coefficient": (0 if condition == "sham" else -1 if condition == "reverse" else 1) if edited else None,
                "direction_sha256": "0" * 64 if edited else None,
                **{key: manifest[key] for key in ("run_id", "phase", "fixture_only", "model_id", "model_revision", "manifest_sha256")}})
            manifest["planned_trials"].append({"trial_id": trial_id, "case_id": case["case_id"], "condition": condition})
    return manifest, cases, trials


def write_fixture(directory: Path):
    """Write new, unmistakably synthetic inputs. Refuse to overwrite a directory."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest, cases, trials = example_bundle()
    from .source_binding import source_metadata
    source = {"artifact_type": "INVENTED_EXECUTION_LOG", "schema_version": 1,
        "fixture_only": True, "model_execution": False,
        "samples": [{"sample_id": row["source_sample_id"], "response": row["raw_response"],
            "execution_status": row["execution_status"], "metadata": {"behavioral_trial": source_metadata(row)}} for row in trials]}
    (directory / "invented-source.json").write_text(canonical(source) + "\n")
    digest = sha(directory / "invented-source.json")
    for row in trials:
        row["source_log_sha256"] = digest
    for name, value in (("cases", cases), ("trials", trials)):
        path = directory / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        manifest["files"][name] = {"path": path.name, "sha256": sha(path)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory / "manifest.json"
