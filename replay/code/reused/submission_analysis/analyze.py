"""Implement the unchanged October 29 analysis contract on supplied saved outputs.

This module does not open a study case bank by default, import a model library,
connect to a provider, grant qualification, or approve a release.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCIENTIFIC_ROOT = ROOT / "negotiation-steering"
PINS = {
    "study/manifests/behavioral_analysis_contract.json": "9702efbcce9643569ad72dae43660e06ca5139d3d07da17b83770d11c9315196",
    "contracts/behavioral_trial.schema.json": "624028e45d84cf7cbf16e60b4208f5c6f46768188317188e3463ed43732562e9",
    "analysis/negotiation.py": "977a02c34c2ad4eee86381f18c4e551d40cad749d68b4122751a3aebc5ff4f51",
    "analysis/parse.py": "088a3c759efd6b6fb72237c913846a3d25022234fdb8548a51b6a69f34b7f237",
}
LABELS = {
    "unchanged": "Baseline", "prompt": "Prompting", "internal": "Internal edit",
    "sham": "Sham edit", "random_1": "Random control 1", "random_2": "Random control 2",
    "random_3": "Random control 3", "reverse": "Reversed direction",
}


class AnalysisError(ValueError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise AnalysisError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise AnalysisError(f"JSON input exceeds 32 MiB: {path.name}")
    def invalid_constant(value):
        raise AnalysisError(f"Non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pinned_sources():
    for relative, digest in PINS.items():
        if sha(SCIENTIFIC_ROOT / relative) != digest:
            raise AnalysisError(f"Pinned scientific source changed: {relative}")
    return load_json(SCIENTIFIC_ROOT / "study/manifests/behavioral_analysis_contract.json")


def scorer():
    pinned_sources()
    sys.path.insert(0, str(SCIENTIFIC_ROOT))
    from analysis import negotiation
    if Path(negotiation.__file__).resolve() != SCIENTIFIC_ROOT / "analysis/negotiation.py":
        raise AnalysisError("Unexpected scorer import")
    return negotiation


def mean(values):
    return math.fsum(values) / len(values) if values else None


def family_bootstrap(values):
    """Paired family resampling, not token, variant, or generation resampling."""
    if len(values) < 2:
        return None
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(20261029))
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    samples = values[indices].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975], method="linear").tolist()


def _input_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AnalysisError("Input paths must be relative to the supplied bundle")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise AnalysisError("Input path leaves the supplied bundle or is not a file")
    return path


def read_bundle(manifest_path: Path):
    """Hash exact inputs and referenced log bytes; never infer a successful run."""
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("artifact_type") != "SAVED_BEHAVIORAL_ANALYSIS_INPUT":
        raise AnalysisError("Not a saved behavioral analysis input manifest")
    if manifest.get("schema_version") != 1:
        raise AnalysisError("Unsupported input manifest version")
    root = manifest_path.parent
    contents, hashes = {}, {manifest_path.name: sha(manifest_path)}
    for name in ("cases", "trials"):
        ref = manifest["files"][name]
        path = _input_path(root, ref["path"])
        digest = sha(path)
        if digest != ref["sha256"]:
            raise AnalysisError(f"Input hash mismatch: {name}")
        contents[name] = load_json(path)
        hashes[ref["path"]] = digest
    for trial in contents["trials"]:
        path = _input_path(root, trial["source_log_path"])
        digest = sha(path)
        if digest != trial["source_log_sha256"]:
            raise AnalysisError(f"Source log hash mismatch: {trial['trial_id']}")
        hashes[trial["source_log_path"]] = digest
    return manifest, contents["cases"], contents["trials"], hashes


def _validate(manifest, cases, trials, contract, scoring):
    if manifest.get("analysis_contract_sha256") != PINS["study/manifests/behavioral_analysis_contract.json"]:
        raise AnalysisError("Analysis contract is not the locked October 29 contract")
    phase = manifest.get("phase")
    if phase not in {"development", "untouched"}:
        raise AnalysisError("This analyzer accepts development or untouched outputs separately; calibration needs its own grid specification")
    if type(manifest.get("fixture_only")) is not bool:
        raise AnalysisError("Fixture status must be explicit")
    if not isinstance(cases, list) or not cases or not isinstance(trials, list) or not trials:
        raise AnalysisError("Cases and trials must be nonempty lists")
    scoring.validate_family_partitions(cases)
    if any(case["partition"] != phase for case in cases):
        raise AnalysisError("Case partitions do not match the single declared analysis phase")
    conditions = contract["required_conditions_by_phase"][phase]
    required = {(case["case_id"], cond) for case in cases for cond in conditions}
    planned = manifest.get("planned_trials", [])
    if len(planned) != len(required):
        raise AnalysisError("Manifest does not plan every case and required condition exactly once")
    planned_pairs = {(row["case_id"], row["condition"]): row["trial_id"] for row in planned}
    if set(planned_pairs) != required or len(set(planned_pairs.values())) != len(required):
        raise AnalysisError("Duplicate, missing, or extra planned trial")
    schema = load_json(SCIENTIFIC_ROOT / "contracts/behavioral_trial.schema.json")
    validator = jsonschema.Draft202012Validator(schema)
    seen = set()
    sample_refs = set()
    condition_settings = {}
    for trial in trials:
        try:
            validator.validate(trial)
        except jsonschema.ValidationError as exc:
            raise AnalysisError(f"Invalid trial exchange record: {exc.message}") from exc
        pair = (trial["case_id"], trial["condition"])
        if pair in seen or pair not in planned_pairs or planned_pairs[pair] != trial["trial_id"]:
            raise AnalysisError("Duplicate, unplanned, or substituted trial")
        seen.add(pair)
        source_ref = (trial["source_log_path"], str(trial["source_sample_id"]))
        if source_ref in sample_refs:
            raise AnalysisError("One source sample cannot supply multiple trials")
        sample_refs.add(source_ref)
        for field in ("run_id", "phase", "model_id", "model_revision", "manifest_sha256", "fixture_only"):
            if trial[field] != manifest.get(field):
                raise AnalysisError(f"Trials mix or contradict manifest field: {field}")
        coefficient = trial["coefficient"]
        if coefficient is not None and (isinstance(coefficient, bool) or not math.isfinite(coefficient)):
            raise AnalysisError("Edit coefficient must be finite")
        if trial["condition"] == "sham" and coefficient != 0:
            raise AnalysisError("Sham edit coefficient must be zero")
        settings = (trial["site"], coefficient, trial["direction_sha256"])
        previous = condition_settings.setdefault(trial["condition"], settings)
        if settings != previous:
            raise AnalysisError("A condition mixes sites, coefficients, or directions across cases")
        if trial["condition"] in {"unchanged", "prompt"} and settings != (None, None, None):
            raise AnalysisError("Baseline and prompt-only trials must not declare an internal edit")
    if seen != required:
        raise AnalysisError("Missing execution records; retain an explicit INDETERMINATE record rather than dropping planned trials")


def _bounds(case, scoring):
    values = [sum(scoring.payoffs(case, package)) for package in scoring.enumerate_packages(case)]
    return [min(values), max(values)]


def _contrast(left, right, outcome, cases, scores, ranges):
    by_family = defaultdict(list)
    for case in cases:
        by_family[case["family_id"]].append(case)
    families, completed_pairs, valid_pairs = [], 0, 0
    all_bounds = []
    for family_id, family_cases in sorted(by_family.items()):
        valid, completed, bounds, invalid_ids, technical_ids = [], [], [], [], []
        for case in sorted(family_cases, key=lambda item: item["case_id"]):
            cid = case["case_id"]
            a, b = scores[(cid, left)], scores[(cid, right)]
            both_completed = a["execution_status"] == b["execution_status"] == "COMPLETED"
            if not both_completed:
                technical_ids.append(cid)
                continue
            completed.append(cid)
            if a["valid"] and b["valid"] and a[outcome] is not None and b[outcome] is not None:
                valid.append((cid, float(a[outcome]), float(b[outcome])))
            else:
                invalid_ids.append(cid)
            if outcome == "proposed_joint_gain":
                lower, upper = ranges[cid]
                alo, ahi = (a[outcome], a[outcome]) if a["valid"] else (lower, upper)
                blo, bhi = (b[outcome], b[outcome]) if b["valid"] else (lower, upper)
                bounds.append((alo - bhi, ahi - blo))
        completed_pairs += len(completed)
        valid_pairs += len(valid)
        full = len(valid) == len(family_cases)
        low = mean([b[0] for b in bounds])
        high = mean([b[1] for b in bounds])
        if bounds:
            all_bounds.append((low, high))
        left_mean = mean([row[1] for row in valid])
        right_mean = mean([row[2] for row in valid])
        delta = mean([row[1] - row[2] for row in valid])
        families.append({"family_id": family_id, "planned_pairs": len(family_cases),
            "completed_pairs": len(completed), "valid_pairs": len(valid),
            "case_ids": sorted(c["case_id"] for c in family_cases),
            "valid_case_ids": [row[0] for row in valid],
            "invalid_or_undefined_case_ids": invalid_ids, "technical_missing_case_ids": technical_ids,
            "left_mean": left_mean if full else None, "right_mean": right_mean if full else None,
            "delta": delta if full else None,
            "valid_pair_subset_delta": delta,
            "completed_pair_subset_bounds": [low, high] if bounds else None})
    complete = valid_pairs == len(cases)
    deltas = [row["delta"] for row in families] if complete else []
    full_bounds = all_bounds and completed_pairs == len(cases)
    return {"left": left, "right": right, "outcome": outcome,
        "planned_pairs": len(cases), "completed_pairs": completed_pairs, "valid_pairs": valid_pairs,
        "family_count": len(families), "families": families,
        "point_estimate": mean(deltas),
        "point_estimate_status": "ALL_PLANNED_PAIRS_VALID" if complete else "FULL_CASE_BANK_POINT_NOT_IDENTIFIED",
        "case_bank_sensitivity_interval": family_bootstrap(deltas) if complete else None,
        "interval_note": ("Paired-family case-bank sensitivity, not population inference." if complete and len(families) >= 2
                          else "No interval: fewer than two families." if complete
                          else "No full-case-bank interval: an outcome is missing, invalid, or undefined."),
        "full_case_bank_invalid_response_bounds": [mean([v[0] for v in all_bounds]), mean([v[1] for v in all_bounds])] if full_bounds else None,
        "completed_pair_subset_bounds": [mean([v[0] for v in all_bounds]), mean([v[1] for v in all_bounds])] if all_bounds else None,
        "valid_pair_subset_description": mean([row["valid_pair_subset_delta"] for row in families if row["valid_pair_subset_delta"] is not None]),
        "subset_warning": "Subset descriptions and bounds do not replace the fixed-bank estimate when planned pairs are absent."}


def analyze(manifest, cases, trials, input_hashes=None, *, source_binding=None):
    contract = pinned_sources()
    scoring = scorer()
    _validate(manifest, cases, trials, contract, scoring)
    if source_binding is not None:
        from .source_binding import require_matching_binding
        require_matching_binding(source_binding, manifest, cases, trials)
    cases = sorted(cases, key=lambda case: (case["family_id"], case["case_id"]))
    case_map = {case["case_id"]: case for case in cases}
    scores, rows = {}, []
    for trial in sorted(trials, key=lambda row: (row["case_id"], contract["conditions"].index(row["condition"]))):
        case = case_map[trial["case_id"]]
        result = (scoring.score_response(case, trial["raw_response"]) if trial["execution_status"] == "COMPLETED"
                  else scoring.technical_failure_result(case, trial["execution_status"]))
        scores[(case["case_id"], trial["condition"])] = result
        rows.append({**trial, "family_id": case["family_id"], "variant_kind": case["variant_kind"],
                     "case_content_sha256": case["case_content_sha256"], "score": result})
    accounting = []
    for condition in contract["conditions"]:
        selected = [row["score"] for row in rows if row["condition"] == condition]
        completed = [row for row in selected if row["execution_status"] == "COMPLETED"]
        valid = [row for row in completed if row["valid"]]
        accounting.append({"condition": condition, "planned_attempts": len(selected), "completed": len(completed),
            "technical_failures": sum(r["execution_status"] == "TECHNICAL_FAILURE" for r in selected),
            "indeterminate": sum(r["execution_status"] == "INDETERMINATE" for r in selected),
            "valid": len(valid), "invalid": len(completed) - len(valid),
            "offers": sum(r["decision"] == "offer" for r in valid),
            "accepted_offers": sum(bool(r["accepted_offer"]) for r in valid),
            "accepted_counterpart_packages": sum(bool(r["accepted_counterpart_package"]) for r in valid),
            "completion_rate": len(completed) / len(selected),
            "validity_among_completed": len(valid) / len(completed) if completed else None,
            "agreement_among_valid": sum(bool(r["agreement"]) for r in valid) / len(valid) if valid else None})
    ranges = {case["case_id"]: _bounds(case, scoring) for case in cases}
    comparisons = [("internal", "unchanged", "primary"), ("internal", "prompt", "key_secondary")]
    comparisons += [(cond, "unchanged", "exploratory") for cond in contract["conditions"] if cond not in {"unchanged", "internal"}]
    contrasts = [{**_contrast(a, b, "proposed_joint_gain", cases, scores, ranges), "role": role}
                 for a, b, role in comparisons]
    secondary = [_contrast("internal", "unchanged", outcome, cases, scores, ranges)
                 for outcome in contract["secondary_outcomes"]]
    decomposition = []
    for cond in contract["conditions"]:
        family_values = defaultdict(list)
        for case in cases:
            result = scores[(case["case_id"], cond)]
            if result["execution_status"] == "COMPLETED" and result["valid"]:
                family_values[case["family_id"]].append(result)
        nvalid = sum(len(values) for values in family_values.values())
        item = {"condition": cond, "valid_cases": nvalid, "planned_cases": len(cases),
                "full_case_bank": nvalid == len(cases)}
        for outcome in ("proposed_self_payoff", "proposed_counterpart_payoff", "proposed_joint_gain"):
            value = mean([mean([row[outcome] for row in values]) for values in family_values.values()])
            item[outcome] = value if item["full_case_bank"] else None
            item[outcome + "_valid_subset"] = value
        decomposition.append(item)
    interpretations = []
    eligible = []
    for case in cases:
        cid = case["case_id"]
        a, b = scores[(cid, "internal")], scores[(cid, "unchanged")]
        if not (a["execution_status"] == b["execution_status"] == "COMPLETED" and a["valid"] and b["valid"]):
            continue
        eligible.append(case)
        own = a["proposed_self_payoff"] - b["proposed_self_payoff"]
        other = a["proposed_counterpart_payoff"] - b["proposed_counterpart_payoff"]
        joint = own + other
        interpretations.append({"case_id": cid, "own_delta": own, "counterpart_delta": other,
            "joint_delta": joint, "mutual_benefit_seeking": joint > 0 and own == 0,
            "yielding": own < 0 and joint == 0,
            "own_surplus_prioritization": own > 0 and other < 0})
    eligible.sort(key=lambda case: (case["variant_kind"] != "base", case["family_id"], case["case_id"]))
    return {"artifact_type": "SAVED_BEHAVIORAL_ANALYSIS", "schema_version": 1,
        "run_id": manifest["run_id"], "phase": manifest["phase"], "fixture_only": manifest["fixture_only"],
        "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
        "release_approved": False, "model_execution": False, "provider_calls": 0,
        "evidence_status": "SYNTHETIC_SOFTWARE_TEST_ONLY" if manifest["fixture_only"] else "SAVED_OUTPUT_ANALYSIS_PENDING_EVIDENCE_AND_RELEASE_REVIEW",
        "claim_label_if_separately_qualified": contract["claim_labels"][manifest["phase"]],
        "provenance": {"input_file_sha256": input_hashes or {}, "scientific_source_sha256": PINS,
            "analysis_implementation_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "analysis_environment": {"python": sys.version, "numpy": np.__version__, "jsonschema": importlib.metadata.version("jsonschema")},
            "source_log_bytes_verified": bool(input_hashes), "source_log_to_response_binding_verified": source_binding is not None,
            "source_binding": source_binding,
            "qualification_verified": False, "untouched_exposure_authorization_verified": False,
            "note": ("Source samples match the supplied responses, execution status, and trial settings. Runtime qualification and release permission require separate evidence."
                if source_binding is not None else "Checksums and rescoring do not establish runtime qualification, source-sample binding, or release permission.")},
        "uncertainty": {**contract["uncertainty"], "rng": "numpy.PCG64", "quantile_method": "linear", "numpy_version": np.__version__},
        "population_inference": False, "p_values": False,
        "case_count": len(cases), "family_count": len({c["family_id"] for c in cases}),
        "accounting": accounting, "contrasts": contrasts, "secondary_outcomes": secondary,
        "payoff_decomposition": decomposition, "case_interpretations": interpretations,
        "featured_case_id": eligible[0]["case_id"] if eligible else None,
        "featured_case_rule": contract["featured_case_rule"], "scored_trials": rows,
        "missing_for_release": ["Independent evidence review" if source_binding is not None else "Source-log sample binding and independent evidence review",
            "Runtime and study qualification, complete comparison and off-target evidence",
            "Separate untouched authorization and freeze evidence when applicable", "Owner release approval"]}
