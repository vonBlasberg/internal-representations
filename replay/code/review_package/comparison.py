"""Prospective ninth-control adapter over the unchanged offline analysis.

This draft accepts saved records only. It never generates, selects or releases
results. Its output cannot certify runtime qualification or case independence.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import jsonschema

from .reuse import legacy, binding, protocol, analysis_provenance
from .history import describe_change
from .scoring import score_completed
from .sensitivity import variant_sensitivity
from .case_validation import validate_setup
from .outcomes import add_agreement_outcomes, agreement_bounds, outcome_contrast

STATUSES = {"COMPLETED", "TECHNICAL_FAILURE", "INDETERMINATE"}


def validate_comparison(manifest, cases, trials, spec, spec_hash, *, allow_partial=False):
    """Validate all nine arms, without changing the original eight-arm validator."""
    validate_setup(manifest, cases, spec, spec_hash)
    if not isinstance(trials, list):
        raise legacy.AnalysisError("A case bank and explicit trial list are required")
    core_contract = legacy.pinned_sources()
    expected_names = set(core_contract["conditions"]) | {"permuted"}
    if set(spec["conditions"]) != expected_names:
        raise legacy.AnalysisError("Expected eight original conditions plus permuted")
    core_manifest = deepcopy(manifest)
    core_manifest["planned_trials"] = [
        row for row in manifest.get("planned_trials", []) if row["condition"] != "permuted"]
    core_trials = [row for row in trials if row["condition"] != "permuted"]
    scoring = legacy.scorer()
    if len(core_trials) == len(cases) * len(core_contract["conditions"]):
        legacy._validate(core_manifest, cases, core_trials, core_contract, scoring)
    elif not allow_partial:
        raise legacy.AnalysisError("Incomplete execution records")
    if not manifest["fixture_only"]:
        # These documents still need independent assessment; hashes alone do not
        # prove non-exposure or scientific validity.
        for key in ("case_bank_freeze_sha256", "prior_exposure_audit_sha256"):
            value = manifest.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise legacy.AnalysisError("Missing prospective evidence reference: " + key)
    required = {(case["case_id"], condition) for case in cases for condition in expected_names}
    planned_rows = manifest.get("planned_trials", [])
    planned = {(row["case_id"], row["condition"]): row["trial_id"] for row in planned_rows}
    if len(planned_rows) != len(required) or set(planned) != required or len(set(planned.values())) != len(required):
        raise legacy.AnalysisError("Incomplete or duplicate nine-condition plan")
    schema = deepcopy(legacy.load_json(legacy.SCIENTIFIC_ROOT / "contracts" / "behavioral_trial.schema.json"))
    schema["properties"]["condition"]["enum"].append("permuted")
    schema["allOf"][2]["if"]["properties"]["condition"]["enum"].append("permuted")
    validator = jsonschema.Draft202012Validator(schema)
    seen, samples = set(), set()
    for trial in trials:
        validator.validate(trial)
        key = trial["case_id"], trial["condition"]
        if key in seen or key not in planned or planned[key] != trial["trial_id"]:
            raise legacy.AnalysisError("Duplicate, missing or unplanned comparison row")
        seen.add(key)
        reference = trial["source_log_path"], binding._id(trial["source_sample_id"])
        if reference in samples:
            raise legacy.AnalysisError("A source sample cannot supply two conditions")
        samples.add(reference)
        for field in ("run_id", "phase", "model_id", "model_revision", "manifest_sha256", "fixture_only"):
            if trial[field] != manifest[field] or type(trial[field]) is not type(manifest[field]):
                raise legacy.AnalysisError("Row contradicts manifest field: " + field)
        expected = spec["conditions"][trial["condition"]]
        for field in ("site", "coefficient", "direction_sha256"):
            value = trial[field]
            if value != expected[field] or (field == "coefficient" and isinstance(value, bool)):
                raise legacy.AnalysisError("Fixed condition differs: " + trial["condition"] + "/" + field)
    if seen != required and not allow_partial:
        raise legacy.AnalysisError("Missing attempts must remain explicit, never dropped")
    return scoring


def bind_all_samples(manifest, trials, root):
    """Reuse Inspect parsing and exact setting checks for every one of nine arms."""
    root = Path(root).resolve()
    seen, documents, records = set(), {}, []
    for trial in trials:
        path = legacy._input_path(root, trial["source_log_path"])
        if path not in documents:
            document = binding._read_exact_source(path, trial["source_log_sha256"])
            if not isinstance(document, dict):
                raise legacy.AnalysisError("Source log must be a JSON object")
            metadata = document.get("metadata") or {}
            if not manifest["fixture_only"] and isinstance(metadata, dict) and metadata.get("fixture_only") is True:
                raise legacy.AnalysisError("An explicit fixture log cannot become real evidence")
            kind, index = binding._source_index(document, manifest["fixture_only"])
            documents[path] = document, kind, index, trial["source_log_sha256"]
        document, kind, index, digest = documents[path]
        if digest != trial["source_log_sha256"]:
            raise legacy.AnalysisError("Conflicting hashes for the same source log")
        sample_key = binding._id(trial["source_sample_id"])
        reference = path, sample_key
        if reference in seen or sample_key not in index:
            raise legacy.AnalysisError("Duplicate or absent physical source sample")
        seen.add(reference)
        sample = index[sample_key]
        binding._settings_match(sample, trial)
        status, response, partial = binding._response(document, sample, kind, trial)
        if status not in STATUSES or status != trial["execution_status"] or response != trial["raw_response"]:
            raise legacy.AnalysisError("Source status or response does not match the comparison")
        records.append({"trial_id": trial["trial_id"], "source_log_sha256": digest,
                        "partial_output_preserved": partial, "source_kind": kind})
    # Reconcile physical samples first. Missing or mistyped self-reported run
    # metadata must never hide an extra attempt in a declared source log.
    logical = set()
    expected = {r["trial_id"] for r in trials}
    for path, (document, kind, index, digest) in documents.items():
        if {(path, key) for key in index} != {ref for ref in seen if ref[0] == path}:
            raise legacy.AnalysisError("Extra, duplicate or unaccounted physical source sample")
        for key, sample in index.items():
            meta = (sample.get("metadata") or {}).get("behavioral_trial") or {}
            if meta.get("run_id") != manifest["run_id"]:
                raise legacy.AnalysisError("Physical source sample has missing or different run metadata")
            tid = meta.get("trial_id")
            if tid not in expected or tid in logical or (path, key) not in seen:
                raise legacy.AnalysisError("Extra, duplicate or unaccounted same-run attempt")
            logical.add(tid)
    return {"status": "EXACT_SAMPLE_MATCH" if records else "NO_DELIVERED_SAMPLES", "records": records,
            "runtime_qualification_verified": False, "case_independence_verified": False}


def analyze_saved(manifest_path, spec_path=None):
    spec, spec_hash = protocol(spec_path)
    manifest, cases, trials, hashes = legacy.read_bundle(Path(manifest_path))
    scoring = validate_comparison(manifest, cases, trials, spec, spec_hash, allow_partial=True)
    source_checks = bind_all_samples(manifest, trials, Path(manifest_path).resolve().parent)
    case_map = {case["case_id"]: case for case in cases}
    conditions = list(spec["conditions"])
    primary = spec['primary_outcome']
    scores, all_rows = {}, []
    for trial in sorted(trials, key=lambda row: (row["case_id"], conditions.index(row["condition"]))):
        case = case_map[trial["case_id"]]
        score = (score_completed(scoring, case, trial["raw_response"])
                 if trial["execution_status"] == "COMPLETED"
                 else scoring.technical_failure_result(case, trial["execution_status"]))
        score = add_agreement_outcomes(score, case)
        scores[(trial["case_id"], trial["condition"])] = score
        all_rows.append({**trial, "family_id": case["family_id"], "variant_kind": case["variant_kind"],
                         "record_origin": "SOURCE_SAMPLE", "score": score})
    delivered = {row["trial_id"] for row in trials}
    missing = [row for row in manifest["planned_trials"] if row["trial_id"] not in delivered]
    if missing:
        # These rows are an inventory of missing delivery, not fabricated Inspect
        # samples or proof that the worker did not start a trial.
        for planned in missing:
            case = case_map[planned["case_id"]]
            all_rows.append({**planned, "family_id": case["family_id"], "phase": "untouched",
                "fixture_only": manifest["fixture_only"], "raw_response": None,
                "execution_status": "INDETERMINATE", "record_origin": "PLAN_ONLY_NO_DELIVERED_SAMPLE",
                "score": add_agreement_outcomes(scoring.technical_failure_result(case, "INDETERMINATE"), case),
                "site": spec["conditions"][planned["condition"]]["site"],
                "coefficient": spec["conditions"][planned["condition"]]["coefficient"],
                "direction_sha256": spec["conditions"][planned["condition"]]["direction_sha256"]})
        return {"artifact_type": "INCOMPLETE_DELIVERY_INVENTORY", "schema_version": 1,
            "run_id": manifest["run_id"], "phase": "untouched", "fixture_only": manifest["fixture_only"],
            "evidence_label": "SYNTHETIC_TEST_ONLY" if manifest["fixture_only"] else "INCOMPLETE_NOT_ANALYZED",
            "planned_attempts": len(manifest["planned_trials"]), "delivered_attempts": len(trials),
            "missing_attempts": len(missing), "all_trials": all_rows, "contrasts": [],
            "source_checks": source_checks, "input_sha256": hashes, "protocol_sha256": spec_hash,
            "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
            "analysis_provenance": analysis_provenance(),
            "selection_performed": False, "model_execution": False, "provider_calls": 0,
            "qualification_verified": False, "release_approved": False,
            "population_inference": False, "p_values": False,
            "interpretation": "Missing records are unresolved delivery, not evidence of zero effect or that a trial never started"}
    proposed_ranges = {case["case_id"]: legacy._bounds(case, scoring) for case in cases}
    adjusted_ranges = {case["case_id"]: agreement_bounds(case, scoring) for case in cases}
    ranges = adjusted_ranges if primary == 'agreement_adjusted_joint_gain' else proposed_ranges
    comparisons = [("internal", "unchanged", "primary"), ("internal", "prompt", "key_secondary")]
    comparisons += [(condition, "unchanged", "control") for condition in conditions
                    if condition not in {"unchanged", "internal", "prompt"}]
    contrasts = [{**outcome_contrast(left, right, primary, cases, scores, ranges),
                  "role": role} for left, right, role in comparisons]
    outcomes = [k for k in legacy.pinned_sources()['secondary_outcomes'] if k != primary]
    if primary == 'agreement_adjusted_joint_gain':
        outcomes += ['proposed_joint_gain', 'agreement_adjusted_self_payoff', 'agreement_adjusted_counterpart_payoff']
    secondary = [outcome_contrast("internal", comparator, outcome, cases, scores,
                                 adjusted_ranges if outcome == 'agreement_adjusted_joint_gain' else proposed_ranges)
                 for comparator in ("unchanged", "prompt")
                 for outcome in outcomes]
    for contrast in contrasts + secondary:
        deltas = [family["delta"] for family in contrast["families"]]
        no_variation = bool(deltas) and all(delta is not None for delta in deltas) and len(set(deltas)) == 1
        contrast["no_observed_family_variation"] = no_variation
        if no_variation:
            contrast["interval_note"] += " No variation across families in this case bank; not population certainty."
    accounting, descriptions = [], []
    for condition in conditions:
        selected = [row["score"] for row in all_rows if row["condition"] == condition]
        counts = Counter(row["execution_status"] for row in selected)
        accounting.append({"condition": condition, "planned": len(selected),
            "completed": counts["COMPLETED"], "technical_failure": counts["TECHNICAL_FAILURE"],
            "indeterminate": counts["INDETERMINATE"], "valid": sum(row["valid"] for row in selected),
            "invalid": sum(row["execution_status"] == "COMPLETED" and not row["valid"] for row in selected),
            "offers": sum(row["valid"] and row["decision"] == "offer" for row in selected),
            "accepted_offers": sum(row["valid"] and bool(row["accepted_offer"]) for row in selected),
            "accepted_counterpart_packages": sum(row["valid"] and bool(row["accepted_counterpart_package"]) for row in selected),
            "simulated_agreements": sum(row["valid"] and row["agreement"] for row in selected),
            "buyer_below_reservation": sum(row['buyer_below_reservation'] is True for row in selected),
            "simulated_agreement_rate_among_valid": legacy.mean([int(row["agreement"]) for row in selected if row["valid"]])})
        for case in cases:
            cid = case["case_id"]
            descriptions.append({"case_id": cid, "family_id": case["family_id"], "condition": condition,
                "payoff_basis": "proposed terms before simulated acceptance",
                **describe_change(scores[(cid, condition)], scores[(cid, "unchanged")])})
        comparable = [row for row in descriptions if row["condition"] == condition and row["status"] == "VALID"]
        accounting[-1].update({"valid_pairs_with_baseline": len(comparable),
            "decision_changes": sum(row["decision_changed"] for row in comparable),
            "package_changes": sum(row["package_changed"] for row in comparable),
            "decision_change_rate_among_valid_pairs": legacy.mean([int(row["decision_changed"]) for row in comparable]),
            "package_change_rate_among_valid_pairs": legacy.mean([int(row["package_changed"]) for row in comparable]),
            "rate_note": "Descriptive case counts among stated valid denominators, not family-weighted effect estimates"})
    headroom = []
    for case in cases:
        cid = case["case_id"]
        baseline = scores[(cid, "unchanged")]
        headroom.append({"case_id": cid, "maximum_joint_payoff": ranges[cid][1],
            "outcome": primary,
            "baseline_joint_payoff": baseline[primary],
            "joint_headroom": ranges[cid][1] - baseline[primary] if baseline["valid"] else None,
            "excluded": False})
    return {"artifact_type": "REVIEW_DRAFT_NINE_CONDITION_ANALYSIS", "schema_version": 1,
        "run_id": manifest["run_id"], "phase": "untouched", "fixture_only": manifest["fixture_only"],
        "evidence_label": "SYNTHETIC_TEST_ONLY" if manifest["fixture_only"] else "SAVED_RESULTS_PENDING_INDEPENDENT_ADMISSION",
        "case_count": len(cases), "family_count": len({case["family_id"] for case in cases}),
        "primary_outcome": primary,
        "accounting": accounting, "contrasts": contrasts, "secondary_contrasts": secondary,
        "variant_sensitivity": variant_sensitivity(cases, scores, conditions, outcome=primary),
        "case_descriptions": descriptions, "headroom": headroom, "all_trials": all_rows,
        "input_sha256": hashes, "protocol_sha256": spec_hash, "source_checks": source_checks,
        "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
        "analysis_provenance": analysis_provenance(),
        "selection_performed": False, "model_execution": False, "provider_calls": 0,
        "release_approved": False, "population_inference": False, "p_values": False,
        "qualification_verified": False, "case_independence_verified": False,
        "interpretation": "A saved comparison, not a certification of a construct or robust findings"}

