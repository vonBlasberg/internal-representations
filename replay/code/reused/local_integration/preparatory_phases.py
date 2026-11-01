"""Plan and record construction/calibration using explicit, separately approved inputs.

No model loader, provider, default study grid, or scientific case-file access is
included. The caller supplies cases and a qualified generation connection.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from submission_analysis.analyze import canonical, mean, pinned_sources, scorer, sha

PHASES = {"construction_fit", "construction_check", "calibration"}
POLICIES = {"mutual_benefit", "own_surplus"}
CALIBRATION_KINDS = {"unchanged", "prompt", "internal", "sham", "random_1", "random_2", "random_3", "reverse", "permuted"}
UNEDITED = {"unchanged", "prompt", *POLICIES}


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
        raise ValueError("Use an explicit safe identifier.")
    return value


def digest_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_arm(arm, phase):
    safe_id(arm["arm_id"])
    kind = arm["kind"]
    if kind not in (CALIBRATION_KINDS if phase == "calibration" else POLICIES):
        raise ValueError("Arm does not belong to the declared phase.")
    suffix = arm["prompt_suffix"]
    if not isinstance(suffix, str):
        raise ValueError("Prompt additions must be explicit text, including an empty string.")
    if kind in {"prompt", *POLICIES} and not suffix.strip():
        raise ValueError("Declare the policy or comparator instruction before execution.")
    if kind not in {"prompt", *POLICIES} and suffix:
        raise ValueError("Internal/control edits must retain the baseline instructions.")
    edit = arm["edit"]
    if kind in UNEDITED:
        if edit is not None:
            raise ValueError("A baseline, prompt or construction policy cannot contain an internal edit.")
    else:
        if not isinstance(edit, dict) or set(edit) != {"layer", "module_path", "coefficient", "direction_sha256"}:
            raise ValueError("Declare the exact edit, including its stored direction hash.")
        if type(edit["layer"]) is not int or edit["layer"] < 0 or not isinstance(edit["module_path"], str) or not edit["module_path"]:
            raise ValueError("The exact edited layer and module path are required.")
        coefficient = edit["coefficient"]
        if type(coefficient) not in (int, float) or not math.isfinite(coefficient) or not digest_string(edit["direction_sha256"]):
            raise ValueError("Invalid coefficient or direction identity.")
        if kind == "sham" and coefficient != 0:
            raise ValueError("A sham must have zero displacement.")
        if kind != "sham" and coefficient == 0:
            raise ValueError("Zero strength belongs to the sham, not a nonzero candidate.")


def prepare_plan(*, run_id, phase, cases, arms, model_id, model_revision,
                 generation_contract, family_partitions, maximum_wall_seconds,
                 fixture_only, qualification_receipt_sha256=None, construction_check_rule=None,
                 selection_rule=None, stop_on_invalid=False):
    """Freeze supplied choices, without choosing layers, strengths or prompts."""
    pinned_sources()
    scoring = scorer()
    safe_id(run_id)
    if phase not in PHASES or type(fixture_only) is not bool:
        raise ValueError("Declare one preparatory phase and its fixture status.")
    if not isinstance(model_id, str) or not model_id or not isinstance(model_revision, str) or len(model_revision) != 40 or any(c not in "0123456789abcdef" for c in model_revision):
        raise ValueError("An exact model identity and revision are required.")
    if type(maximum_wall_seconds) not in (int, float) or not math.isfinite(maximum_wall_seconds) or maximum_wall_seconds <= 0:
        raise ValueError("Declare a positive wall-clock bound.")
    if not cases or not arms:
        raise ValueError("Explicit nonempty cases and arms are required.")
    if any(generation_contract.get(key) != value for key, value in
           (("model_id", model_id), ("model_revision", model_revision), ("synthetic_test", fixture_only), ("max_generation_calls", 1))):
        raise ValueError("Generation contract must bind this model and one attempt per trial.")
    if phase == "construction_check":
        keys = {"minimum_valid_pair_fraction", "minimum_joint_gap", "minimum_own_gap"}
        if not isinstance(construction_check_rule, dict) or set(construction_check_rule) != keys:
            raise ValueError("Freeze construction-check thresholds before generating responses.")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in construction_check_rule.values()) or not 0 <= construction_check_rule["minimum_valid_pair_fraction"] <= 1 or construction_check_rule["minimum_joint_gap"] < 0 or construction_check_rule["minimum_own_gap"] < 0:
            raise ValueError("Invalid construction-check thresholds.")
    elif construction_check_rule is not None:
        raise ValueError("Construction-check thresholds belong to the check phase only.")
    if type(stop_on_invalid) is not bool:
        raise ValueError("Invalid-response stopping must be explicit.")
    if selection_rule is not None:
        from .calibration_selection import RULE
        if phase != "calibration" or selection_rule != RULE:
            raise ValueError("Unsupported calibration rule.")
    scoring.validate_family_partitions(cases)
    if any(c["partition"] != phase or family_partitions.get(c["family_id"]) != phase for c in cases):
        raise ValueError("Cases and their family allocation must belong to this phase.")
    if len({a["arm_id"] for a in arms}) != len(arms):
        raise ValueError("Arm IDs must be unique.")
    for arm in arms:
        validate_arm(arm, phase)
    kinds = [a["kind"] for a in arms]
    if phase != "calibration" and (len(arms) != 2 or set(kinds) != POLICIES):
        raise ValueError("Construction requires exactly the two declared response policies.")
    if phase == "calibration" and (kinds.count("unchanged") != 1 or not {"prompt", "internal"}.issubset(kinds)):
        raise ValueError("Calibration requires one shared baseline and explicit internal and prompt candidates.")
    if qualification_receipt_sha256 is not None and not digest_string(qualification_receipt_sha256):
        raise ValueError("Malformed qualification receipt identity.")
    trials = []
    for case in cases:
        base = [{"role": "system", "content": case["system_text"]}, {"role": "user", "content": case["user_text"]}]
        for arm in arms:
            messages = deepcopy(base)
            if arm["prompt_suffix"]:
                messages[0]["content"] += "\n" + arm["prompt_suffix"]
            trial_id = safe_id(case["case_id"] + "--" + arm["arm_id"])
            trials.append({"trial_id": trial_id, "case_id": case["case_id"], "family_id": case["family_id"],
                "arm_id": arm["arm_id"], "kind": arm["kind"], "messages": messages,
                "messages_sha256": identity(messages), "edit": deepcopy(arm["edit"]),
                "case_content_sha256": case["case_content_sha256"],
                "output_contract_version": case.get("output_contract_version", 1)})
    return {"artifact_type": "PREPARATORY_PHASE_PLAN", "schema_version": 1, "run_id": run_id,
        "phase": phase, "fixture_only": fixture_only, "model_id": model_id, "model_revision": model_revision,
        "generation_contract": deepcopy(generation_contract), "family_partitions": deepcopy(family_partitions),
        "cases_sha256": identity(cases), "arms": deepcopy(arms), "trials": trials,
        "maximum_generation_calls": len(trials), "maximum_wall_seconds": maximum_wall_seconds,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "construction_check_rule": deepcopy(construction_check_rule),
        "implementation_sha256": sha(Path(__file__)), "automatic_retry": False,
        "selection_rule": deepcopy(selection_rule), "selection_authorized": selection_rule is not None,
        "stop_on_invalid": stop_on_invalid, "deployment_authorized": False}


def validate_plan(plan, cases):
    if plan.get("artifact_type") != "PREPARATORY_PHASE_PLAN" or plan.get("schema_version") != 1:
        raise ValueError("Not a preparatory-phase plan.")
    rebuilt = prepare_plan(**{key: plan[key] for key in ("run_id", "phase", "arms", "model_id", "model_revision",
        "generation_contract", "family_partitions", "maximum_wall_seconds", "fixture_only", "qualification_receipt_sha256", "construction_check_rule", "selection_rule", "stop_on_invalid")}, cases=cases)
    if canonical(plan) != canonical(rebuilt):
        raise ValueError("Cases, settings, planned trials or source differ from the prepared plan.")


def validate_authority(plan, approval):
    if approval.get("owner_authorized") is not True:
        raise PermissionError("Explicit owner approval is required.")
    required = {"owner_authorized": True, "plan_sha256": identity(plan), "phase": plan["phase"],
        "maximum_generation_calls": plan["maximum_generation_calls"], "automatic_retry": False,
        "fixture_only": plan["fixture_only"], "deployment_authorized": False}
    if any(approval.get(key) != value for key, value in required.items()):
        raise PermissionError("This exact phase and call allowance need their own approval.")
    if not plan["fixture_only"] and (not digest_string(plan["qualification_receipt_sha256"]) or
        approval.get("paid_phase_authorized") is not True or not approval.get("owner_authorization_reference")):
        raise PermissionError("Real execution needs a qualified route and explicit paid-phase approval.")
    if not plan["fixture_only"] and plan["phase"] == "calibration" and plan["selection_rule"] is None:
        raise PermissionError("The real calibration selection rule remains to be specified and frozen before execution.")


def execute_phase(plan, cases, approval, runs_root, *, generate):
    """Execute a fixed order once. A provider supervisor must enforce the wall cap.

    `generate` is an admitted callable, not a provider name or dynamic import.
    The supplied Inspect connection writes its original sample and state record.
    """
    validate_plan(plan, cases)
    validate_authority(plan, approval)
    scoring = scorer()
    directory = Path(runs_root) / plan["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    save_new(directory / "started.json", {"plan": plan, "approval": approval})
    by_id = {c["case_id"]: c for c in cases}
    records = []
    started = time.monotonic()
    failure = None
    for index, trial in enumerate(plan["trials"]):
        row = {"trial_id": trial["trial_id"], "case_id": trial["case_id"], "family_id": trial["family_id"],
            "arm_id": trial["arm_id"], "kind": trial["kind"], "phase": plan["phase"],
            "plan_sha256": identity(plan), "execution_status": "NOT_RUN_AFTER_FAILURE", "raw_response": None}
        if failure is None:
            location = directory / trial["trial_id"]
            location.mkdir(exist_ok=False)
            save_new(location / "trial.json", trial)
            try:
                if time.monotonic() - started >= plan["maximum_wall_seconds"]:
                    raise TimeoutError("Phase wall allowance exhausted; no next call.")
                output = generate(trial, location)
                if not isinstance(output, dict) or not isinstance(output.get("raw_response"), str):
                    raise ValueError("A completed generation must return exact text.")
                row.update(execution_status="COMPLETED", raw_response=output["raw_response"], source=output)
                # Preserve the returned text before scoring or dependent decisions.
                save_new(location / "generation.json", row)
                row["score"] = scoring.score_response(by_id[trial["case_id"]], row["raw_response"])
                if not row["score"]["valid"] and (plan["stop_on_invalid"] or trial["kind"] == "unchanged"):
                    raise ValueError("Invalid response at a declared stopping gate; preserve it and do not continue.")
                if time.monotonic() - started > plan["maximum_wall_seconds"]:
                    raise TimeoutError("Phase wall allowance exceeded after preserving the completed response.")
            except Exception as error:
                failure = {"trial_id": trial["trial_id"], "error_type": type(error).__name__, "error": str(error)}
                if row["execution_status"] != "COMPLETED":
                    row["execution_status"] = "TECHNICAL_FAILURE"
                    row["score"] = scoring.technical_failure_result(by_id[trial["case_id"]])
                save_new(location / "failure.json", failure)
        records.append(row)
        save_new(directory / f"record-{index:05d}.json", row)
    result = {"artifact_type": "PREPARATORY_PHASE_RESULT", "schema_version": 1,
        "run_id": plan["run_id"], "phase": plan["phase"], "plan_sha256": identity(plan),
        "status": "FAILED_NO_RETRY" if failure else "COMPLETE_NOT_SCIENTIFICALLY_QUALIFIED",
        "fixture_only": plan["fixture_only"], "records": records, "failure": failure,
        "elapsed_seconds": time.monotonic() - started, "automatic_retry": False,
        "behavioral_qualification": False, "selection_authorized": False, "release_approved": False}
    save_new(directory / "result.json", result)
    return result


def checked_scores(plan, cases, result):
    validate_plan(plan, cases)
    if result.get("plan_sha256") != identity(plan) or result.get("phase") != plan["phase"] or result.get("fixture_only") != plan["fixture_only"]:
        raise ValueError("Result belongs to another phase or plan.")
    records = result["records"]
    if [row["trial_id"] for row in records] != [t["trial_id"] for t in plan["trials"]]:
        raise ValueError("Every planned attempt must be present exactly once in order.")
    scoring = scorer()
    by_id = {c["case_id"]: c for c in cases}
    scores = {}
    for trial, row in zip(plan["trials"], records):
        if any(row.get(k) != trial[k] for k in ("trial_id", "case_id", "family_id", "arm_id", "kind")):
            raise ValueError("Result row has substituted trial settings.")
        status = row["execution_status"]
        if status not in {"COMPLETED", "TECHNICAL_FAILURE", "INDETERMINATE", "NOT_RUN_AFTER_FAILURE"}:
            raise ValueError("Unknown execution status.")
        if status != "COMPLETED" and row["raw_response"] is not None:
            raise ValueError("A failure must not supply a behavioral response.")
        scores[trial["trial_id"]] = scoring.score_response(by_id[trial["case_id"]], row["raw_response"]) if status == "COMPLETED" else None
    return scores


def _family_mean(rows):
    grouped = defaultdict(list)
    for family, value in rows:
        grouped[family].append(value)
    return mean([mean(values) for values in grouped.values()])


def check_construction_labels(plan, cases, result):
    """Apply caller-declared thresholds to separate construction-check families."""
    if plan["phase"] != "construction_check":
        raise ValueError("Construction labels must be checked on construction-check cases, not fit cases.")
    scores = checked_scores(plan, cases, result)
    rule = plan["construction_check_rule"]
    minimum_valid_pair_fraction, minimum_joint_gap, minimum_own_gap = (rule[k] for k in ("minimum_valid_pair_fraction", "minimum_joint_gap", "minimum_own_gap"))
    joined = {(row["case_id"], row["kind"]): scores[row["trial_id"]] for row in result["records"]}
    joint, own = [], []
    for case in cases:
        positive, negative = (joined[(case["case_id"], label)] for label in ("mutual_benefit", "own_surplus"))
        if positive and negative and positive["valid"] and negative["valid"]:
            joint.append((case["family_id"], positive["proposed_joint_gain"] - negative["proposed_joint_gain"]))
            own.append((case["family_id"], negative["proposed_self_payoff"] - positive["proposed_self_payoff"]))
    valid_fraction = len(joint) / len(cases)
    joint_gap, own_gap = _family_mean(joint), _family_mean(own)
    all_families_present = {family for family, _ in joint} == {c["family_id"] for c in cases}
    passed = result["failure"] is None and all_families_present and valid_fraction >= minimum_valid_pair_fraction and joint_gap is not None and own_gap is not None and joint_gap > minimum_joint_gap and own_gap > minimum_own_gap
    return {"status": "PASS" if passed else "FAIL", "phase": "construction_check", "plan_sha256": identity(plan),
        "valid_pairs": len(joint), "planned_pairs": len(cases), "valid_pair_fraction": valid_fraction,
        "joint_gap_mutual_minus_own": joint_gap, "self_gap_own_minus_mutual": own_gap,
        "all_families_have_valid_pairs": all_families_present,
        "thresholds": {"minimum_valid_pair_fraction": minimum_valid_pair_fraction, "minimum_joint_gap": minimum_joint_gap, "minimum_own_gap": minimum_own_gap},
        "means_describe": "Valid paired responses only; all planned pairs remain in the validity denominator.",
        "steering_evidence": False, "release_approved": False}


def summarize_calibration(plan, cases, result):
    """Show every declared arm without selecting a candidate or dropping failures."""
    if plan["phase"] != "calibration":
        raise ValueError("Only calibration belongs in this grid summary.")
    scores = checked_scores(plan, cases, result)
    baseline = next(a["arm_id"] for a in plan["arms"] if a["kind"] == "unchanged")
    by_pair = {(r["case_id"], r["arm_id"]): r for r in result["records"]}
    rows = []
    for arm in plan["arms"]:
        valid, pairs, completed = [], [], 0
        counts = defaultdict(int)
        for case in cases:
            record = by_pair[(case["case_id"], arm["arm_id"])]
            counts[record["execution_status"]] += 1
            score = scores[record["trial_id"]]
            if record["execution_status"] == "COMPLETED":
                completed += 1
            if score and score["valid"]:
                valid.append((case["family_id"], score))
                base_score = scores[by_pair[(case["case_id"], baseline)]["trial_id"]]
                if base_score and base_score["valid"]:
                    pairs.append((case["family_id"], score["proposed_joint_gain"] - base_score["proposed_joint_gain"]))
        rows.append({"arm_id": arm["arm_id"], "kind": arm["kind"], "edit": arm["edit"],
            "planned": len(cases), "completed": completed, "valid": len(valid), "invalid": completed - len(valid),
            "execution_counts": dict(counts), "valid_pair_count": len(pairs),
            "full_bank_joint_difference": _family_mean(pairs) if len(pairs) == len(cases) else None,
            "valid_pair_joint_difference_description": _family_mean(pairs),
            "valid_only_means": {key: _family_mean([(family, score[key]) for family, score in valid])
                for key in ("proposed_joint_gain", "proposed_self_payoff", "proposed_counterpart_payoff")}})
    return {"phase": "calibration", "plan_sha256": identity(plan), "fixture_only": plan["fixture_only"],
        "rows": rows, "selected_arm": None, "selection_requires_predeclared_rule": True,
        "development_or_untouched_evidence": False, "release_approved": False}
