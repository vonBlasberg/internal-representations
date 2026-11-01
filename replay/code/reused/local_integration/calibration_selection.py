"""Fixed, exact-payoff calibration rule. No outcome-dependent grid extension."""
from fractions import Fraction

from .preparatory_phases import checked_scores, identity

RULE = {
    "name": "joint-gain-with-own-payoff-preserved-v1",
    "eligibility": "Every response valid; joint gain strictly exceeds baseline and own payoff is no lower on every calibration case.",
    "internal_order": ["descending_equal_family_mean_joint_gain", "ascending_absolute_strength", "ascending_layer"],
    "prompt_order": ["descending_equal_family_mean_joint_gain", "declared_prompt_order"],
    "controls": "Retain all controls; no control or prompt outcome is a candidate-selection threshold.",
    "no_eligible_internal": "Stop with no selected internal candidate; do not extend the grid.",
    "inference_scope": "Calibration cases only; not held-out evidence or population inference.",
}

def select(plan, cases, result):
    if plan["phase"] != "calibration" or plan["selection_rule"] != RULE:
        raise ValueError("Selection requires the exact prospective calibration rule.")
    scores = checked_scores(plan, cases, result)
    by_pair = {(r["case_id"], r["arm_id"]): scores[r["trial_id"]] for r in result["records"]}
    base = next(a["arm_id"] for a in plan["arms"] if a["kind"] == "unchanged")
    eligible, rejected = [], []
    for index, arm in enumerate(plan["arms"]):
        if arm["kind"] not in {"prompt", "internal"}:
            continue
        values, reasons = {}, []
        for case in cases:
            score, baseline = by_pair[(case["case_id"], arm["arm_id"])], by_pair[(case["case_id"], base)]
            if not score or not baseline or not score["valid"] or not baseline["valid"]:
                reasons.append(case["case_id"] + ": incomplete or invalid")
                continue
            if score["proposed_joint_gain"] <= baseline["proposed_joint_gain"]:
                reasons.append(case["case_id"] + ": no strict joint-gain improvement")
            if score["proposed_self_payoff"] < baseline["proposed_self_payoff"]:
                reasons.append(case["case_id"] + ": own payoff lower")
            values.setdefault(case["family_id"], []).append(score["proposed_joint_gain"])
        if reasons or result["failure"] is not None:
            rejected.append({"arm_id": arm["arm_id"], "reasons": reasons or ["technical/stopping failure"]})
        else:
            average = sum((Fraction(sum(v), len(v)) for v in values.values()), Fraction()) / len(values)
            eligible.append((arm, average, index))
    internal = sorted((v for v in eligible if v[0]["kind"] == "internal"),
        key=lambda v: (-v[1], abs(v[0]["edit"]["coefficient"]), v[0]["edit"]["layer"]))
    prompts = sorted((v for v in eligible if v[0]["kind"] == "prompt"), key=lambda v: (-v[1], v[2]))
    return {"status": "SELECTED_FOR_LATER_EVALUATION" if internal else "NO_ELIGIBLE_INTERNAL_STOP",
        "plan_sha256": identity(plan), "rule": RULE,
        "selected_internal": internal[0][0]["arm_id"] if internal else None,
        "selected_prompt": prompts[0][0]["arm_id"] if prompts else None,
        "eligible": [v[0]["arm_id"] for v in eligible], "rejected": rejected,
        "held_out_result": False, "further_execution_authorized": False}
