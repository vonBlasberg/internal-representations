"""Descriptive variant comparisons. Does not replace the frozen family estimand."""
from collections import defaultdict
from .reuse import legacy


def variant_sensitivity(cases, scores, conditions, *, outcome='proposed_joint_gain'):
    result = []
    prefix = 'agreement_adjusted_' if outcome == 'agreement_adjusted_joint_gain' else 'proposed_'
    for left in conditions:
        if left == "unchanged":
            continue
        for right in ("unchanged", "prompt") if left == "internal" else ("unchanged",):
            families = defaultdict(list)
            for case in cases:
                a, b = scores[(case["case_id"], left)], scores[(case["case_id"], right)]
                comparable = a["valid"] and b["valid"]
                families[case["family_id"]].append({
                    "case_id": case["case_id"], "variant": case["variant_kind"],
                    "left_status": a["execution_status"], "right_status": b["execution_status"],
                    "left_valid": a["valid"], "right_valid": b["valid"],
                    "joint_delta": a[outcome] - b[outcome] if comparable else None,
                    "buyer_delta": a[prefix + 'self_payoff'] - b[prefix + 'self_payoff'] if comparable else None,
                    "supplier_delta": a[prefix + 'counterpart_payoff'] - b[prefix + 'counterpart_payoff'] if comparable else None})
            for family, rows in sorted(families.items()):
                base = [r for r in rows if r["variant"] == "base"]
                base = base[0] if len(base) == 1 else None
                values = [r["joint_delta"] for r in rows if r["joint_delta"] is not None]
                complete = len(values) == len(rows)
                for row in rows:
                    row["delta_minus_base_variant"] = (row["joint_delta"] - base["joint_delta"]
                        if base and row["joint_delta"] is not None and base["joint_delta"] is not None else None)
                result.append({"left": left, "right": right, "family_id": family, "variants": rows, "outcome": outcome,
                    "all_variants_comparable": complete,
                    "observed_sign_reversal": bool(values and min(values) < 0 < max(values)),
                    "validity_changes_across_variants": len({(r["left_valid"], r["right_valid"]) for r in rows}) > 1,
                    "delta_range": [min(values), max(values)] if values else None,
                    "scope": "Descriptive observed variants; no robustness pass or population claim"})
    return result
