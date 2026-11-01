"""Prepare a display of saved A10 evidence, without selecting another candidate."""
from .reuse import PACKAGE, legacy, analysis_provenance
from .scoring import score_completed

A10_SHA256 = "e05e035e3163b5419eb7978eb1f759a351b9197eeeb77ea7fb90ba9adec5b63b"
HISTORY_HASHES = {
    "a05": "7202393343420a44e923bf4f23f48cf0a1a4e2c9ceee3865ff0642c90d152cb8",
    "a06": "815153b224ccb5699937c82713acb4798220b23caea8badd477a5889b2863055",
    "a09": "c6c09da4ed8c84a73424b393c1117f10ca04ecf06043a170ddef1fbe867aad28",
    "a10": A10_SHA256,
}


def describe_change(score, baseline):
    if score["execution_status"] != "COMPLETED":
        return {"status": score["execution_status"], "decision_changed": None,
                "package_changed": None, "buyer_delta": None, "supplier_delta": None,
                "joint_delta": None, "neither_party_loses": None,
                "buyer_gains_supplier_loses": None, "both_parties_lose": None}
    if not score["valid"] or not baseline["valid"]:
        return {"status": "INVALID_OR_UNDEFINED", "decision_changed": None,
                "package_changed": None, "buyer_delta": None, "supplier_delta": None,
                "joint_delta": None, "neither_party_loses": None,
                "buyer_gains_supplier_loses": None, "both_parties_lose": None}
    buyer = score["proposed_self_payoff"] - baseline["proposed_self_payoff"]
    supplier = score["proposed_counterpart_payoff"] - baseline["proposed_counterpart_payoff"]
    return {"status": "VALID", "decision_changed": score["decision"] != baseline["decision"],
            "package_changed": score["package"] != baseline["package"],
            "buyer_delta": buyer, "supplier_delta": supplier, "joint_delta": buyer + supplier,
            "neither_party_loses": buyer >= 0 and supplier >= 0,
            "buyer_gains_supplier_loses": buyer > 0 and supplier < 0,
            "both_parties_lose": buyer < 0 and supplier < 0}


def selection_display(publication):
    """Display the recorded rule and outcome. Never invoke selection on evaluation."""
    rows = publication["rows"]
    if any(row["phase"] != "calibration" for row in rows):
        raise legacy.AnalysisError("Selection display accepts historical calibration only")
    baseline_rows = [row for row in rows if row["kind"] == "unchanged"]
    if len(baseline_rows) != 1:
        raise legacy.AnalysisError("This historical display expects one calibration case")
    baseline = baseline_rows[0]["score"]
    selection = publication["selection"]
    reasons = {row["arm_id"]: row["reasons"] for row in selection.get("rejected", [])}
    eligible = set(selection.get("eligible", []))
    predicates_by_arm = {}
    if len({row['arm_id'] for row in rows}) != len(rows):
        raise legacy.AnalysisError('Duplicate historical arm identity')
    for row in rows:
        if row['kind'] in {'internal', 'prompt'}:
            change = describe_change(row['score'], baseline)
            valid = change['status'] == 'VALID'
            predicates_by_arm[row['arm_id']] = {
                'valid_response': valid,
                'strict_joint_gain': change['joint_delta'] > 0 if valid else None,
                'buyer_payoff_preserved': change['buyer_delta'] >= 0 if valid else None}
    computed = {arm for arm, checks in predicates_by_arm.items() if all(v is True for v in checks.values())}
    if eligible != computed or len(selection.get('eligible', [])) != len(eligible):
        raise legacy.AnalysisError('Saved eligibility disagrees with independently checked historical predicates')
    ranked = {}
    for kind in ("internal", "prompt"):
        candidates = [(index, r) for index, r in enumerate(rows) if r["kind"] == kind and r["arm_id"] in eligible]
        def order(value):
            index, row = value
            edit = row.get("edit") or {}
            return (-row["score"]["proposed_joint_gain"], abs(edit["coefficient"]), edit["layer"]) if kind == "internal" else (-row["score"]["proposed_joint_gain"], index)
        ordered = sorted(candidates, key=order)
        # Check consistency with the saved selection, without replacing it.
        saved = selection.get("selected_" + kind)
        if (ordered[0][1]["arm_id"] if ordered else None) != saved:
            raise legacy.AnalysisError("Saved selection and declared historical ranking disagree")
        for rank, (_, row) in enumerate(ordered, 1):
            ranked[row["arm_id"]] = rank
    output = []
    for row in rows:
        is_candidate = row["kind"] in {"internal", "prompt"}
        change = describe_change(row["score"], baseline)
        predicates = predicates_by_arm.get(row['arm_id'])
        output.append({
            "trial_id": row["trial_id"], "case_id": row["case_id"], "family_id": row["family_id"],
            "phase": row["phase"], "execution_status": row["execution_status"], "fixture_only": False,
            "arm_id": row["arm_id"], "kind": row["kind"], "edit": row["edit"],
            "candidate_for_selection": is_candidate,
            "historical_rule_qualified": row["arm_id"] in computed if is_candidate else None,
            "historically_selected": row["arm_id"] in {
                selection.get("selected_internal"), selection.get("selected_prompt")},
            "selection_reasons": (reasons.get(row["arm_id"]) or
                (["Valid response; combined payoff increases; buyer payoff does not fall.",
                  "Rank " + str(ranked[row["arm_id"]]) + " among eligible " + row["kind"] + " candidates."]
                 if row["arm_id"] in ranked else ["Comparison condition, not a selection candidate."])),
            "eligibility_checks": predicates, "historical_rank": ranked.get(row["arm_id"]),
            "ranking_rule": selection["rule"]["internal_order" if row["kind"] == "internal" else "prompt_order"] if is_candidate else None,
            "selection_scope": "This calibration only, not robustness or construct validity",
            "decision_and_payoffs": change,
            "score": row["score"], "raw_response": row["raw_response"], "source": row["source"],
        })
    return output


def read_calibration(name="a10", path=None):
    if name not in HISTORY_HASHES:
        raise legacy.AnalysisError("No admitted publication for that attempt")
    path = path or PACKAGE / "evidence" / (name + "-publication.json")
    expected_hash = HISTORY_HASHES[name]
    if legacy.sha(path) != expected_hash:
        raise legacy.AnalysisError("Publication differs from the admitted saved copy")
    saved = legacy.load_json(path)
    if saved["generation_count"] != 41 or len(saved["rows"]) != 41:
        raise legacy.AnalysisError("Incomplete historical display")
    scoring = legacy.scorer()
    case = saved["case"]
    seen = set()
    for row in saved["rows"]:
        if row["trial_id"] in seen or row["case_id"] != case["case_id"]:
            raise legacy.AnalysisError("Duplicate or substituted historical row")
        seen.add(row["trial_id"])
        score = (score_completed(scoring, case, row["raw_response"])
                 if row["execution_status"] == "COMPLETED"
                 else scoring.technical_failure_result(case, row["execution_status"]))
        if legacy.canonical(score) != legacy.canonical(row["score"]):
            raise legacy.AnalysisError("Saved response and payoff score disagree")
    display_rows = selection_display(saved)
    geometry = None
    if name == 'a10':
        from .geometry import attach_a10_geometry
        geometry = attach_a10_geometry(display_rows)
    return {
        "artifact_type": "HISTORICAL_SELECTION_REVIEW",
        "attempt": name, "publication_sha256": expected_hash,
        "model_id": saved["model"], "model_revision": saved["model_revision"],
        "selection": saved["selection"], "rows": display_rows, "post_hoc_geometry": geometry,
        "case": case, "attempts": len(saved["rows"]),
        "valid": sum(row["score"]["valid"] for row in saved["rows"]),
        "scope": "One-case exploratory calibration",
        "full_mechanical_audit_replayed_here": False,
        "source_log_samples_replayed_here": False,
        "saved_response_scores_replayed_here": True,
        "analysis_provenance": analysis_provenance(),
        "model_execution": False, "release_approved": False,
    }


def read_a10(path=None):
    return read_calibration("a10", path)

