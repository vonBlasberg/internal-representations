"""Spreadsheet-readable export of the same verified records shown on screen."""
import csv
import io
import json


COLUMNS = (
    "run_id", "case_id", "condition", "kind", "layer", "strength", "valid",
    "decision", "offer", "buyer_payoff", "supplier_payoff", "proposed_joint_payoff",
    "agreement_adjusted_joint_payoff", "joint_change_from_baseline", "buyer_change_from_baseline",
    "eligible", "validity_check", "joint_improvement_check", "buyer_preservation_check",
    "technical_check", "selected_internal", "selected_prompt", "raw_response",
    "model", "model_revision", "source_log", "source_log_sha256", "evidence_sha256",
)


def spreadsheet_text(value):
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def csv_bytes(data, evidence_sha256):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader()
    compared = {(r["case_id"], r["arm_id"]): r for r in data["comparison"]["rows"]}
    for original in data["rows"]:
        replay = compared[(original["case_id"], original["arm_id"])]
        score, checks, edit = original["score"], replay["criteria"], original.get("edit") or {}
        row = dict(zip(COLUMNS, (
            data["run_id"], original["case_id"], original["arm_id"], original["kind"],
            edit.get("layer"), edit.get("coefficient"), score["valid"], score.get("decision"),
            json.dumps(score.get("package"), sort_keys=True) if score.get("package") is not None else None,
            score.get("proposed_self_payoff"), score.get("proposed_counterpart_payoff"), score.get("proposed_joint_gain"),
            score.get("agreement_adjusted_joint_gain"), checks["joint_delta"], checks["own_delta"],
            checks["eligible"] if original["kind"] in {"internal", "prompt"} else False,
            checks["valid"], checks["joint_improves"], checks["own_preserved"], checks["technical_complete"],
            original["arm_id"] == data["selection"]["selected_internal"],
            original["arm_id"] == data["selection"]["selected_prompt"], original["raw_response"],
            data["model"], data["model_revision"], replay["source_log"], replay["source_log_sha256"], evidence_sha256,
        )))
        writer.writerow({name: spreadsheet_text(value) for name, value in row.items()})
    # BOM makes non-ASCII responses legible when opened directly in spreadsheet applications.
    return stream.getvalue().encode("utf-8-sig")
