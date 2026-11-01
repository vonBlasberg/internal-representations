"""Offline review entry point. No launch or generation actions exist."""
import argparse
import csv
from pathlib import Path

from .reuse import legacy
from .history import read_calibration
from .comparison import analyze_saved
from .presentation import page


def write_new(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # No overwrite or publication action. The whole result includes every row.
    with (directory / "review.json").open("x", encoding="utf-8") as handle:
        handle.write(legacy.canonical(result) + "\n")
    with (directory / "index.html").open("x", encoding="utf-8") as handle:
        handle.write(page(result))
    if result["artifact_type"] == "INPUT_ADMISSION_REVIEW":
        return
    rows = result.get("all_trials", result.get("rows", []))
    if result["artifact_type"] == "REUSABLE_SYNTHETIC_WORKFLOW_CHECK":
        columns = ["study_name", "fixture_only", "evidence_label", "empirical_evidence", "model_execution",
                   "case_id", "condition", "execution_status", "valid", *result["outcome_names"], "raw_response", "source", "csv_escaped_fields"]
        with (directory / "all_attempts.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                values = {k: row.get(k) for k in columns}
                values.update(study_name=result["name"], fixture_only=result["fixture_only"],
                              evidence_label=result["evidence_label"], empirical_evidence=result["empirical_evidence"],
                              model_execution=result["model_execution"])
                values.update(row["outcomes"])
                values["source"] = legacy.canonical(row.get("source"))
                escaped = []
                for key, value in values.items():
                    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                        values[key] = "'" + value
                        escaped.append(key)
                values["csv_escaped_fields"] = ','.join(escaped)
                writer.writerow(values)
        return
    columns = ["trial_id", "arm_id", "condition", "case_id", "family_id", "phase", "fixture_only",
               "evidence_label", "record_origin", "model_id", "model_revision", "site", "coefficient", "direction_sha256",
               "source_log_path", "source_log_sha256", "source_sample_id", "historically_selected",
               "publication_sha256", "protocol_sha256", "historical_rule_qualified",
               "selection_status", "selection_rule_json", "selection_scope", "selection_reasons", "eligibility_checks", "historical_rank", "ranking_rule", "execution_status",
               "valid", "decision", "buyer_payoff", "supplier_payoff", "joint_payoff",
               "agreement_adjusted_joint_payoff", "direction_norm", "cosine_with_intended",
               "intended_displacement_l2", "intended_projection_on_target",
               "realized_displacement_l2_min", "realized_displacement_l2_max",
               "realized_displacement_observations", "realized_displacement_source",
               "realized_displacement_source_sha256", "geometry_status",
               "raw_response", "csv_escaped_fields"]
    prospective = result['artifact_type'] != 'HISTORICAL_SELECTION_REVIEW'
    if prospective:
        columns += ['primary_outcome', 'agreement_adjusted_buyer_payoff',
                    'agreement_adjusted_supplier_payoff', 'buyer_below_reservation']
    with (directory / "all_attempts.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            score = row["score"]
            source, edit = row.get("source") or {}, row.get("edit") or {}
            exported = {**{key: row.get(key) for key in columns},
                "condition": row.get("condition", row.get("kind")),
                "phase": row.get("phase", result.get("phase")),
                "fixture_only": row.get("fixture_only", result.get("fixture_only", False)),
                "evidence_label": result.get("evidence_label", result.get("scope")),
                "record_origin": row.get("record_origin", "SAVED_PUBLICATION"),
                "model_id": row.get("model_id", result.get("model_id")),
                "model_revision": row.get("model_revision", result.get("model_revision")),
                "site": row.get("site", edit.get("module_path")),
                "coefficient": row.get("coefficient", edit.get("coefficient")),
                "direction_sha256": row.get("direction_sha256", edit.get("direction_sha256")),
                "source_log_path": row.get("source_log_path", source.get("source_log_path")),
                "source_log_sha256": row.get("source_log_sha256", source.get("source_log_sha256")),
                "source_sample_id": row.get("source_sample_id", source.get("source_sample_id")),
                "publication_sha256": result.get("publication_sha256"),
                "protocol_sha256": result.get("protocol_sha256"),
                "selection_status": result.get("selection", {}).get("status", "NOT_PERFORMED_ON_EVALUATION"),
                "selection_rule_json": legacy.canonical(result.get("selection", {}).get("rule")),
                "selection_scope": row.get("selection_scope", "No selection performed on evaluation"),
                "selection_reasons": legacy.canonical(row.get("selection_reasons", [])),
                "eligibility_checks": legacy.canonical(row.get("eligibility_checks")),
                "ranking_rule": legacy.canonical(row.get("ranking_rule")),
                "execution_status": score["execution_status"],
                "valid": score["valid"], "decision": score["decision"],
                "buyer_payoff": score["proposed_self_payoff"],
                "supplier_payoff": score["proposed_counterpart_payoff"],
                "joint_payoff": score["proposed_joint_gain"],
                "agreement_adjusted_joint_payoff": score["agreement_adjusted_joint_gain"],
                "raw_response": row["raw_response"]}
            if prospective:
                exported.update(primary_outcome=result.get('primary_outcome'),
                    agreement_adjusted_buyer_payoff=score.get('agreement_adjusted_self_payoff'),
                    agreement_adjusted_supplier_payoff=score.get('agreement_adjusted_counterpart_payoff'),
                    buyer_below_reservation=score.get('buyer_below_reservation'))
            escaped = []
            geometry = row.get('geometry') or {}
            for key in ('direction_norm', 'cosine_with_intended', 'intended_displacement_l2',
                        'intended_projection_on_target', 'realized_displacement_l2_min',
                        'realized_displacement_l2_max', 'realized_displacement_observations',
                        'realized_displacement_source', 'realized_displacement_source_sha256'):
                exported[key] = geometry.get(key)
            exported['geometry_status'] = geometry.get('status', 'NOT_AVAILABLE_IN_THIS_RECORD')
            for key, value in exported.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    exported[key] = "'" + value
                    escaped.append(key)
            exported["csv_escaped_fields"] = ",".join(escaped)
            writer.writerow(exported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["history", "compare-saved", "check-inputs"])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--attempt", choices=["a05", "a06", "a09", "a10"], default="a10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real-saved-records", action="store_true")
    parser.add_argument("--original-plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; preserve previous exports")
    if args.action == "history":
        result = read_calibration(args.attempt)
    else:
        if not args.manifest:
            parser.error("--manifest is required")
        header = legacy.load_json(args.manifest)
        if header.get("fixture_only") is not True and not args.allow_real_saved_records:
            parser.error("Real saved records require explicit --allow-real-saved-records")
        if args.action == "check-inputs":
            from .admission import verify_inputs_and_attempts
            if not args.original_plan or not args.plan_sha256:
                parser.error("--original-plan and its separately pinned --plan-sha256 are required")
            result = verify_inputs_and_attempts(args.manifest, args.original_plan, args.plan_sha256)
            result["artifact_type"] = "INPUT_ADMISSION_REVIEW"
        else:
            result = analyze_saved(args.manifest)
    write_new(args.output, result)


if __name__ == "__main__":
    main()

