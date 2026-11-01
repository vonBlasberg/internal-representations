"""Combine two complete, independently audited repeats. No model or provider call."""
import csv
import hashlib
import json
from pathlib import Path

import audit

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "REPEATABILITY_CHECK_2026-10-31"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "verification/combined-terminal-A01"
    assert not destination.exists(), "Never overwrite a combined audit"
    old, new = read(OLD / "prepared/plan.json"), read(ROOT / "prepared/plan.json")
    for field in ("source_request_sha256", "source_transport_sha256", "source_report_sha256",
                  "source_case_ids", "cases", "model", "generation_contract", "decoding",
                  "runtime_reference", "application_rule"):
        assert old[field] == new[field], field
    assert read(OLD / "prepared/reference.json") == read(ROOT / "prepared/reference.json")
    selected = [old["sessions"][1], new["sessions"][0]]
    assert [r["original"] for r in selected[0]["trials"]] == [r["original"] for r in selected[1]["trials"]]
    sources = [ROOT / "verification/retained-R01-repeat-2-A01", ROOT / "verification/R02-terminal-A01"]
    rows, sessions, inputs = [], [], {}
    for session, source in zip(selected, sources):
        report = read(source / "audit.json")
        assert report["status"] == "REPEATABILITY_INDEPENDENTLY_VERIFIED"
        assert report["audited_session_ids"] == [session["session_id"]]
        assert report["output_denominator"] == 36 and report["contrast_denominator"] == 12
        assert report["sessions"] == 1 and len(report["audited_call_ids"]) == 1
        current = read(source / "outcomes.json")
        assert len(current) == 36
        rows.extend(current)
        sessions.append(report)
        for name in ("audit.json", "outcomes.json", "comparison.json"):
            inputs[str((source / name).relative_to(ROOT))] = sha(source / name)
    assert len({s["audited_call_ids"][0] for s in sessions}) == 2
    assert len({s["session_id"] for s in selected}) == 2
    result = audit.comparisons(rows, read(ROOT / "prepared/reference.json")["rows"], {**new, "sessions": selected})
    assert result["output_denominator"] == 72 and result["contrast_denominator"] == 24
    for path in (OLD / "prepared/plan.json", ROOT / "prepared/plan.json",
                 ROOT / "prepared/reference.json", Path(__file__), ROOT / "audit.py"):
        inputs[str(path.relative_to(ROOT.parent))] = sha(path)
    failed = OLD / "verification/R01-terminal-failure-A01.md"
    inputs[str(failed.relative_to(ROOT.parent))] = sha(failed)
    summary = {"status": "TWO_COMPLETE_REPEATS_INDEPENDENTLY_VERIFIED", "sessions": sessions,
               "source_sha256": inputs, "source_request_sha256": new["source_request_sha256"],
               "source_report_sha256": new["source_report_sha256"],
               "failed_attempt": {"session_id": old["sessions"][0]["session_id"],
                                  "status": "TIMEOUT", "completed_prefix": 32,
                                  "included_in_complete_repeat_totals": False},
               **{key: value for key, value in result.items() if key not in ("outputs", "contrasts")}}
    destination.mkdir()
    for name, value in (("summary.json", summary), ("comparison.json", result), ("outcomes.json", rows)):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")
    with (destination / "outputs.csv").open("x", newline="") as handle:
        fields = ["trial_id", "session_id", "original_trial_id", "case_id", "condition", "behavior_matches",
                  "text_matches", "tokens_match", "processed_logits_match", "different_outcome_fields"]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key]
                         for key in fields} for row in result["outputs"])
    print(json.dumps({key: value for key, value in summary.items() if key not in ("sessions", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
