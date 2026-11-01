"""Read-only presentation of one independently verified calibration.

No discovery of running jobs, provider access, model imports or selection writes.
An explicit, hash-pinned publication is required before another run can appear.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "records/construction_calibration_2026-10-30_A04/calibration/"
CORE_FILES = {
    BASE + "terminal_audit.json": "a289a4b620e7542511f983b7a5e2789c260c0e7d4b960e401ac040bd2f80095b",
    BASE + "terminal_closeout.json": "b00640495fbe4210c2d969948ff723003eb430c8895bdce87f6ac472e35f6988",
    BASE + "launch/returned/evidence/attempt/selection.json": "ab97299c0c711dc8558644653870f08029f0c66bb1d09d033244ba55afdbba67",
    BASE + "launch/returned/evidence/attempt/calibration_summary.json": "8cb9083d6acb08c63a631271a7d3017c7ee60eeca24e2c36539fee1805b76c1f",
}
COMPARISON = "verification/candidate-comparison-2026-10-30.ZnZp39/replay-02/comparison.json"
COMPARISON_SHA256 = "49907bc77eaabf5e7e761526147b6dfce3a72c04ef27c03c3dbac4a348014214"
FILES = {**CORE_FILES, COMPARISON: COMPARISON_SHA256}
REQUEST = "2c6a90b53089928635adccd1d79521a2a5afd00488e191369806a7237e48cbc5"
A05_PUBLICATION = (
    "verification/reviewer-walkthrough-2026-10-30.f9ITU8/a05-publication.json",
    "7202393343420a44e923bf4f23f48cf0a1a4e2c9ceee3865ff0642c90d152cb8",
)
A06_PUBLICATION = (
    "verification/a06-reviewer-import-2026-10-31.IcAnkY/publication/a06-publication.json",
    "815153b224ccb5699937c82713acb4798220b23caea8badd477a5889b2863055",
)
A09_PUBLICATION = (
    "verification/a09-reviewer-import-2026-10-31.yr6PFJ/publication/a09-publication.json",
    "c6c09da4ed8c84a73424b393c1117f10ca04ecf06043a170ddef1fbe867aad28",
)
A10_PUBLICATION = (
    "verification/a10-reviewer-import-2026-10-31.CTBM0H/publication/a10-publication.json",
    "e05e035e3163b5419eb7978eb1f759a351b9197eeeb77ea7fb90ba9adec5b63b",
)
CURRENT_PUBLICATION = A10_PUBLICATION
HISTORICAL_PUBLICATIONS = {"A05": A05_PUBLICATION, "A06": A06_PUBLICATION, "A09": A09_PUBLICATION}


def read_core(root=ROOT):
    documents = []
    for relative, expected in CORE_FILES.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Saved calibration record does not match its verified publication.")
        documents.append(json.loads(data))
    audit, closeout, selection, summary = documents
    if (audit["status"] != "CALIBRATION_VERIFIED" or audit["request_sha256"] != REQUEST
            or closeout["request_sha256"] != REQUEST
            or not closeout["all_41_original_records_tokens_scores_and_settings_checked"]
            or audit["checks"]["selection"] != selection
            or summary["plan_sha256"] != selection["plan_sha256"]
            or len(audit["rows"]) != 41 or len(summary["rows"]) != 41):
        raise ValueError("Incomplete or inconsistent calibration publication.")
    rows = []
    details = {row["arm_id"]: row for row in summary["rows"]}
    for original in audit["rows"]:
        row = dict(original)
        row["edit"] = details[row["arm_id"]]["edit"]
        rows.append(row)
    return {
        "schema_version": 1, "status": "VERIFIED_CALIBRATION", "model_execution": False,
        "run_id": "negotiation-authored-construction-calibration-2026-10-30-A04-calibration",
        "label": "Completed calibration A04", "verified_at": audit["checked_at_utc"],
        "model": "Qwen/Qwen3.5-9B", "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "case_count": 1, "generation_count": 41, "maximum_joint_payoff": 147,
        "selection": selection, "rows": rows, "source_sha256": FILES,
        "scope": "One-case calibration. Not held-out evidence, population inference or a test with human negotiators.",
        "other_run": "The separate ceiling-corrected A05 calibration is not imported into this view. Its records and execution are untouched. Only independently verified terminal results can be added.",
        "state_choices": [
            {"layer": 2, "site": "model.language_model.layers.2", "block_type": "Linear-attention block"},
            {"layer": 3, "site": "model.language_model.layers.3", "block_type": "Full-attention block"},
        ],
        "strengths": [0.5, 1.0, 2.0],
        "robustness": [
            {"check": "Complete execution and independent scoring", "status": "Checked", "detail": "41 responses, original logs and scores checked; the saved selection was reproduced. This page verifies the hashes of those audit records."},
            {"check": "Zero, reverse, random and permuted-label edits", "status": "Checked on one case", "detail": "2 zero edits, 6 reverse edits, 18 edits using three fixed random directions, and 6 permuted-label edits are retained. All returned the baseline decision. This does not establish general selectivity."},
            {"check": "Room for improvement", "status": "Failed in A04", "detail": "The unchanged offer already achieved the maximum joint payoff of 147. No candidate could meet the strict improvement rule. A separate correction does not replace this result."},
            {"check": "Validity, agreement and each party's payoff", "status": "Checked on one case", "detail": "41/41 responses were valid and met the simulated agreement rule. Proposed and agreement-adjusted joint payoffs coincide here; prompting redistributed payoffs without raising their total."},
            {"check": "Paraphrases, issue order and new case families", "status": "Not established", "detail": "A one-case calibration cannot establish these forms of robustness. Related scenarios must stay together when separating calibration from evaluation."},
            {"check": "Independent evaluation and other models", "status": "Not established", "detail": "Freeze any eligible site, direction, strength and prompt before evaluation on held-out families. A model change requires new competence and edit checks. No further run is authorized by this page."},
        ],
    }


def read_a04_verified(root=ROOT):
    data = read_core(root)
    raw = (root / COMPARISON).read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMPARISON_SHA256:
        raise ValueError("Saved comparison does not match its verified publication.")
    comparison = json.loads(raw)
    if (comparison["status"] != "SAVED_COMPARISON_VERIFIED" or comparison["request_sha256"] != REQUEST
            or comparison["run_id"] != data["run_id"] or comparison["selection"] != data["selection"]
            or len(comparison["rows"]) != 41 or comparison["inspect"]["logs_replayed"] != 41
            or not comparison["inspect"]["original_logs_unchanged"]
            or not comparison["inspect"]["all_original_scores_match"]
            or comparison["model_generations"] or comparison["model_loads"] or comparison["provider_requests"]):
        raise ValueError("Incomplete saved-response comparison.")
    by_id = {(r["case_id"], r["arm_id"]): r for r in comparison["rows"]}
    if len(by_id) != 41:
        raise ValueError("Duplicate comparison rows.")
    for row in data["rows"]:
        replayed = by_id[(row["case_id"], row["arm_id"])]
        if replayed["score"] != row["score"] or replayed["raw_response"] != row["raw_response"]:
            raise ValueError("Comparison differs from the original audited response.")
    data["comparison"] = comparison
    return data


def read_verified(root=ROOT, *, calibration="current"):
    if calibration not in {"current", "A04", *HISTORICAL_PUBLICATIONS}:
        raise ValueError("Only explicitly published calibrations are available.")
    if calibration == "A04" or CURRENT_PUBLICATION is None:
        return read_a04_verified(root)
    relative, expected = CURRENT_PUBLICATION if calibration == "current" else HISTORICAL_PUBLICATIONS[calibration]
    raw = (root / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Verified publication changed.")
    data = json.loads(raw)
    if (data["status"] != "VERIFIED_CALIBRATION" or len(data["rows"]) != 41
            or data["model_execution"] or data["selection"]["held_out_result"]
            or data["selection"]["further_execution_authorized"]
            or data["comparison"]["status"] != "SAVED_COMPARISON_VERIFIED"
            or data["comparison"]["inspect"]["logs_replayed"] != 41
            or data["comparison"]["selection"] != data["selection"]
            or data["comparison"]["model_generations"] or data["comparison"]["provider_requests"]):
        raise ValueError("Incomplete independently verified publication.")
    for name, required in data["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != required:
            raise ValueError("A verified source record changed or is unavailable.")
    return data
