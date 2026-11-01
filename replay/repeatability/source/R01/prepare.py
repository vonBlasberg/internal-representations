"""Prepare/check exact repeatability inputs. No model, network or launch entry point."""
import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent / "INTERVENTION_REVIEW_PACKAGE_2026-10-31"
EXECUTION = PACKAGE / "execution/fixed-comparison-2026-10-31-A03"
REPORT = PACKAGE / "outputs/fixed-comparison-verified-A03"
RUN_ID = "negotiation-same-prompt-repeatability-2026-10-31-R01"
SOURCE_REQUEST = "9ca15d9337561c2d868b35d0419b9e2ad418102d21d6f21a56a13c1045a82953"
SOURCE_TRANSPORT = "1a398ddefb7a35df9715afc7fcf368b3ab2796a05e44cb5f26d14103b3b6be96"
SOURCE_REPORT = "8f6d7bf4951ea16e95fcb77031027aab6aa8a9720695716ba2a344a1dd2ac0a0"
CONDITIONS = ("unchanged", "prompt", "internal")
LIMITS = {
    "sessions": 2, "generations_per_session": 36, "maximum_new_generations": 72,
    "maximum_model_loads": 2, "maximum_parallel_containers": 2,
    "gpu_per_container": "L40S", "gpu_count_per_container": 1,
    "cpu_physical_cores_per_container": 8, "memory_mib_per_container": 65536,
    "worker_seconds_per_session": 2400, "container_seconds_per_session": 2700,
    "retries": 0, "maximum_reserved_usd": 10.0,
    "new_checkpoint_download": False, "deployment": False,
}
OUTCOME_FIELDS = (
    "valid", "decision", "package", "effective_package", "agreement",
    "proposed_self_payoff", "proposed_counterpart_payoff", "proposed_joint_gain",
    "agreement_adjusted_self_payoff", "agreement_adjusted_counterpart_payoff",
    "agreement_adjusted_joint_gain", "buyer_below_reservation",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open("x") as handle:
        handle.write(canonical(value) + "\n")


def load_sources():
    request, original, transport = [read(EXECUTION / name) for name in
                                   ("request.json", "prepared-plan.json", "transport.json")]
    audit = read(REPORT / "audit.json")
    if digest(request) != SOURCE_REQUEST or digest(transport) != SOURCE_TRANSPORT:
        raise ValueError("Original request or transport changed")
    if file_sha(REPORT / "report/review.json") != SOURCE_REPORT:
        raise ValueError("Verified report changed")
    if (audit["status"] != "FIXED_COMPARISON_INDEPENDENTLY_VERIFIED"
            or audit["attempts"] != 324 or audit["delivered_files"] != 2604
            or audit["request_sha256"] != SOURCE_REQUEST
            or audit["transport_sha256"] != SOURCE_TRANSPORT
            or audit["zero_edit_checks"] != 36):
        raise ValueError("Independent terminal verification differs")
    archive = EXECUTION / "source_payload.zip"
    if file_sha(archive) != read(EXECUTION / "source_archive.json")["sha256"]:
        raise ValueError("Original source archive changed")
    if digest(original) != request["prepared_plan_sha256"]:
        raise ValueError("Original prepared inputs changed")
    entries = {item["path"]: item for item in transport["files"]}
    if len(entries) != 454 or digest(transport["files"]) != transport["inventory_sha256"]:
        raise ValueError("Original source inventory differs")
    with zipfile.ZipFile(archive) as payload:
        if len(payload.namelist()) != len(set(payload.namelist())) or set(payload.namelist()) != set(entries):
            raise ValueError("Original payload membership differs")
        for name, entry in entries.items():
            raw = payload.read(name)
            if len(raw) != entry["size_bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("Original payload changed: " + name)
            if name.startswith(("code/", "protocol/")) and (PACKAGE / name).read_bytes() != raw:
                raise ValueError("Working scientific source changed: " + name)
        if json.loads(payload.read("request/research.json")) != request:
            raise ValueError("Readable request differs from archived request")
        if json.loads(payload.read("request/prepared-plan.json")) != original:
            raise ValueError("Readable inputs differ from archived inputs")
    return request, original, read(REPORT / "report/review.json")


def select_trials(request, original):
    case_ids = [case["case_id"] for case in request["cases"] if case["variant_kind"] == "base"]
    if len(case_ids) != 12 or len(set(case_ids)) != 12:
        raise ValueError("Expected all twelve original-wording cases")
    selected = [deepcopy(row) for row in original["trials"]
                if row["case_id"] in case_ids and row["condition"] in CONDITIONS]
    expected = [(case_id, condition) for case_id in case_ids for condition in CONDITIONS]
    if [(r["case_id"], r["condition"]) for r in selected] != expected:
        raise ValueError("Subset membership or inherited order differs")
    for case_id in case_ids:
        rows = {r["condition"]: r for r in selected if r["case_id"] == case_id}
        if rows["unchanged"]["execution_input"] != rows["internal"]["execution_input"]:
            raise ValueError("Baseline and internal inputs differ")
        edit = rows["internal"]
        if (edit["site"] != "model.language_model.layers.14" or edit["coefficient"] != 0.5
                or edit["direction_sha256"] != "dba785066f697f22ae67b7416c7231779567a81eb9f5c571b98c02ba8c8824cf"):
            raise ValueError("Frozen intervention differs")
    return case_ids, selected


def build_plan(request, original):
    case_ids, selected = select_trials(request, original)
    sessions = []
    for repeat in (1, 2):
        session_id = RUN_ID + "-repeat-" + str(repeat)
        sessions.append({
            "session_id": session_id, "fresh_process": True, "fresh_model_load": True,
            "trial_count": 36,
            "trials": [{"repeat_trial_id": session_id + "--" + row["trial_id"],
                        "original_trial_id": row["trial_id"],
                        "original": deepcopy(row)} for row in selected],
        })
    rate = 0.000542 + 8 * 0.0000131 + 64 * 0.00000222
    return {
        "artifact_type": "SAME_PROMPT_REPEATABILITY_PLAN", "schema_version": 1,
        "run_id": RUN_ID, "status": "PREPARED_NOT_AUTHORIZED", "owner_authorized": False,
        "source_request_sha256": SOURCE_REQUEST, "source_transport_sha256": SOURCE_TRANSPORT,
        "source_report_sha256": SOURCE_REPORT,
        "selection_rule": "All 12 base/original-wording cases; inherited unchanged, prompt, internal order. No selection by observed effect.",
        "source_case_ids": case_ids,
        "cases": [deepcopy(c) for c in request["cases"] if c["case_id"] in case_ids],
        "model": {k: request["manifest"][k] for k in ("model_id", "model_revision")},
        "generation_contract": deepcopy(request["qualification"]["generation_contract"]),
        "decoding": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                     "enable_thinking": False, "seed_changes": False},
        "runtime_reference": deepcopy(request["qualification"]["runtime_reference"]),
        "application_rule": "cast_direction_to_model_dtype_then_scale_then_add_at_last_active_prefill_position_and_each_decode_position",
        "limits": deepcopy(LIMITS), "sessions": sessions,
        "analysis": {
            "primary_question": "Do fresh executions reproduce each saved decision/package and the baseline-versus-edit contrast?",
            "reference": "The existing A03 execution, not a new observation or another calibration.",
            "behavior_fields": list(OUTCOME_FIELDS),
            "full_behavioral_repeatability": "All 72 new case/condition outputs match the saved valid decision, package, acceptance and payoff fields, and all 24 per-case edit-versus-baseline contrasts match.",
            "text_token_identity": "Report separately for every output; formatting-only differences do not become behavioral differences.",
            "failure_reporting": "Retain every mismatch and invalid response. A mechanical/source failure stops that session; no replacement or retry. A technically valid behavioral mismatch is a result and does not trigger reselection or early selective truncation.",
            "uncertainty": "Descriptive fixed-case repeatability only. No population p-values, equivalence claim or independent-observation inflation from greedy repeats.",
            "scope": "Original wording and frozen greedy configuration only; not cross-wording, stochastic, cross-model, cooperation-construct or human-study validation.",
        },
        "execution_requirements": [
            "Separate isolated worker process and fresh checkpoint load in each session; no cross-session cache or model reuse.",
            "Retain the qualified Transformers/PyTorch, Steerability and Inspect route and frozen scorer.",
            "Verify exact messages, input token IDs, model/runtime, applied edit, restoration and output-token/response correspondence.",
            "Give each session/trial a new identity linked to the original; retain original metadata as provenance.",
            "One-use approval and independent source-bound terminal audit are required before results are accepted.",
            "The old 324-trial transport must not be invoked or altered for this check. A separately capped subset adapter must pass offline dispatch/approval tests before launch.",
        ],
        "cost": {
            "checked_date": "2026-10-31", "source": "https://modal.com/pricing",
            "rate_usd_per_hour_per_container": round(rate * 3600, 6),
            "gpu_rate_usd_per_second": 0.000542, "cpu_core_rate_usd_per_second": 0.0000131,
            "memory_gib_rate_usd_per_second": 0.00000222,
            "original_324_completion_span_seconds": 12933,
            "simple_72_response_time_seconds": 12933 / 324 * 72,
            "expected_parallel_gpu_wall_minutes": [30, 45],
            "expected_retrieval_and_verification_minutes": [10, 20],
            "planning_end_to_end_minutes": [45, 65],
            "expected_compute_usd": [3, 5],
            "configured_resource_time_ceiling_usd": round(2 * 2700 * rate, 4),
            "approval_reservation_usd": 10,
            "limitations": "Planning estimates, not invoices. Provider queue, image build/setup, egress, storage and tax are additional; no new subscription or region premium. Reservation is not a provider-enforced invoice cap.",
        },
        "non_actions": {"model_loaded": False, "provider_called": False, "run_launched": False,
                        "historical_results_changed": False, "design_of_original_study_changed": False,
                        "deployment": False},
    }


def validate_plan(plan, request, original):
    if plan != build_plan(request, original):
        raise ValueError("Repeatability plan differs from the exact permitted subset")
    return {"status": "EXACT_INPUTS_CHECKED_NOT_AUTHORIZED", "trials": 72,
            "original_case_condition_pairs": 36, "cases": 12, "sessions": 2}


def reference_rows(request, original, report):
    _, selected = select_trials(request, original)
    by_id = {row["trial_id"]: row for row in report["all_trials"]}
    root = EXECUTION / "launch/returned/evidence/attempt/execution"
    rows, preservation = [], []
    for trial in selected:
        old = by_id[trial["trial_id"]]
        state_path = root / "trials" / trial["trial_id"] / "state_execution.json"
        state = read(state_path)["records"]
        if len(state) != 1 or state[0]["input_token_ids"] != trial["execution_input"]["input_ids"]:
            raise ValueError("Saved executed input differs")
        if not old["score"]["valid"]:
            raise ValueError("Unexpected invalid reference in original-wording subset")
        row = {"original_trial_id": trial["trial_id"], "case_id": trial["case_id"],
               "condition": trial["condition"], "raw_response": old["raw_response"],
               "generated_token_ids": state[0]["generated_token_ids"],
               "score": deepcopy(old["score"]),
               "state_execution_sha256": file_sha(state_path)}
        rows.append(row)
        preservation.append({"path": str(state_path.relative_to(PACKAGE)), "sha256": file_sha(state_path)})
    return rows, preservation


def readable_prompts(plan):
    text = "# Exact retained messages\n\nNo prompts are rewritten. Internal and baseline messages are identical. The selected prompt remains appended to the system message.\n"
    for trial in plan["sessions"][0]["trials"]:
        source = trial["original"]
        if source["condition"] != "internal":
            text += "\n## " + source["trial_id"] + "\n"
            for message in source["execution_input"]["messages"]:
                text += "\n### " + message["role"] + "\n\n" + message["content"] + "\n"
    return text


def readable_cases(plan, reference):
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(["case_id", "baseline_decision", "internal_decision", "prompt_decision", "internal_changed"])
    for case_id in plan["source_case_ids"]:
        rows = {v["condition"]: v for v in reference if v["case_id"] == case_id}
        decisions = [rows[c]["score"]["decision"] for c in CONDITIONS]
        writer.writerow([case_id, decisions[0], decisions[2], decisions[1], decisions[0] != decisions[2]])
    return text.getvalue()


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Preserve existing preparation; choose a new directory")
    request, original, report = load_sources()
    plan = build_plan(request, original)
    check = validate_plan(plan, request, original)
    reference, preservation = reference_rows(request, original, report)
    destination.mkdir(parents=True, exist_ok=False)
    save_new(destination / "plan.json", plan)
    save_new(destination / "reference.json", {"source_report_sha256": SOURCE_REPORT, "rows": reference})
    save_new(destination / "source_checks.json", {
        **check, "plan_sha256": digest(plan), "verified_source_payloads": 454,
        "request_sha256": SOURCE_REQUEST, "transport_sha256": SOURCE_TRANSPORT,
        "report_sha256": SOURCE_REPORT, "reference_state_records": preservation,
        "model_loads": 0, "provider_calls": 0,
        "execution_adapter_status": "Not supplied by this planning package; frozen 324-trial adapter is not valid for the subset.",
    })
    save_new(destination / "approval.NOT_AUTHORIZED.json", {
        "owner_authorized": False, "run_id": RUN_ID, "plan_sha256": digest(plan),
        "limits": LIMITS, "authorization_reference": None, "transport_sha256": None,
        "launch_precondition": "Approve scope/cap, then verify the exact subset adapter and transport without changing scientific settings. No automatic launch from this file.",
    })
    for name, text in [("cases.csv", readable_cases(plan, reference)), ("PROMPTS.md", readable_prompts(plan))]:
        with (destination / name).open("x", newline="") as handle:
            handle.write(text)
    return check


def check_prepared(directory):
    directory = Path(directory)
    request, original, report = load_sources()
    plan = read(directory / "plan.json")
    result = validate_plan(plan, request, original)
    reference, preservation = reference_rows(request, original, report)
    if read(directory / "reference.json") != {"source_report_sha256": SOURCE_REPORT, "rows": reference}:
        raise ValueError("Reference outcomes changed")
    if ((directory / "cases.csv").read_bytes().decode() != readable_cases(plan, reference)
            or (directory / "PROMPTS.md").read_text() != readable_prompts(plan)):
        raise ValueError("Readable review files differ from the exact inputs")
    saved = read(directory / "source_checks.json")
    if saved["plan_sha256"] != digest(plan) or saved["reference_state_records"] != preservation:
        raise ValueError("Preparation binding or saved source records changed")
    if read(directory / "approval.NOT_AUTHORIZED.json")["owner_authorized"] is not False:
        raise ValueError("Preparation does not authorize execution")
    return {**result, "plan_sha256": digest(plan), "reference_rows": len(reference),
            "frozen_payloads_verified": 454, "model_loads": 0, "provider_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "check"])
    parser.add_argument("--directory", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    result = prepare(args.directory) if args.action == "prepare" else check_prepared(args.directory)
    print(json.dumps(result, indent=2))

