"""Reuse A03 unchanged; add one separately authorized completion session. No execution on import."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
from unittest.mock import patch

try:
    from . import prepare as spec
except ImportError:
    import prepare as spec

PLAN_SHA = "1c85479ab372e13ec72d3d257375ae07c0e66520f9f1b2c5286872898fc166c1"
BUILD = "/opt/repeatability-build"
EVIDENCE_BYTES = 4 * 1024**3
FILES = ("__init__.py", "prepare.py", "adapter.py")


def check_plan(plan, request, original):
    if spec.digest(plan) != PLAN_SHA:
        raise ValueError("Frozen repeatability plan changed")
    spec.validate_plan(plan, request, original)


def session_inputs(plan, request, original, session_id):
    check_plan(plan, request, original)
    selected = [s for s in plan["sessions"] if s["session_id"] == session_id]
    if len(selected) != 1:
        raise ValueError("Unknown session; no additional session is allowed")
    new_request, new_plan = deepcopy(request), deepcopy(original)
    new_request["run_id"] = session_id
    new_request["manifest"].update(run_id=session_id, phase="repeatability")
    rows = []
    for index, item in enumerate(selected[0]["trials"]):
        row = deepcopy(item["original"])
        row.update(run_id=session_id, phase="repeatability", trial_id=item["repeat_trial_id"],
                   source_sample_id=item["repeat_trial_id"], source_log_path=f"source-{index:04d}.json",
                   original_trial_id=item["original_trial_id"], repeatability_plan_sha256=PLAN_SHA)
        rows.append(row)
    new_plan.pop("preparation_only", None)
    new_plan.pop("prompt_review", None)
    new_plan.update(run_id=session_id, phase="repeatability", trials=rows,
                    source_logs=[r["source_log_path"] for r in rows],
                    repeatability_plan_sha256=PLAN_SHA, source_request_sha256=spec.SOURCE_REQUEST)
    return new_request, new_plan


@contextmanager
def session_connection(connection):
    """Only the inventory/time limit and provenance phase differ from A03."""
    old_phase = connection.phase_plan
    def phase(request, original):
        value = old_phase(request, original)
        value["phase"] = "repeatability"
        return value
    limits = {**connection.LIMITS, "maximum_generation_calls": 36,
              "worker_seconds": 3600, "container_seconds": 3900, "maximum_reserved_usd": 5.0}
    with patch.object(connection, "LIMITS", limits), patch.object(connection, "phase_plan", phase):
        yield connection


def check_approval(plan, approval, transport_hash):
    if (spec.digest(plan) != PLAN_SHA or approval.get("owner_authorized") is not True
            or approval.get("plan_sha256") != PLAN_SHA
            or approval.get("transport_sha256") != transport_hash
            or spec.digest(approval.get("limits")) != spec.digest(spec.LIMITS)
            or approval.get("session_ids") != [s["session_id"] for s in plan["sessions"]]
            or approval.get("automatic_retry") is not False
            or approval.get("deployment_authorized") is not False
            or not isinstance(approval.get("owner_authorization_reference"), str)
            or not approval["owner_authorization_reference"].strip()):
        raise PermissionError("Exact one-session approval and source binding required")


def check_payloads(transport, payloads):
    entries = {r["path"]: r for r in transport["files"]}
    if (transport.get("artifact_type") != "FROZEN_REPEATABILITY_TRANSPORT"
            or transport.get("plan_sha256") != PLAN_SHA
            or transport.get("run_id") != spec.RUN_ID
            or transport.get("limits") != spec.LIMITS
            or transport.get("base_image_id") != "im-T0JhRRVpTMVV3J7CIpVhsZ"
            or transport.get("inventory_sha256") != spec.digest(transport["files"])
            or len(entries) != len(transport["files"]) or set(entries) != set(payloads)):
        raise ValueError("Transport settings or membership changed")
    for name, raw in payloads.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("Unsafe source path")
        if len(raw) != entries[name]["size_bytes"] or hashlib.sha256(raw).hexdigest() != entries[name]["sha256"]:
            raise ValueError("Source changed: " + name)
    old = json.loads(payloads["repeatability/source-transport.json"])
    if spec.digest(old) != spec.SOURCE_TRANSPORT:
        raise ValueError("A03 source inventory changed")
    for entry in old["files"]:
        if entries.get(entry["path"]) != entry:
            raise ValueError("A03 source was changed: " + entry["path"])
    additions = {"repeatability/" + name for name in FILES} | {
        "repeatability/source-transport.json", "repeatability/plan.json"}
    if set(entries) != {r["path"] for r in old["files"]} | additions:
        raise ValueError("Unexpected executable source")
    request = json.loads(payloads["request/research.json"])
    original = json.loads(payloads["request/prepared-plan.json"])
    plan = json.loads(payloads["repeatability/plan.json"])
    if spec.digest(request) != spec.SOURCE_REQUEST:
        raise ValueError("A03 request changed")
    check_plan(plan, request, original)
    return plan, request, original


def build(directory):
    spec.check_prepared(spec.ROOT / "prepared")
    with zipfile.ZipFile(spec.EXECUTION / "source_payload.zip") as archive:
        payloads = {name: archive.read(name) for name in archive.namelist()}
    payloads.update({"repeatability/" + name: (spec.ROOT / name).read_bytes() for name in FILES})
    payloads["repeatability/source-transport.json"] = (spec.EXECUTION / "transport.json").read_bytes()
    payloads["repeatability/plan.json"] = (spec.ROOT / "prepared/plan.json").read_bytes()
    files = [{"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
             for name, raw in sorted(payloads.items())]
    transport = {"artifact_type": "FROZEN_REPEATABILITY_TRANSPORT", "run_id": spec.RUN_ID,
                 "plan_sha256": PLAN_SHA, "source_transport_sha256": spec.SOURCE_TRANSPORT,
                 "base_image_id": "im-T0JhRRVpTMVV3J7CIpVhsZ", "limits": spec.LIMITS,
                 "files": files, "inventory_sha256": spec.digest(files)}
    check_payloads(transport, payloads)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    spec.save_new(directory / "transport.json", transport)
    with zipfile.ZipFile(directory / "source_payload.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payloads.items()):
            archive.writestr(name, raw)
    spec.save_new(directory / "source_archive.json", {"sha256": spec.file_sha(directory / "source_payload.zip")})
    return {"status": "PREPARED_NOT_LAUNCHED", "files": len(files), "transport_sha256": spec.digest(transport)}


def load(directory):
    directory = Path(directory)
    if spec.file_sha(directory / "source_payload.zip") != spec.read(directory / "source_archive.json")["sha256"]:
        raise ValueError("Archive changed")
    with zipfile.ZipFile(directory / "source_payload.zip") as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate archive entry")
        payloads = {name: archive.read(name) for name in archive.namelist()}
    transport = spec.read(directory / "transport.json")
    check_payloads(transport, payloads)
    for name in FILES:
        if (spec.ROOT / name).read_bytes() != payloads["repeatability/" + name]:
            raise ValueError("Adapter changed after freezing: " + name)
    return transport, payloads


def unpack(payloads, root):
    for name, raw in payloads.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def connection(root):
    sys.path.insert(0, str(Path(root) / "code"))
    sys.path.insert(0, str(root))
    from review_package import gpu_connection
    return gpu_connection


def worker(root, approval, session_id, snapshot, destination, transport_hash):
    root, destination = Path(root), Path(destination)
    plan = spec.read(root / "repeatability/plan.json")
    check_approval(plan, approval, transport_hash)
    request, original = session_inputs(plan, spec.read(root / "request/research.json"),
                                      spec.read(root / "request/prepared-plan.json"), session_id)
    route = connection(root)
    route.runtime(root)
    from local_integration.package_qualification import validate_plan
    from local_integration import qualification_worker, research_pipeline
    from local_integration.reused_model import SequentialModelSession, ReusedInspectGenerator
    from local_integration.calibration_diagnostics import calibration_diagnostic_connection
    from local_integration.generation_sensitivity import verify_tokenizer_files
    from review_package.cache_coverage import guarded_cache_summary
    from review_package.runtime_preflight import load_with_runtime_record
    from transformers import AutoTokenizer
    contract = validate_plan(request["qualification"])
    verify_tokenizer_files(request["qualification"], snapshot, root)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    for row in original["trials"]:
        ids = tokenizer.apply_chat_template(row["execution_input"]["messages"], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)
        if ids != row["execution_input"]["input_ids"] or not 0 < len(ids) <= contract.max_input_tokens:
            raise ValueError("Input tokens changed; no model load")
    destination.mkdir(parents=True, exist_ok=False)
    spec.save_new(destination / "approved-plan.json", original)
    spec.save_new(destination / "approval.json", approval)
    spec.save_new(destination / "process.json", {"pid": os.getpid(), "session_id": session_id,
                  "modal_task_id": os.environ.get("MODAL_TASK_ID"), "plan_sha256": PLAN_SHA})
    directions = route.load_directions(root, request)
    def loader(directory):
        model, tokens, _ = load_with_runtime_record(request["qualification"], snapshot, directory, qualification_worker)
        return model, tokens
    session = SequentialModelSession(destination / "shared_model", loader,
        expected_weights_sha256=request["qualification"]["model_weight_entries_sha256"], contract=contract)
    recorder = research_pipeline.summarize_cache
    def guarded(cache, **kwargs):
        return guarded_cache_summary(recorder, cache, **kwargs)
    with session_connection(route), calibration_diagnostic_connection(), patch.object(research_pipeline, "summarize_cache", guarded):
        generator = ReusedInspectGenerator(route.phase_plan(request, original), contract, model_session=session,
            direction_loader=lambda edit, _: directions[edit["direction_sha256"]])
        outer, _ = route.execute_sequence(request, original, destination / "approved-plan.json",
            destination / "execution", generator, route.mechanical_verifier(request, directions))
    if session.failed or session.loads != 1 or session.generation_attempts != 36 or session.completed_trials != 36:
        raise RuntimeError("Unexpected model load or generation count")
    spec.save_new(destination / "result.json", {"status": "EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED",
        "plan_sha256": PLAN_SHA, "source_request_sha256": spec.SOURCE_REQUEST, "session_id": session_id,
        "session": session.summary(), "attempt_count": outer["attempt_count"]})


def remote(approval, transport, payloads, session_id, claim, commit):
    plan, request, original = check_payloads(transport, payloads)
    check_approval(plan, approval, spec.digest(transport))
    session_inputs(plan, request, original, session_id)
    delivery = connection(BUILD).runtime(BUILD)
    if not claim(session_id, {"plan_sha256": PLAN_SHA, "transport_sha256": spec.digest(transport)}):
        return {"status": "ALREADY_CLAIMED_NO_EXECUTION", "run_id": session_id, "files": {}}
    outer = Path(delivery.MOUNT) / delivery.result_prefix(session_id)
    outer.mkdir(parents=True, exist_ok=False)
    spec.save_new(outer / "claimed.json", {"session_id": session_id, "plan_sha256": PLAN_SHA})
    commit()
    try:
        root = Path("/tmp/repeatability-execution")
        unpack(payloads, root)
        spec.save_new(root / "repeatability/approval.json", approval)
        with patch.dict(os.environ, {"PYTHONPATH": str(root / "code") + os.pathsep + str(root)}):
            process = delivery.run_process([sys.executable, "-B", str(root / "repeatability/adapter.py"),
                "worker", "--root", str(root), "--approval", str(root / "repeatability/approval.json"),
                "--session", session_id, "--snapshot", delivery.SNAPSHOT, "--output", str(outer / "attempt"),
                "--transport-hash", spec.digest(transport)], root, outer / "stdout.log", outer / "stderr.log", 3600)
        result = spec.read(outer / "attempt/result.json") if (outer / "attempt/result.json").is_file() else {}
        status = ("EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED" if process["returncode"] == 0
                  and not process["timed_out"] and result.get("status") == "EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED"
                  else "FAILED_OR_INDETERMINATE_NO_RETRY")
        terminal = {"run_id": session_id, "status": status, "plan_sha256": PLAN_SHA,
                    "process": process, "automatic_retry": False}
    except BaseException as error:
        terminal = {"run_id": session_id, "status": "FAILED_NO_RETRY", "plan_sha256": PLAN_SHA,
                    "error_type": type(error).__name__, "error": str(error), "automatic_retry": False}
    spec.save_new(outer / "terminal.json", terminal)
    with patch.object(delivery, "MAX_EVIDENCE_TOTAL_BYTES", EVIDENCE_BYTES):
        return delivery.publish_result(outer, terminal, transport, commit)


def driver(approval, transport, payloads):
    import modal
    stage = Path(tempfile.mkdtemp(prefix="repeatability-image-"))
    unpack(payloads, stage)
    delivery = connection(stage).runtime(stage)
    requirements = [line.strip() for line in payloads[delivery.LOCK].decode().splitlines()
                    if line.strip() and not line.startswith(("#", "./upstream"))]
    image = (modal.Image.from_id(delivery.BASE_IMAGE).pip_install(requirements)
             .add_local_dir(str(stage), BUILD, copy=True)
             .pip_install(BUILD + "/" + delivery.UPSTREAM, extra_options="--no-deps"))
    volume = modal.Volume.from_name(delivery.VOLUME, create_if_missing=False)
    app = modal.App("orgsci-same-prompt-repeatability-r02", include_source=False)
    @app.function(image=image, gpu="L40S", cpu=(8.0, 8.0), memory=(65536, 65536),
        volumes={delivery.MOUNT: volume}, timeout=3900, retries=0, max_containers=1, max_inputs=1,
        block_network=True, serialized=True, include_source=False,
        env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    def execute(approval, transport, payloads, session_id):
        import sys
        import modal
        sys.path.insert(0, "/opt/repeatability-build")
        from repeatability import adapter
        claims = modal.Dict.from_name("orgsci-repeatability-one-use", create_if_missing=True)
        volume = modal.Volume.from_name("wp3-qwen35-l40s-modal-01-data", create_if_missing=False)
        return adapter.remote(approval, transport, payloads, session_id,
            lambda key, value: claims.put(key, value, skip_if_exists=True), volume.commit)
    return modal, app, execute


def launch(directory, approval_path):
    transport, payloads = load(directory)
    plan, _, _ = check_payloads(transport, payloads)
    approval = spec.read(approval_path)
    check_approval(plan, approval, spec.digest(transport))
    path = Path(directory) / "launch"
    path.mkdir(exist_ok=False)
    spec.save_new(path / "submission-claim.json", {"run_id": spec.RUN_ID, "plan_sha256": PLAN_SHA})
    spec.save_new(path / "approval.json", approval)
    receipts = []
    try:
        modal, app, function = driver(approval, transport, payloads)
        with modal.enable_output(), app.run(detach=True):
            for session in plan["sessions"]:
                target = path / session["session_id"]
                target.mkdir(exist_ok=False)
                spec.save_new(target / "submission-claim.json", {"session_id": session["session_id"], "plan_sha256": PLAN_SHA})
                call = function.spawn(approval, transport, payloads, session["session_id"])
                receipt = {"artifact_type": "PACKAGE_QUALIFICATION_DETACHED_LAUNCH", "run_id": session["session_id"],
                    "modal_app_id": app.app_id, "modal_function_call_id": call.object_id,
                    "transport_sha256": spec.digest(transport), "status": "SUBMITTED_NOT_VERIFIED"}
                spec.save_new(target / "launch.json", receipt)
                receipts.append(receipt)
        return {"status": "ONE_SESSION_SUBMITTED_NOT_VERIFIED", "sessions": receipts}
    except BaseException as error:
        spec.save_new(path / "submission-error.json", {"status": "ERROR_OR_INDETERMINATE_NO_RETRY",
            "error_type": type(error).__name__, "error": str(error), "submitted_sessions": receipts})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check", "launch", "retrieve", "worker"])
    for name in ("directory", "root", "approval", "output", "snapshot"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--session")
    parser.add_argument("--transport-hash")
    args = parser.parse_args()
    if args.action == "build":
        result = build(args.directory)
    elif args.action == "worker":
        worker(args.root, spec.read(args.approval), args.session, args.snapshot, args.output, args.transport_hash)
        return
    elif args.action == "launch":
        result = launch(args.directory, args.approval)
    else:
        transport, payloads = load(args.directory)
        plan, _, _ = check_payloads(transport, payloads)
        if args.action == "check":
            result = {"status": "SOURCE_CHECKED_NO_PROVIDER_CONTACT", "transport_sha256": spec.digest(transport), "plan_sha256": PLAN_SHA}
        else:
            if args.session not in [s["session_id"] for s in plan["sessions"]]:
                raise ValueError("Exact session required")
            with tempfile.TemporaryDirectory(prefix="repeatability-retrieve-") as temporary:
                unpack(payloads, temporary)
                delivery = connection(temporary).runtime(temporary)
                with patch.object(delivery, "MAX_EVIDENCE_TOTAL_BYTES", EVIDENCE_BYTES):
                    result = delivery.retrieve(args.directory / "launch" / args.session)
    print(spec.canonical(result))


if __name__ == "__main__":
    main()

