"""Independent, model-free verification of the two saved R01 deliveries."""
import argparse
from pathlib import Path
import tempfile
import adapter
import prepare as spec


def require(condition, message):
    if not condition:
        raise ValueError(message)


def comparisons(rows, references, plan):
    reference = {r["original_trial_id"]: r for r in references}
    planned = {r["repeat_trial_id"]: (session["session_id"], r) for session in plan["sessions"] for r in session["trials"]}
    require(len(rows) == 72 and len({r["trial_id"] for r in rows}) == 72
            and {r["trial_id"] for r in rows} == set(planned), "Incomplete comparison inventory")
    details, indexed = [], {}
    for row in rows:
        session_id, pin = planned[row["trial_id"]]
        old = reference[pin["original_trial_id"]]
        require(row["original_trial_id"] == old["original_trial_id"] and row["session_id"] == session_id,
                "Repeated outcome references another original")
        require(set(spec.OUTCOME_FIELDS) <= set(row["score"]) and set(spec.OUTCOME_FIELDS) <= set(old["score"]),
                "Outcome fields missing")
        differences = [key for key in spec.OUTCOME_FIELDS if row["score"][key] != old["score"][key]]
        details.append({"trial_id": row["trial_id"], "session_id": session_id,
            "original_trial_id": old["original_trial_id"], "case_id": old["case_id"], "condition": old["condition"],
            "behavior_matches": not differences, "different_outcome_fields": differences,
            "text_matches": row["raw_response"] == old["raw_response"],
            "tokens_match": row["generated_token_ids"] == old["generated_token_ids"],
            "processed_logits_match": row["processed_logits_match"],
            "score": row["score"], "reference_score": old["score"]})
        indexed[session_id, old["case_id"], old["condition"]] = row
    def contrast(base, edited):
        result = {"baseline_valid": base["valid"], "edited_valid": edited["valid"],
                  "decision_transition": [base["decision"], edited["decision"]],
                  "package_transition": [base["effective_package"], edited["effective_package"]]}
        for field in spec.OUTCOME_FIELDS:
            if "payoff" in field or "joint_gain" in field:
                left, right = base[field], edited[field]
                result[field] = right - left if left is not None and right is not None else None
        return result
    contrasts = []
    for session in plan["sessions"]:
        sid = session["session_id"]
        for case in plan["source_case_ids"]:
            old_base = next(r["score"] for r in references if r["case_id"] == case and r["condition"] == "unchanged")
            old_edit = next(r["score"] for r in references if r["case_id"] == case and r["condition"] == "internal")
            current = contrast(indexed[sid, case, "unchanged"]["score"], indexed[sid, case, "internal"]["score"])
            original = contrast(old_base, old_edit)
            contrasts.append({"session_id": sid, "case_id": case, "matches": current == original,
                              "contrast": current, "reference_contrast": original})
    return {"outputs": details, "contrasts": contrasts,
        "behavior_matching_outputs": sum(r["behavior_matches"] for r in details), "output_denominator": 72,
        "matching_contrasts": sum(r["matches"] for r in contrasts), "contrast_denominator": 24,
        "text_matching_outputs": sum(r["text_matches"] for r in details),
        "token_matching_outputs": sum(r["tokens_match"] for r in details),
        "logit_matching_outputs": sum(r["processed_logits_match"] for r in details),
        "full_behavioral_repeatability": all(r["behavior_matches"] for r in details) and all(r["matches"] for r in contrasts),
        "scope": "Two fresh executions of the twelve original-wording cases, with fixed greedy decoding. Not new independent cases, cross-wording robustness or improved outcomes."}


def audit(directory, output, snapshot):
    directory, output = Path(directory), Path(output)
    require(not output.exists(), "Never overwrite an existing audit")
    spec.check_prepared(spec.ROOT / "prepared")
    transport, payloads = adapter.load(directory)
    plan, source_request, source_plan = adapter.check_payloads(transport, payloads)
    reference = spec.read(spec.ROOT / "prepared/reference.json")["rows"]
    rows, receipts, total_files = [], [], 0
    route = adapter.connection(spec.PACKAGE)
    from review_package.gpu_audit import verify_runtime_preflight
    from review_package.reuse import legacy, binding
    from review_package.scoring import score_completed
    from review_package.outcomes import add_agreement_outcomes
    from review_package.inputs import validate_messages
    with tempfile.TemporaryDirectory(prefix="repeatability-audit-") as temporary:
        root = Path(temporary)
        adapter.unpack(payloads, root)
        route.runtime(root)
        # The diagnostic verifier also imports worker.receipts from this retained package.
        import sys
        sys.path.insert(0, str(root / "negotiation-steering"))
        from transformers import AutoTokenizer
        from local_integration.generation_sensitivity import verify_tokenizer_files
        from inspect_ai.log import read_eval_log
        verify_tokenizer_files(source_request["qualification"], snapshot, root)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
        directions = route.load_directions(root, source_request)
        cases = {c["case_id"]: c for c in plan["cases"]}
        for selected in plan["sessions"]:
            sid = selected["session_id"]
            launch = directory / "launch" / sid
            receipt = spec.read(launch / "launch.json")
            require(receipt["run_id"] == sid and receipt["transport_sha256"] == spec.digest(transport), "Launch identity differs")
            require(receipt["modal_function_call_id"] not in [r["modal_function_call_id"] for r in receipts], "Duplicate function call")
            receipts.append(receipt)
            envelope = spec.read(launch / "returned/manifest.json")
            evidence = launch / "returned/evidence"
            entries = {r["path"]: r for r in envelope["files"]}
            files = {p.relative_to(evidence).as_posix(): p for p in evidence.rglob("*")
                     if p.is_file() and p.name != "result-manifest.json"}
            require(envelope["run_id"] == sid and envelope["transport_sha256"] == spec.digest(transport)
                    and len(entries) == len(envelope["files"]) and set(entries) == set(files), "Delivery identity or membership differs")
            for name, path in files.items():
                require(not path.is_symlink() and path.stat().st_size == entries[name]["size_bytes"]
                        and spec.file_sha(path) == entries[name]["sha256"], "Changed delivered file: " + name)
            total_files += len(files)
            terminal = spec.read(evidence / "terminal.json")
            require(terminal == envelope["terminal"] and terminal["run_id"] == sid
                    and terminal["plan_sha256"] == adapter.PLAN_SHA
                    and terminal["status"] == "EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED"
                    and terminal["process"]["returncode"] == 0 and terminal["process"]["timed_out"] is False,
                    "Failed or incomplete outer execution; preserve diagnostics, no result admission")
            attempt = evidence / "attempt"
            req, prepared = adapter.session_inputs(plan, source_request, source_plan, sid)
            approval = spec.read(attempt / "approval.json")
            adapter.check_approval(plan, approval, spec.digest(transport))
            require(prepared == spec.read(attempt / "approved-plan.json"), "Session inputs changed")
            result = spec.read(attempt / "result.json")
            expected_session = {"model_load_attempts": 1, "attempted_trials": 36, "generation_attempts": 36,
                "generation_attempts_lower_bound": 36, "generation_count_verified": True, "completed_trials": 36,
                "failed": False, "automatic_retry": False, "concurrent_generation": False,
                "reuse_policy": "same_admitted_weights_new_pipeline_and_dynamic_cache_per_trial"}
            require(result["session"] == expected_session and result["session_id"] == sid
                    and result["plan_sha256"] == adapter.PLAN_SHA and result["source_request_sha256"] == spec.SOURCE_REQUEST
                    and result["attempt_count"] == 36 and result["status"] == terminal["status"], "Model/session counts differ")
            process = spec.read(attempt / "process.json")
            require(process["session_id"] == sid and process["plan_sha256"] == adapter.PLAN_SHA
                    and type(process["pid"]) is int and process["pid"] > 0, "Worker identity differs")
            runtime = spec.read(attempt / "shared_model/runtime.json")
            verify_runtime_preflight(spec.read(attempt / "shared_model/runtime_preflight.json"), plan["runtime_reference"], runtime)
            loaded = spec.read(attempt / "shared_model/loaded_model.json")
            require(loaded["weights"]["entries_sha256"] == req["qualification"]["model_weight_entries_sha256"], "Loaded weights differ")
            execution = attempt / "execution"
            outer = spec.read(execution / "outer.json")
            expected_ids = [r["trial_id"] for r in prepared["trials"]]
            require(outer["status"] == "completed" and outer["run_id"] == sid and outer["trial_ids"] == expected_ids
                    and outer["attempt_count"] == 36 and outer["retry_count"] == 0 and len(outer["records"]) == 36
                    and {p.name for p in (execution / "trials").iterdir()} == set(expected_ids), "Extra, missing or failed attempts")
            with adapter.session_connection(route):
                phase = route.phase_plan(req, prepared)
            verify = route.mechanical_verifier(req, directions)
            for index, (trial, source, link) in enumerate(zip(phase["trials"], prepared["trials"], outer["records"], strict=True)):
                location = execution / "trials" / trial["trial_id"]
                require(link["trial_id"] == trial["trial_id"] and link["directory"] == location.relative_to(execution).as_posix()
                        and link["state_sha256"] == spec.file_sha(location / "state_execution.json")
                        and spec.read(location / "verified.json") == link, "Execution link differs")
                require(spec.read(location / "attempt.json") == {"trial_id": trial["trial_id"], "index": index,
                    "original_plan_sha256": spec.file_sha(attempt / "approved-plan.json"), "retry_count": 0}, "Attempt changed")
                log = location / link["source_log_path"]
                require(log.resolve().is_relative_to(location.resolve()) and spec.file_sha(log) == link["source_log_sha256"]
                        and list((location / "inspect").glob("*.json")) == [log], "Missing or extra response log")
                state = spec.read(location / "state_execution.json")["records"][0]
                state = verify(trial, source, location, {"generated_token_ids": state["generated_token_ids"]})
                ids = tokenizer.apply_chat_template(trial["messages"], tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False)
                require(ids == source["execution_input"]["input_ids"]
                        and spec.read(location / "input.json") == {"input_token_ids": ids,
                            "input_token_ids_sha256": spec.digest(tuple(ids)), "messages_sha256": trial["messages_sha256"]}, "Actual input differs")
                require(spec.read(location / "loaded_model.json") == loaded, "Per-trial weights differ")
                use = {"trial_id": trial["trial_id"], "attempt_index": index + 1, "load_index": 1,
                    "reused_weights": index > 0, "load_record": "shared_model/loaded_model.json",
                    "load_record_sha256": spec.file_sha(attempt / "shared_model/loaded_model.json"),
                    "fresh_dynamic_cache_per_generation": True, "past_key_values_supplied": False, "concurrent_generation": False}
                require(spec.read(location / "model_use.json") == use, "Model reuse or cache policy differs")
                counts = {**expected_session, **{key: index + 1 for key in
                    ("attempted_trials", "generation_attempts", "generation_attempts_lower_bound", "completed_trials")}}
                require(spec.read(location / "model_session.json") == counts, "Cumulative attempts differ")
                document = binding._read_exact_source(log, spec.file_sha(log))
                kind, samples = binding._source_index(document, False)
                require(kind == "inspect_json_v2" and set(samples) == {binding._id(source["source_sample_id"])}, "Incomplete original Inspect record")
                sample = samples[binding._id(source["source_sample_id"])]
                binding._settings_match(sample, source)
                require(sample["metadata"]["execution_input"] == source["execution_input"]
                        and sample["metadata"]["original_plan_sha256"] == spec.file_sha(attempt / "approved-plan.json")
                        and validate_messages(sample["input"], inspect_serialized=True) == trial["messages"], "Logged messages/metadata changed")
                expected_metadata = {"run_id": sid, "phase": "repeatability", "plan_sha256": spec.digest(phase),
                    "fixture_only": False, "model_id": phase["model_id"], "model_revision": phase["model_revision"],
                    **{key: trial[key] for key in ("trial_id", "case_id", "family_id", "arm_id", "kind", "edit",
                                                  "messages_sha256", "case_content_sha256", "output_contract_version")}}
                require(sample["metadata"]["preparatory_trial"] == expected_metadata, "Generation metadata changed")
                status, raw, partial = binding._response(document, sample, kind, source)
                saved = read_eval_log(log)
                require(status == "COMPLETED" and not partial and saved.status == "success" and saved.samples[0].error is None
                        and raw == saved.samples[0].output.completion == tokenizer.decode(state["generated_token_ids"], skip_special_tokens=True),
                        "Text does not match generated tokens")
                score = add_agreement_outcomes(score_completed(legacy.scorer(), cases[trial["case_id"]], raw), cases[trial["case_id"]])
                old_path = spec.EXECUTION / "launch/returned/evidence/attempt/execution/trials" / source["original_trial_id"] / "state_execution.json"
                old_state = spec.read(old_path)["records"][0]
                rows.append({"trial_id": trial["trial_id"], "session_id": sid, "original_trial_id": source["original_trial_id"],
                    "raw_response": raw, "generated_token_ids": state["generated_token_ids"], "score": score,
                    "processed_logits_match": state["processed_logit_steps"] == old_state["processed_logit_steps"],
                    "source_log_sha256": spec.file_sha(log), "state_execution_sha256": spec.file_sha(location / "state_execution.json")})
    result = comparisons(rows, reference, plan)
    output.mkdir(parents=True, exist_ok=False)
    spec.save_new(output / "outcomes.json", rows)
    spec.save_new(output / "comparison.json", result)
    report = {"status": "REPEATABILITY_INDEPENDENTLY_VERIFIED", "plan_sha256": adapter.PLAN_SHA,
        "transport_sha256": spec.digest(transport), "auditor_sha256": spec.file_sha(Path(__file__)),
        "delivered_files": total_files, "verified_new_responses": len(rows), "sessions": 2,
        "model_loads_during_audit": 0, "provider_calls_during_audit": 0,
        **{key: value for key, value in result.items() if key not in ("outputs", "contrasts")}}
    spec.save_new(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    print(spec.canonical(audit(args.directory, args.output, args.snapshot)))
