"""Fixed comparison connection. Imports do not load a model or contact a provider."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from .reuse import legacy, binding, protocol
from .admission import identity
from .recording import save_new, inspect_sample, collect_saved_inspect
from .case_validation import require_prompt_review
from .runtime_preflight import MEMORY_POLICY

RUN_ID = 'negotiation-transfer-2026-10-31-A03'
LIMITS = {'maximum_generation_calls': 324, 'maximum_model_loads': 1,
    'worker_seconds': 28200, 'container_seconds': 28800, 'maximum_reserved_usd': 50.0,
    'gpu': 'L40S', 'cpu': 8, 'memory_mib': 65536, 'retries': 0,
    'evidence_bytes': 16 * 1024**3}


def read(path):
    return json.loads(Path(path).read_text())


def runtime(root):
    """Use the complete preserved runtime, not the small offline analysis copy."""
    # Both trees contain the identical scorer. Prime the pinned offline copy so
    # runtime imports cannot displace it and invalidate later source checks.
    legacy.scorer()
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    import local_integration
    local_integration.__path__ = [str(root / 'local_integration')]
    for name, module in tuple(sys.modules.items()):
        if name.startswith('local_integration.') and getattr(module, '__file__', None):
            wanted = root / (name.replace('.', '/') + '.py')
            if wanted.is_file() and Path(module.__file__).read_bytes() != wanted.read_bytes():
                raise RuntimeError('A different runtime module is already loaded: ' + name)
    return importlib.import_module('local_integration.qualification_transport')


def approve_plan(request, prepared, approval, transport_hash):
    """Only authorization metadata can change after the inputs are prepared."""
    if (request['run_id'] != RUN_ID or request['limits'] != LIMITS
            or request.get('runtime_memory_policy') != MEMORY_POLICY
            or approval.get('owner_authorized') is not True
            or approval.get('request_sha256') != identity(request)
            or approval.get('transport_sha256') != transport_hash
            or approval.get('limits') != LIMITS
            or not isinstance(approval.get('owner_authorization_reference'), str)
            or not approval['owner_authorization_reference'].strip()
            or approval.get('automatic_retry') is not False
            or approval.get('deployment_authorized') is not False):
        raise PermissionError('The exact one-use request needs explicit owner approval')
    if identity(prepared) != request['prepared_plan_sha256'] or prepared.get('preparation_only') is not True:
        raise ValueError('Prepared inputs changed')
    approved = deepcopy(prepared)
    approved.pop('preparation_only')
    review = approved['prompt_review']
    review.update(status='APPROVED', reviewer=approval.get('reviewer'), checks=approval.get('input_review_checks'))
    require_prompt_review(review, request['cases'], protocol()[0])
    return approved


def phase_plan(request, original):
    cases = {c['case_id']: c for c in request['cases']}
    rows = []
    for row in original['trials']:
        edit = None if row['site'] is None else {'layer': 14, 'module_path': row['site'],
            'coefficient': row['coefficient'], 'direction_sha256': row['direction_sha256']}
        messages = row['execution_input']['messages']
        rows.append({'trial_id': row['trial_id'], 'case_id': row['case_id'],
            'family_id': cases[row['case_id']]['family_id'], 'arm_id': row['condition'],
            'kind': row['condition'], 'edit': edit, 'messages': messages,
            'messages_sha256': identity(messages), 'case_content_sha256': cases[row['case_id']]['case_content_sha256'],
            'output_contract_version': 2})
    return {'run_id': request['run_id'], 'phase': 'untouched', 'fixture_only': original['fixture_only'],
        'model_id': request['manifest']['model_id'], 'model_revision': request['manifest']['model_revision'],
        'generation_contract': request['qualification']['generation_contract'], 'trials': rows}


@contextmanager
def attach_input_record(original, pin):
    """Keep the retained Inspect generator and add prospective metadata before generation."""
    from local_integration import preparatory_inspect as bridge
    sample_type = bridge.Sample
    def sample(**kwargs):
        expected = inspect_sample(original, pin)
        actual_messages = [{'role': m.role, 'content': m.content} for m in kwargs['input']]
        if kwargs['id'] != expected.id or actual_messages != original['execution_input']['messages']:
            raise ValueError('Runtime input differs from the prepared original')
        return sample_type(**{**kwargs, 'metadata': {**kwargs['metadata'], **expected.metadata}})
    with patch.object(bridge, 'Sample', sample):
        yield


def execute_sequence(request, original, plan_path, destination, generator, verify, *, clock=time.monotonic):
    """Shared real/synthetic loop: exact order, no retries, preserve every prefix on failure."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    pin = legacy.sha(plan_path)
    if read(plan_path) != original or len(original['trials']) != LIMITS['maximum_generation_calls']:
        raise ValueError('Original input inventory differs')
    phase = phase_plan(request, original)
    started, deliveries, records = clock(), {}, []
    attempted = []
    baselines = {}
    try:
        for trial, source in zip(phase['trials'], original['trials'], strict=True):
            if clock() - started >= LIMITS['worker_seconds']:
                raise TimeoutError('Worker wall-time limit; no next attempt')
            directory = destination / 'trials' / trial['trial_id']
            directory.mkdir(parents=True, exist_ok=False)
            save_new(directory / 'attempt.json', {'trial_id': trial['trial_id'], 'index': len(attempted),
                'original_plan_sha256': pin, 'retry_count': 0})
            attempted.append(trial['trial_id'])
            with attach_input_record(source, pin):
                output = generator(trial, directory)
            record = verify(trial, source, directory, output)
            # Zero edits must reproduce the same case's baseline token/logit trace.
            trace = {k: record[k] for k in ('generated_token_ids', 'processed_logit_steps')}
            if trial['arm_id'] == 'unchanged':
                baselines[trial['case_id']] = trace
            if trial['arm_id'] == 'sham' and trace != baselines[trial['case_id']]:
                raise RuntimeError('Zero-edit token or processed-logit trace differs from baseline')
            log = directory / output['source_log_path']
            if legacy.sha(log) != output['source_log_sha256']:
                raise ValueError('Original Inspect log changed')
            deliveries[trial['trial_id']] = log
            records.append({'trial_id': trial['trial_id'], 'directory': str(directory.relative_to(destination)),
                'state_sha256': legacy.sha(directory / 'state_execution.json'),
                'source_log_path': output['source_log_path'], 'source_log_sha256': legacy.sha(log)})
            save_new(directory / 'verified.json', records[-1])
        result = {'status': 'completed', 'run_id': request['run_id'], 'attempt_count': len(attempted),
            'trial_ids': attempted, 'retry_count': 0, 'fixture_only': original['fixture_only'], 'records': records}
    except BaseException as error:
        result = {'status': 'FAILED_NO_RETRY', 'run_id': request['run_id'], 'attempt_count': len(attempted),
            'trial_ids': attempted, 'verified_count': len(records), 'records': records,
            'retry_count': 0, 'fixture_only': original['fixture_only'],
            'error_type': type(error).__name__, 'error': str(error)}
        save_new(destination / 'outer.json', result)
        raise
    save_new(destination / 'outer.json', result)
    return result, deliveries


def load_directions(root, request):
    import numpy as np
    import torch
    from worker.receipts import tensor_sha256
    result = {}
    for tensor_hash, entry in request['directions'].items():
        path = Path(root) / entry['path']
        if legacy.sha(path) != entry['file_sha256']:
            raise ValueError('Direction file changed')
        tensor = torch.from_numpy(np.frombuffer(path.read_bytes(), dtype='<f4').copy()).reshape(1, 4096)
        if tensor_sha256(tensor) != tensor_hash or tensor.shape != (1, 4096):
            raise ValueError('Direction tensor changed')
        result[tensor_hash] = tensor
    return result


def mechanical_verifier(request, directions):
    from local_integration.package_qualification import validate_plan
    from local_integration.depth_calibration_audit import audit_record
    contract = validate_plan(request['qualification'])
    def verify(trial, original, directory, output):
        records = read(directory / 'state_execution.json')['records']
        if len(records) != 1:
            raise ValueError('Expected one state execution record')
        record = records[0]
        edit = trial['edit']
        direction = directions[edit['direction_sha256']] if edit else None
        audit_record(record, trial, contract, request['qualification']['model_weight_entries_sha256'],
            original['execution_input']['input_ids'], direction)
        if record['cache_state']['field_coverage']['complete'] is not True:
            raise ValueError('Incomplete cache-field coverage')
        if output['generated_token_ids'] != record['generated_token_ids']:
            raise ValueError('Response token receipt differs')
        return record
    return verify


def run_worker(root, approval, snapshot, destination, transport_hash):
    """Only called by the separately approved transport. Never during preparation."""
    root, destination = Path(root), Path(destination)
    runtime(root)
    request, prepared = read(root / 'request/research.json'), read(root / 'request/prepared-plan.json')
    original = approve_plan(request, prepared, approval, transport_hash)
    from local_integration.package_qualification import validate_plan
    from local_integration import qualification_worker
    from local_integration.reused_model import SequentialModelSession, ReusedInspectGenerator
    from local_integration.calibration_diagnostics import calibration_diagnostic_connection
    from local_integration.generation_sensitivity import verify_tokenizer_files
    from local_integration import research_pipeline
    from .cache_coverage import guarded_cache_summary
    from .runtime_preflight import load_with_runtime_record
    from transformers import AutoTokenizer
    contract = validate_plan(request['qualification'])
    verify_tokenizer_files(request['qualification'], snapshot, root)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    for row in original['trials']:
        ids = tokenizer.apply_chat_template(row['execution_input']['messages'], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=False)
        if ids != row['execution_input']['input_ids'] or not 0 < len(ids) <= contract.max_input_tokens:
            raise ValueError('Prepared real tokens differ; no model load')
    destination.mkdir(parents=True, exist_ok=False)
    save_new(destination / 'approved-plan.json', original)
    save_new(destination / 'approval.json', approval)
    directions = load_directions(root, request)
    def loader(directory):
        model, tokens, unused = load_with_runtime_record(
            request['qualification'], snapshot, directory, qualification_worker)
        del unused
        return model, tokens
    session = SequentialModelSession(destination / 'shared_model', loader,
        expected_weights_sha256=request['qualification']['model_weight_entries_sha256'], contract=contract)
    generator = ReusedInspectGenerator(phase_plan(request, original), contract, model_session=session,
        direction_loader=lambda edit, _: directions[edit['direction_sha256']])
    recorder = research_pipeline.summarize_cache
    def guarded(cache, **kwargs):
        return guarded_cache_summary(recorder, cache, **kwargs)
    with calibration_diagnostic_connection(), patch.object(research_pipeline, 'summarize_cache', guarded):
        outer, _ = execute_sequence(request, original, destination / 'approved-plan.json', destination / 'execution',
            generator, mechanical_verifier(request, directions))
    if session.failed or session.loads != 1 or session.generation_attempts != 324 or session.completed_trials != 324:
        raise RuntimeError('Model session count differs')
    save_new(destination / 'result.json', {'status': 'EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED',
        'request_sha256': identity(request), 'session': session.summary(), 'attempt_count': outer['attempt_count']})


if __name__ == '__main__':
    root, approval, snapshot, destination, transport_hash = sys.argv[1:]
    run_worker(root, read(approval), snapshot, destination, transport_hash)
