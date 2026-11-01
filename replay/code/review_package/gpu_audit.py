"""Independently replay delivered originals. No research model or provider access."""
import argparse
import math
import tempfile
from pathlib import Path
from .reuse import legacy
from .admission import identity
from .recording import save_new, collect_saved_inspect
from .gpu_transport import load, unpack
from .gpu_connection import read, runtime, approve_plan, phase_plan, load_directions, mechanical_verifier
from .comparison import analyze_saved
from .cli import write_new


def verify_runtime_preflight(preflight, reference, actual_runtime):
    if (preflight['status'] != 'ADMITTED' or preflight['expected'] != reference
            or preflight.get('memory_policy') != 'minimum_capacity_reference_minus_0.1_GiB'
            or preflight.get('minimum_memory_gib') != reference['actual_gpu']['total_memory_gib'] - 0.1
            or preflight['reference_sha256'] != identity(reference)
            or preflight['differences'] or preflight['observation_errors']
            or preflight['observed']['cuda'] != {'available': True, 'device_count': 1}
            or {k: preflight['observed'][k] for k in actual_runtime} != actual_runtime):
        raise ValueError('Runtime preflight differs from the retained loader record')
    verify_runtime_identity(actual_runtime, reference)


def verify_runtime_identity(actual_runtime, reference):
    """Independently enforce exact identities and the declared capacity floor."""
    if actual_runtime['versions'] != reference['actual_runtime_versions']:
        raise ValueError('Recorded runtime versions differ')
    gpu, expected_gpu = actual_runtime['gpu'], reference['actual_gpu']
    memory = gpu['total_memory_gib']
    if (gpu['name'] != expected_gpu['name'] or gpu['driver_version'] != expected_gpu['driver_version']
            or type(memory) not in (int, float) or not math.isfinite(memory)
            or memory < expected_gpu['total_memory_gib'] - 0.1
            or gpu['bf16_supported'] is not True
            or actual_runtime['optional_kernels'] != reference['optional_kernel_module_visibility']
            or actual_runtime['packages'] != {'steerability':'0.5.7','inspect-ai':'0.3.286'}):
        raise ValueError('GPU, minimum capacity, optional kernels or package versions differ')


def audit(prepared, returned, output, tokenizer):
    request, pending, transport, payloads = load(prepared)
    returned, output = Path(returned), Path(output)
    if output.exists():
        raise FileExistsError('Never replace an audit')
    envelope = read(returned / 'manifest.json')
    returned = returned / 'evidence'
    if envelope['run_id'] != request['run_id'] or envelope['transport_sha256'] != identity(transport):
        raise ValueError('Outer delivery belongs to another request')
    entries = {r['path']: r for r in envelope['files']}
    files = {str(p.relative_to(returned)): p for p in returned.rglob('*') if p.is_file() and p.name != 'result-manifest.json'}
    if len(entries) != len(envelope['files']) or set(files) != set(entries):
        raise ValueError('Incomplete or extra delivery files')
    for name, path in files.items():
        if path.is_symlink() or path.stat().st_size != entries[name]['size_bytes'] or legacy.sha(path) != entries[name]['sha256']:
            raise ValueError('Delivered bytes differ: ' + name)
    terminal = read(returned / 'terminal.json')
    if (terminal != envelope['terminal'] or terminal['status'] != 'EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED'
            or terminal['request_sha256'] != identity(request)
            or terminal['process']['returncode'] != 0 or terminal['process']['timed_out'] is not False):
        raise ValueError('Outer process failed or incomplete; preserve diagnostics, no complete-results analysis')
    attempt = returned / 'attempt'
    original = approve_plan(request, pending, read(attempt / 'approval.json'), identity(transport))
    if original != read(attempt / 'approved-plan.json'):
        raise ValueError('Only approval metadata may differ from prepared inputs')
    result = read(attempt / 'result.json')
    session = result['session']
    if (result['request_sha256'] != identity(request) or result['attempt_count'] != 324 or session['failed']
            or session['model_load_attempts'] != 1 or session['generation_attempts'] != 324
            or session['completed_trials'] != 324 or session['automatic_retry'] is not False):
        raise ValueError('Session counts differ')
    actual_runtime = read(attempt / 'shared_model/runtime.json')
    reference = request['qualification']['runtime_reference']
    preflight = read(attempt / 'shared_model/runtime_preflight.json')
    verify_runtime_preflight(preflight, reference, actual_runtime)
    loaded = read(attempt / 'shared_model/loaded_model.json')
    if loaded['weights']['entries_sha256'] != request['qualification']['model_weight_entries_sha256']:
        raise ValueError('Recorded loaded weights differ')
    execution = attempt / 'execution'
    outer = read(execution / 'outer.json')
    expected_ids = [t['trial_id'] for t in original['trials']]
    if (outer['status'] != 'completed' or outer['trial_ids'] != expected_ids or outer['attempt_count'] != 324
            or outer['retry_count'] != 0 or len(outer['records']) != 324):
        raise ValueError('Outer attempt inventory differs')
    deliveries, baselines = {}, {}
    with tempfile.TemporaryDirectory(prefix='fixed-comparison-audit-') as temporary:
        unpack(payloads, temporary)
        runtime(temporary)
        from inspect_ai.log import read_eval_log
        directions = load_directions(temporary, request)
        verify = mechanical_verifier(request, directions)
        for trial, source, ref in zip(phase_plan(request, original)['trials'], original['trials'], outer['records'], strict=True):
            directory = execution / 'trials' / trial['trial_id']
            if (ref['trial_id'] != trial['trial_id'] or ref['directory'] != str(directory.relative_to(execution))
                    or ref['state_sha256'] != legacy.sha(directory / 'state_execution.json')):
                raise ValueError('State record reference differs')
            log = directory / ref['source_log_path']
            if not log.resolve().is_relative_to(directory.resolve()) or legacy.sha(log) != ref['source_log_sha256']:
                raise ValueError('Source log reference differs')
            record = read(directory / 'state_execution.json')['records'][0]
            record = verify(trial, source, directory, {'generated_token_ids': record['generated_token_ids']})
            ids = tokenizer.apply_chat_template(trial['messages'], tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False)
            if ids != source['execution_input']['input_ids']:
                raise ValueError('Input tokenization differs')
            saved = read_eval_log(log)
            if len(saved.samples or []) != 1 or saved.samples[0].output.completion != tokenizer.decode(record['generated_token_ids'], skip_special_tokens=True):
                raise ValueError('Saved response differs from generated tokens')
            trace = {k: record[k] for k in ('generated_token_ids', 'processed_logit_steps')}
            if trial['arm_id'] == 'unchanged':
                baselines[trial['case_id']] = trace
            if trial['arm_id'] == 'sham' and trace != baselines[trial['case_id']]:
                raise ValueError('Zero edit changed baseline token/logit trace')
            deliveries[trial['trial_id']] = log
    output.mkdir(parents=True)
    manifest = collect_saved_inspect(output / 'delivery', request['manifest'], request['cases'],
        attempt / 'approved-plan.json', legacy.sha(attempt / 'approved-plan.json'), deliveries, outer)
    analysis = analyze_saved(manifest)
    write_new(output / 'report', analysis)
    report = {'status':'FIXED_COMPARISON_INDEPENDENTLY_VERIFIED', 'request_sha256':identity(request),
        'transport_sha256':identity(transport), 'attempts':len(deliveries), 'delivered_files':len(entries),
        'zero_edit_checks':len(baselines), 'population_inference':False, 'model_loads':0,
        'provider_calls':0, 'candidate_reselection':False}
    save_new(output / 'audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--returned', type=Path, required=True)
    parser.add_argument('--tokenizer-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_snapshot), local_files_only=True, trust_remote_code=False)
    print(legacy.canonical(audit(args.prepared, args.returned, args.output, tokenizer)))
