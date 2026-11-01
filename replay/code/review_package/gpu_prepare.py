"""Prepare the fixed GPU request and real token IDs locally. Never loads weights."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from .reuse import PACKAGE, legacy, binding, protocol
from .admission import identity
from .case_validation import make_prompt_review
from .recording import prepare_plan, save_new
from .gpu_connection import LIMITS, RUN_ID, runtime, read
from .runtime_preflight import MEMORY_POLICY

A10_ARCHIVE = '9133de1df9753f6048fbe4c653ee839e2212de0f64712612b0713f33ad17ffbc'
CASE_FILE = 'preparation/outcome-amendment-2026-10-31/draft-cases.json'
CASE_HASH = 'cf9f0c52c6fe140b7743989aab758e12da3bf2ee0d76082710924dfce90f1a06'


def parts(cases, *, fixture_only=False):
    spec, digest = protocol()
    manifest = {'artifact_type': 'SAVED_BEHAVIORAL_ANALYSIS_INPUT', 'schema_version': 1,
        'run_id': RUN_ID if not fixture_only else RUN_ID + '-SYNTHETIC-ONLY', 'phase': 'untouched',
        'fixture_only': fixture_only, 'model_id': spec['model_id'], 'model_revision': spec['model_revision'],
        'protocol_sha256': digest, 'prompt_text': spec['prompt_text'],
        'manifest_sha256': identity({'cases': cases, 'protocol': spec, 'run_id': RUN_ID, 'fixture_only': fixture_only}),
        'analysis_contract_sha256': legacy.PINS['study/manifests/behavioral_analysis_contract.json'],
        'planned_trials': [], 'files': {}}
    trials = []
    for case in cases:
        for condition, settings in spec['conditions'].items():
            row = {**{k: manifest[k] for k in ('run_id','phase','fixture_only','model_id','model_revision','manifest_sha256')},
                'trial_id': case['case_id'] + '--' + condition, 'case_id': case['case_id'], 'condition': condition, **settings}
            trials.append(row)
            manifest['planned_trials'].append({k: row[k] for k in ('trial_id','case_id','condition')})
    return manifest, trials


def prepare(destination, research_root):
    destination, research_root = Path(destination).resolve(), Path(research_root).resolve()
    if destination.exists():
        raise FileExistsError('A prepared request is immutable; choose a new folder')
    source = research_root / 'records/expanded_construction_2026-10-31_A10/calibration'
    archive = source / 'source_payload.zip'
    if legacy.sha(archive) != A10_ARCHIVE or legacy.sha(PACKAGE / CASE_FILE) != CASE_HASH:
        raise ValueError('A10 source or approved draft case bytes changed')
    entries = {r['path']: r for r in read(source / 'transport.json')['files']}
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(entries):
            raise ValueError('A10 archive inventory differs')
        payloads = {name: z.read(name) for name in z.namelist()}
    for name, raw in payloads.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or hashlib.sha256(raw).hexdigest() != entries[name]['sha256']:
            raise ValueError('Unsafe or changed A10 source: ' + name)
        if not name.startswith('request/') and (research_root / name).read_bytes() != raw:
            raise ValueError('Canonical retained runtime changed: ' + name)
    payloads = {name: raw for name, raw in payloads.items() if not name.startswith('request/')}
    qualification = read(source / 'request.json')['program']['qualification_plan']
    cases = read(PACKAGE / CASE_FILE)
    manifest, trials = parts(cases)
    manifest['case_bank_freeze_sha256'] = CASE_HASH
    manifest['prior_exposure_audit_sha256'] = legacy.sha(PACKAGE / 'preparation/outcome-amendment-2026-10-31/declared-prior-input-check.json')
    # Import the complete retained implementation, not an incomplete offline copy.
    runtime(research_root)
    from local_integration.generation_sensitivity import verify_tokenizer_files
    snapshot = research_root / 'negotiation-steering/snapshots/qwen35-9b' / manifest['model_revision'] / 'verified-snapshot'
    token_hashes = verify_tokenizer_files(qualification, snapshot, research_root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = 'left'
    def tokenize(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False)
        if not 0 < len(ids) <= qualification['generation_contract']['max_input_tokens']:
            raise ValueError('Input exceeds unchanged allowance; do not truncate')
        return ids
    plan = prepare_plan(manifest, cases, trials, tokenize=tokenize,
        tokenizer_record={'model_id': manifest['model_id'], 'revision': manifest['model_revision'],
            'file_sha256': token_hashes, 'add_generation_prompt': True, 'enable_thinking': False,
            'padding_side': 'left', 'truncation': False},
        prompt_review=make_prompt_review(cases, protocol()[0]), for_review=True)
    directions = {}
    historical = read(source / 'request.json')['directions']
    for wanted in {row['direction_sha256'] for row in trials if row['direction_sha256']}:
        matches = [e for values in historical.values() for e in values.values() if e['sha256'] == wanted]
        if len(matches) != 1:
            raise ValueError('Direction is not uniquely bound to A10')
        entry = matches[0]
        original = research_root / 'records/expanded_construction_2026-10-31_A10/construction/launch/returned/evidence/attempt/capture/directions' / entry['path']
        raw = original.read_bytes()
        if hashlib.sha256(raw).hexdigest() != wanted or len(raw) != entry['size_bytes']:
            raise ValueError('A10 direction differs')
        name = 'request/directions/' + entry['path']
        payloads[name] = raw
        directions[wanted] = {'path': name, 'file_sha256': wanted, 'size_bytes': len(raw), 'shape': entry['shape']}
    request = {'artifact_type': 'FIXED_324_TRIAL_GPU_REQUEST', 'run_id': RUN_ID, 'limits': dict(LIMITS),
        'runtime_memory_policy': MEMORY_POLICY,
        'manifest': manifest, 'cases': cases, 'prepared_plan_sha256': identity(plan),
        'qualification': qualification, 'directions': directions,
        'source_A10_archive_sha256': A10_ARCHIVE, 'case_file_sha256': CASE_HASH,
        'candidate_reselection': False, 'direction_construction': False, 'checkpoint_download': False,
        'deployment': False, 'existing_goal_budget_reused': False,
        'cost': {'source': 'https://modal.com/pricing', 'checked_date': '2026-10-31',
            'usd_per_second': {'L40S': 0.000542, 'cpu_core': 0.0000131, 'memory_gib': 0.00000222},
            'eight_hour_compute_usd': round(28800 * (0.000542 + 8*0.0000131 + 64*0.00000222), 4),
            'storage_egress_build_reserve_usd': 20.0,
            'scope': 'Resource/time caps bound compute; not a provider-enforced invoice cap. No new subscription or region premium.'}}
    # Keep the runtime source tree intact; add only the prospective adapter/analysis.
    for base in (PACKAGE / 'code', PACKAGE / 'protocol'):
        for path in sorted(base.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                payloads[str(path.relative_to(PACKAGE))] = path.read_bytes()
    payloads['request/research.json'] = (legacy.canonical(request) + '\n').encode()
    payloads['request/prepared-plan.json'] = (legacy.canonical(plan) + '\n').encode()
    payloads['request/prior-input-check.json'] = (PACKAGE / 'preparation/outcome-amendment-2026-10-31/declared-prior-input-check.json').read_bytes()
    inventory = [{'path': name, 'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                 for name, raw in sorted(payloads.items())]
    transport = {'artifact_type': 'FIXED_COMPARISON_TRANSPORT', 'run_id': RUN_ID,
        'request_sha256': identity(request), 'files': inventory, 'inventory_sha256': identity(inventory),
        'limits': dict(LIMITS), 'base_image_id': 'im-T0JhRRVpTMVV3J7CIpVhsZ'}
    approval = {'owner_authorized': False, 'request_sha256': identity(request), 'transport_sha256': identity(transport),
        'limits': dict(LIMITS), 'reviewer': None, 'input_review_checks': {
            'no_private_scoring_information': False, 'variant_meaning_preserved': False},
        'owner_authorization_reference': None, 'automatic_retry': False, 'deployment_authorized': False}
    destination.mkdir(parents=True)
    for name, value in [('request.json', request), ('prepared-plan.json', plan), ('transport.json', transport),
                        ('approval.NOT_AUTHORIZED.json', approval)]:
        save_new(destination / name, value)
    with zipfile.ZipFile(destination / 'source_payload.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(payloads.items()):
            z.writestr(name, raw)
    save_new(destination / 'source_archive.json', {'sha256': legacy.sha(destination / 'source_payload.zip')})
    result = {'status': 'PREPARED_NOT_AUTHORIZED', 'request_sha256': identity(request),
        'transport_sha256': identity(transport), 'source_archive_sha256': legacy.sha(destination / 'source_payload.zip'),
        'trials': len(plan['trials']), 'cases': len(cases), 'families': len({c['family_id'] for c in cases}),
        'input_token_range': [min(len(r['execution_input']['input_ids']) for r in plan['trials']),
                              max(len(r['execution_input']['input_ids']) for r in plan['trials'])],
        'source_files': len(payloads), 'model_loads': 0, 'provider_calls': 0, 'paid_run_started': False}
    save_new(destination / 'preparation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--research-root', type=Path, required=True)
    args = parser.parse_args()
    print(legacy.canonical(prepare(args.output, args.research_root)))
