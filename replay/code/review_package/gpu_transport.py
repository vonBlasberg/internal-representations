"""Explicit check/launch/retrieve actions. Checking never imports Modal or submits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
from unittest.mock import patch
from .reuse import PACKAGE, legacy
from .admission import identity
from .recording import save_new
from .gpu_connection import LIMITS, RUN_ID, approve_plan, read, runtime


def check_payloads(transport, payloads):
    if (transport['artifact_type'] != 'FIXED_COMPARISON_TRANSPORT' or transport['run_id'] != RUN_ID
            or transport['limits'] != LIMITS or transport['base_image_id'] != 'im-T0JhRRVpTMVV3J7CIpVhsZ'
            or identity(transport['files']) != transport['inventory_sha256']):
        raise ValueError('Transport settings changed')
    entries = {e['path']: e for e in transport['files']}
    if len(entries) != len(transport['files']) or set(entries) != set(payloads):
        raise ValueError('Transport file set differs')
    for name, raw in payloads.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise ValueError('Unsafe source path')
        if len(raw) != entries[name]['size_bytes'] or hashlib.sha256(raw).hexdigest() != entries[name]['sha256']:
            raise ValueError('Source changed: ' + name)
    request = json.loads(payloads['request/research.json'])
    prepared = json.loads(payloads['request/prepared-plan.json'])
    if (identity(request) != transport['request_sha256'] or request['limits'] != LIMITS
            or request['run_id'] != RUN_ID or identity(prepared) != request['prepared_plan_sha256']):
        raise ValueError('Request or prepared inputs differ')
    if len(prepared['trials']) != 324 or len({t['trial_id'] for t in prepared['trials']}) != 324:
        raise ValueError('Expected exactly 324 unique trials')
    return request, prepared


def load(directory):
    directory = Path(directory)
    if legacy.sha(directory / 'source_payload.zip') != read(directory / 'source_archive.json')['sha256']:
        raise ValueError('Source archive changed')
    with zipfile.ZipFile(directory / 'source_payload.zip') as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate archive entry')
        payloads = {name: z.read(name) for name in z.namelist()}
    transport = read(directory / 'transport.json')
    request, prepared = check_payloads(transport, payloads)
    if request != read(directory / 'request.json') or prepared != read(directory / 'prepared-plan.json'):
        raise ValueError('Readable request differs from executable archive')
    for name, raw in payloads.items():
        if name.startswith(('code/', 'protocol/')) and (PACKAGE / name).read_bytes() != raw:
            raise ValueError('Prepared package code changed: ' + name)
    return request, prepared, transport, payloads


def unpack(payloads, root):
    root = Path(root)
    for name, raw in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(raw)


def remote(approval, transport, payloads, claim, commit):
    request, prepared = check_payloads(transport, payloads)
    approve_plan(request, prepared, approval, identity(transport))
    delivery = runtime(Path('/opt/comparison-build'))
    if not claim(RUN_ID, {'request_sha256': identity(request)}):
        return {'status': 'ALREADY_CLAIMED_NO_EXECUTION', 'run_id': RUN_ID, 'files': {}}
    outer = Path(delivery.MOUNT) / delivery.result_prefix(RUN_ID)
    outer.mkdir(parents=True, exist_ok=False)
    save_new(outer / 'claimed.json', {'run_id': RUN_ID, 'request_sha256': identity(request)})
    commit()
    try:
        root = Path('/tmp/fixed-comparison-execution')
        unpack(payloads, root)
        save_new(root / 'request/approval.json', approval)
        with patch.dict(os.environ, {'PYTHONPATH': str(root / 'code') + os.pathsep + str(root)}):
            process = delivery.run_process([sys.executable, '-B', '-m', 'review_package.gpu_connection',
                str(root), str(root / 'request/approval.json'), delivery.SNAPSHOT, str(outer / 'attempt'), identity(transport)],
                root, outer / 'stdout.log', outer / 'stderr.log', LIMITS['worker_seconds'])
        result = read(outer / 'attempt/result.json') if (outer / 'attempt/result.json').is_file() else {}
        status = ('EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED' if process['returncode'] == 0
                  and not process['timed_out'] and result.get('status') == 'EXECUTION_COMPLETE_NOT_INDEPENDENTLY_VERIFIED'
                  else 'FAILED_OR_INDETERMINATE_NO_RETRY')
        terminal = {'run_id': RUN_ID, 'status': status, 'request_sha256': identity(request),
            'process': process, 'automatic_retry': False}
    except BaseException as error:
        terminal = {'run_id': RUN_ID, 'status': 'FAILED_NO_RETRY', 'request_sha256': identity(request),
            'error_type': type(error).__name__, 'error': str(error), 'automatic_retry': False}
    save_new(outer / 'terminal.json', terminal)
    with patch.object(delivery, 'MAX_EVIDENCE_TOTAL_BYTES', LIMITS['evidence_bytes']):
        return delivery.publish_result(outer, terminal, transport, commit)


def driver(approval, transport, payloads):
    request, prepared = check_payloads(transport, payloads)
    approve_plan(request, prepared, approval, identity(transport))
    import modal
    stage = Path(tempfile.mkdtemp(prefix='fixed-comparison-image-'))
    unpack(payloads, stage)
    delivery = runtime(stage)
    requirements = [line.strip() for line in payloads[delivery.LOCK].decode().splitlines()
        if line.strip() and not line.startswith(('#', './upstream'))]
    image = (modal.Image.from_id(delivery.BASE_IMAGE).pip_install(requirements)
        .add_local_dir(str(stage), '/opt/comparison-build', copy=True)
        .pip_install('/opt/comparison-build/' + delivery.UPSTREAM, extra_options='--no-deps'))
    volume = modal.Volume.from_name(delivery.VOLUME, create_if_missing=False)
    app = modal.App('orgsci-fixed-negotiation-transfer-a03', include_source=False)
    @app.function(image=image, gpu='L40S', cpu=(8.0, 8.0), memory=(65536, 65536),
        volumes={delivery.MOUNT: volume}, timeout=LIMITS['container_seconds'], retries=0,
        max_containers=1, max_inputs=1, block_network=True, serialized=True, include_source=False,
        env={'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
             'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    def execute(approval, transport, payloads):
        import sys
        import modal
        sys.path.insert(0, '/opt/comparison-build/code')
        from review_package import gpu_transport as run
        claims = modal.Dict.from_name('orgsci-fixed-transfer-one-use', create_if_missing=True)
        volume = modal.Volume.from_name('wp3-qwen35-l40s-modal-01-data', create_if_missing=False)
        return run.remote(approval, transport, payloads,
            lambda key, value: claims.put(key, value, skip_if_exists=True), volume.commit)
    return modal, app, execute


def launch(directory, approval_path):
    request, prepared, transport, payloads = load(directory)
    approval = read(approval_path)
    approve_plan(request, prepared, approval, identity(transport))
    # Claim BEFORE any provider contact, including image building. Never retry a failed submission.
    launch_path = Path(directory) / 'launch'
    launch_path.mkdir(exist_ok=False)
    save_new(launch_path / 'submission-claim.json', {'run_id': RUN_ID, 'request_sha256': identity(request)})
    save_new(launch_path / 'approval.json', approval)
    try:
        modal, app, function = driver(approval, transport, payloads)
        with modal.enable_output(), app.run(detach=True):
            call = function.spawn(approval, transport, payloads)
            receipt = {'artifact_type': 'PACKAGE_QUALIFICATION_DETACHED_LAUNCH', 'run_id': RUN_ID,
                'modal_app_id': app.app_id, 'modal_function_call_id': call.object_id,
                'transport_sha256': identity(transport), 'status': 'SUBMITTED_NOT_VERIFIED'}
            save_new(launch_path / 'launch.json', receipt)
            return receipt
    except BaseException as error:
        save_new(launch_path / 'submission-error.json', {'error_type': type(error).__name__,
            'error': str(error), 'status': 'ERROR_OR_INDETERMINATE_NO_RETRY'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'launch', 'retrieve'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--approval', type=Path)
    args = parser.parse_args()
    request, prepared, transport, payloads = load(args.prepared)
    if args.action == 'check':
        result = {'status': 'SOURCE_AND_INPUTS_CHECKED_NOT_AUTHORIZED', 'request_sha256': identity(request),
            'transport_sha256': identity(transport), 'trials': len(prepared['trials']), 'provider_calls': 0}
    elif args.action == 'launch':
        if not args.approval:
            raise PermissionError('Explicit approval file is required')
        result = launch(args.prepared, args.approval)
    else:
        with tempfile.TemporaryDirectory(prefix='fixed-transfer-retrieve-') as temporary:
            unpack(payloads, temporary)
            delivery = runtime(temporary)
            with patch.object(delivery, 'MAX_EVIDENCE_TOTAL_BYTES', LIMITS['evidence_bytes']):
                result = delivery.retrieve(args.prepared / 'launch')
    print(legacy.canonical(result))


if __name__ == '__main__':
    main()
