"""Admit the recorded runtime using a minimum GPU-memory capacity requirement."""
from copy import deepcopy
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

from .admission import identity
from .recording import save_new

MEMORY_POLICY = 'minimum_capacity_reference_minus_0.1_GiB'


def observe_runtime():
    import torch
    observed, errors = {}, []

    def read(name, operation):
        try:
            return operation()
        except Exception as error:
            errors.append({'field': name, 'type': type(error).__name__, 'error': str(error)})
            return None

    observed['versions'] = {'python': platform.python_version(), 'pytorch': str(torch.__version__),
        'transformers': read('versions.transformers', lambda: importlib.metadata.version('transformers')),
        'cuda': torch.version.cuda}
    available = read('cuda.available', torch.cuda.is_available)
    count = read('cuda.device_count', torch.cuda.device_count)
    observed['cuda'] = {'available': available, 'device_count': count}
    observed['gpu'] = {}
    if available and count:
        observed['gpu'] = {
            'name': read('gpu.name', lambda: torch.cuda.get_device_name(0)),
            'total_memory_gib': read('gpu.total_memory_gib', lambda: torch.cuda.get_device_properties(0).total_memory / 1024**3),
            'bf16_supported': read('gpu.bf16_supported', torch.cuda.is_bf16_supported),
            'driver_version': read('gpu.driver_version', lambda: subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=30).strip())}
    observed['optional_kernels'] = {name: read('optional_kernels.' + name,
        lambda name=name: importlib.util.find_spec(name) is not None) for name in ('causal_conv1d', 'fla', 'triton')}
    observed['packages'] = {name: read('packages.' + name,
        lambda name=name: importlib.metadata.version(name)) for name in ('steerability', 'inspect-ai')}
    return observed, errors


def differences(observed, reference):
    expected = {'versions': reference['actual_runtime_versions'],
        'cuda': {'available': True, 'device_count': 1},
        'gpu': {**reference['actual_gpu'], 'bf16_supported': True},
        'optional_kernels': reference['optional_kernel_module_visibility'],
        'packages': {'steerability': '0.5.7', 'inspect-ai': '0.3.286'}}
    result = []
    for section, fields in expected.items():
        for name, wanted in fields.items():
            actual = observed.get(section, {}).get(name)
            matches = type(actual) is type(wanted) and actual == wanted
            if (section, name) == ('gpu', 'total_memory_gib'):
                matches = type(actual) in (int, float) and math.isfinite(actual) and actual >= wanted - 0.1
            if not matches:
                result.append({'field': section + '.' + name, 'expected': wanted, 'observed': actual})
    return result


def record_runtime(plan, directory):
    path = Path(directory) / 'runtime_preflight.json'
    if path.exists():
        raise FileExistsError('Preserve the existing runtime observation')
    reference = plan['runtime_reference']
    try:
        observed, errors = observe_runtime()
    except Exception as error:
        observed, errors = {}, [{'field': 'runtime_observation', 'type': type(error).__name__, 'error': str(error)}]
    mismatch = differences(observed, reference)
    report = {'status': 'ADMITTED' if not mismatch and not errors else 'REJECTED',
        'memory_policy': MEMORY_POLICY,
        'minimum_memory_gib': reference['actual_gpu']['total_memory_gib'] - 0.1,
        'reference_sha256': identity(reference), 'expected': reference,
        'observed': observed, 'differences': mismatch, 'observation_errors': errors}
    save_new(path, report)
    if report['status'] != 'ADMITTED':
        raise RuntimeError('Runtime mismatch before model loading: ' + ', '.join(
            row['field'] for row in mismatch + errors) + '. See runtime_preflight.json for expected and observed values.')
    return report


def load_with_runtime_record(plan, snapshot, directory, worker):
    report = record_runtime(plan, directory)
    def admitted_identity(checked_plan):
        if identity(checked_plan['runtime_reference']) != report['reference_sha256']:
            raise ValueError('The loader reference changed after runtime admission')
        return deepcopy({k: report['observed'][k] for k in ('versions', 'gpu', 'optional_kernels', 'packages')})
    # Amend only runtime admission. Keep the retained snapshot, weights and
    # generation checks unchanged, and restore the original function on exit.
    with patch.object(worker, 'runtime_identity', admitted_identity):
        return worker.load_admitted_model(plan, snapshot, directory)
