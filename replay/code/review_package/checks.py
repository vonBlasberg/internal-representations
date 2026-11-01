"""Run package software tests and retain their complete output in a new folder."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from .reuse import PACKAGE, legacy

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONPATH': str(PACKAGE / 'code') + os.pathsep + str(PACKAGE / 'code/reused')}
    reports = []
    for name, folder, pattern in [('retained', 'tests/original', 'test_*.py'), ('package', 'tests', 'test_review_package.py'), ('integration', 'tests', 'test_offline_implementation.py'), ('preparation', 'tests', 'test_local_preparation.py'), ('audit_fixes', 'tests', 'test_audit_fixes.py'), ('validation', 'tests', 'test_validation.py'), ('outcome_amendment', 'tests', 'test_outcome_amendment.py'), ('gpu_connection', 'tests', 'test_gpu_connection.py'), ('runtime_preflight', 'tests', 'test_runtime_preflight.py')]:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', folder, '-p', pattern, '-v']
        try:
            run = subprocess.run(command, cwd=PACKAGE, env=env, capture_output=True, text=True, timeout=180)
            output, status = run.stdout + run.stderr, run.returncode
        except subprocess.TimeoutExpired as exc:
            output, status = str(exc), 124
        (args.output / (name + '.txt')).write_text(output)
        count = re.search(r'Ran (\d+) tests?', output)
        reports.append({'suite': name, 'command': command, 'returncode': status, 'tests': int(count.group(1)) if count else None,
                        'output_sha256': legacy.sha(args.output / (name + '.txt'))})
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'suites': reports,
              'total_tests': sum(r['tests'] or 0 for r in reports), 'passed': all(r['returncode'] == 0 for r in reports),
              'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'jsonschema', 'inspect-ai', 'steerability', 'torch', 'transformers', 'pandas')},
              'model_runs': 0, 'provider_calls': 0, 'deployment': False}
    (args.output / 'checks.json').write_text(legacy.canonical(result) + '\n')
    print(legacy.canonical(result))
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__': main()
