"""Verify a package inventory and run model-free tests into a new external folder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error('Write verification outside the inventoried package')
    output.mkdir(parents=True, exist_ok=False)
    inventory = json.loads(args.inventory.read_text())
    failures = []
    paths = set()
    for row in inventory['files']:
        relative = row['path']
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or relative in paths:
            raise ValueError('Unsafe or duplicate inventory path')
        paths.add(relative)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append(relative)
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    extras = actual - paths - set(inventory.get('excluded_inventory_paths', []))
    result = {'entries': len(paths), 'differences': failures, 'unexpected_files': sorted(extras),
              'passed': not failures and not extras, 'model_runs': 0}
    (output / 'inventory_check.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        print(json.dumps(result))
        raise SystemExit(1)
    if not args.inventory_only:
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(root / 'code')}
        subprocess.run([sys.executable, '-B', '-m', 'review_package.checks', '--output', str(output / 'tests')],
                       env=env, cwd=root, check=True)
    print(json.dumps(result))


if __name__ == '__main__': main()
