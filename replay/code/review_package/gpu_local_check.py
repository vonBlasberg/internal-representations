"""324 real Inspect/Steerability dispatches with authored answers. No model or network."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time
from unittest.mock import patch
from .reuse import legacy
from .admission import identity
from .recording import save_new, collect_saved_inspect
from .inspect_bridge import AuthoredResponsePipeline, no_network
from .gpu_connection import read, runtime, phase_plan, execute_sequence
from .gpu_transport import load
from .comparison import analyze_saved
from .cli import write_new


def run(prepared_dir, research_root, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request, original, transport, _ = load(prepared_dir)
    request, original = deepcopy(request), deepcopy(original)
    request['run_id'] += '-SYNTHETIC-ONLY'
    request['manifest'].update(run_id=request['run_id'], fixture_only=True)
    original.update(run_id=request['run_id'], fixture_only=True)
    original.pop('preparation_only')
    original.pop('prompt_review')
    for trial in original['trials']:
        trial.update(run_id=request['run_id'], fixture_only=True)
    save_new(output / 'original-plan.json', original)
    runtime(research_root)
    from local_integration.reused_model import ReusedInspectGenerator
    from local_integration.package_qualification import validate_plan
    from local_integration import preparatory_inspect as bridge
    from transformers import AutoTokenizer
    snapshot = Path(research_root) / 'negotiation-steering/snapshots/qwen35-9b' / request['manifest']['model_revision'] / 'verified-snapshot'
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    contract = validate_plan(request['qualification'])
    cases = {c['case_id']: c for c in request['cases']}
    by_id = {t['trial_id']: t for t in phase_plan(request, original)['trials']}
    calls = []

    class Session:
        """Replace ONLY weight acquisition/session mechanics, never Inspect or ordering."""
        def begin(self, trial, directory):
            self.active = trial
        def acquire(self, directory):
            return object(), tokenizer
        def finish(self, success):
            if not success:
                raise RuntimeError('Authored dispatch failed; no retry')

    class Pipeline(AuthoredResponsePipeline):
        def __init__(self, model, tokens, supplied_contract, **kwargs):
            if asdict(supplied_contract) != asdict(contract):
                raise ValueError('Contract changed in connection')
            super().__init__({})
            self.tokenizer, self.records = tokens, []
        @contextmanager
        def trial(self, trial_id, expected):
            self.active, self.expected = by_id[trial_id], list(expected)
            wanted = next(r for r in original['trials'] if r['trial_id'] == trial_id)['execution_input']['input_ids']
            if self.expected != wanted:
                raise ValueError('Real tokenizer differs from frozen inputs')
            yield
        def inspect_model(self):
            from steerability.evaluation.provider import as_inspect_model, ProviderOptions
            return as_inspect_model(self, model_name=request['manifest']['model_id'],
                options=ProviderOptions(max_batch_size=1, default_max_tokens=160))
        def generate(self, *, messages=None, **kwargs):
            import torch
            from steerability.algorithms.core.output import Output
            conversation = messages if isinstance(messages[0], dict) else messages[0]
            if conversation != self.active['messages']:
                raise ValueError('Provider changed prepared messages')
            case = cases[self.active['case_id']]
            arm = self.active['arm_id']
            # Test accepted, offered, rejected and malformed decisions. No empirical interpretation.
            if arm == 'permuted':
                answer = '{"decision":[],"package":{}}'
            elif arm == 'reverse':
                package = {i['issue_id']: i['options'][0]['option_id'] for i in case['issues']}
                answer = legacy.canonical({'decision':'offer','package':package})
            elif arm in ('internal', 'prompt'):
                package = {i['issue_id']: i['options'][-1]['option_id'] for i in case['issues']}
                answer = legacy.canonical({'decision':'offer','package':package})
            else:
                answer = legacy.canonical({'decision':'accept','package':case['counterpart_last_package']})
            ids = self.tokenizer.encode(answer, add_special_tokens=False)
            self.records.append({'fixture_only': True, 'model_execution': False, 'trial_id': self.active['trial_id'],
                'generated_token_ids': ids, 'processed_logit_steps': [{'SYNTHETIC_ONLY': True, 'token': i} for i in ids],
                'input_token_ids': self.expected, 'edit': self.active['edit']})
            calls.append(self.active['trial_id'])
            result = Output(output_ids=torch.tensor([ids]), adapted_input_ids=torch.tensor([self.expected]), finish_reasons=('eos',))
            return result if isinstance(messages[0], dict) else [result]

    def verify(trial, source, directory, returned):
        records = read(directory / 'state_execution.json')['records']
        if len(records) != 1 or records[0].get('fixture_only') is not True or records[0]['model_execution'] is not False:
            raise ValueError('Synthetic verifier accepts only explicit synthetic receipts')
        return records[0]
    generator = ReusedInspectGenerator(phase_plan(request, original), contract,
        model_session=Session(), direction_loader=lambda *_: None)
    import inspect_ai._util.appdirs as appdirs
    began = time.monotonic()
    with no_network(), patch.object(bridge, 'RecordedSteeringPipeline', Pipeline), \
            patch.object(appdirs, 'user_data_path', lambda _: output / 'local-data'), \
            patch.object(appdirs, 'user_cache_path', lambda _: output / 'local-cache'):
        outer, deliveries = execute_sequence(request, original, output / 'original-plan.json', output / 'execution', generator, verify)
    manifest = collect_saved_inspect(output / 'delivery', request['manifest'], request['cases'],
        output / 'original-plan.json', legacy.sha(output / 'original-plan.json'), deliveries, outer)
    analysis = analyze_saved(manifest)
    write_new(output / 'report', analysis)
    if len(calls) != len(set(calls)) or len(calls) != 324 or len(analysis['all_trials']) != 324:
        raise RuntimeError('The full inventory was not processed exactly once')
    result = {'status':'PASSED', 'fixture_only':True, 'empirical_evidence':False,
        'authored_dispatches':len(calls), 'model_loads':0, 'provider_service_calls':0,
        'network_disabled':True, 'real_tokenizer':True, 'retained_inspect_generator':True,
        'real_steerability_provider':True, 'real_inspect_original_logs':len(deliveries),
        'invalid_retained':sum(not r['score']['valid'] for r in analysis['all_trials']),
        'request_sha256': transport['request_sha256'], 'transport_sha256':identity(transport),
        'seconds':time.monotonic()-began, 'GPU_mechanics_tested':False,
        'remaining':'Real GPU execution and independent terminal mechanical audit require approval.'}
    save_new(output / 'checks.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(legacy.canonical(run(args.prepared, args.research_root, args.output)))
