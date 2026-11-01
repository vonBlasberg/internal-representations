"""Sequential reuse of admitted weights, never of generation state.

This module has no provider or execution entry point. A separately approved
worker supplies its loader and plan. The frozen Inspect and Steerability route
still creates a new pipeline and dynamic generation cache for every trial.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .preparatory_inspect import InspectPhaseGenerator
from .preparatory_phases import identity, save_new
from .research_pipeline import _hooks


class SequentialModelSession:
    def __init__(self, directory, loader, *, expected_weights_sha256, contract):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.loader = loader
        self.expected_weights = expected_weights_sha256
        self.contract = contract
        self.model = self.tokenizer = None
        self.loads = self.completed_trials = self.generation_attempts = 0
        self.generation_attempts_lower_bound = 0
        self.attempted_trials = 0
        self.failed = False
        self.active = None
        self.seen = set()

    def begin(self, trial, directory):
        if self.failed or self.active is not None or trial['trial_id'] in self.seen:
            raise RuntimeError('No concurrent trial, repeated trial or retry after failure.')
        self.seen.add(trial['trial_id'])
        self.attempted_trials += 1
        self.active = (trial, Path(directory), False)

    def acquire(self, directory):
        if self.failed or self.active is None or self.active[1] != Path(directory) or self.active[2]:
            raise RuntimeError('One acquisition is required for the active trial only.')
        trial, directory, _ = self.active
        self.active = (trial, directory, True)
        if self.model is None:
            self.loads += 1
            if self.loads != 1:
                raise RuntimeError('A failed load cannot be repeated.')
            self.model, self.tokenizer = self.loader(self.directory)
        if self.model.training or any(a or b for a, b in _hooks(self.model).values()):
            raise RuntimeError('Shared weights must be in evaluation mode with no remaining hooks.')
        original = self.directory / 'loaded_model.json'
        loaded = json.loads(original.read_text())
        if (loaded['weights']['entries_sha256'] != self.expected_weights
                or identity(loaded['generation_contract']) != identity(asdict(self.contract))):
            raise ValueError('The loaded weights or contract do not match this phase.')
        save_new(directory / 'loaded_model.json', loaded)
        save_new(directory / 'model_use.json', {
            'trial_id': trial['trial_id'], 'attempt_index': self.attempted_trials,
            'load_index': 1, 'reused_weights': self.attempted_trials > 1,
            'load_record': str(original.relative_to(directory.parent.parent.parent)),
            'load_record_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
            'fresh_dynamic_cache_per_generation': True,
            'past_key_values_supplied': False, 'concurrent_generation': False,
        })
        return self.model, self.tokenizer

    def finish(self, success):
        if self.active is None:
            raise RuntimeError('No active trial.')
        trial, directory, acquired = self.active
        try:
            record_path = directory / 'state_execution.json'
            try:
                records = json.loads(record_path.read_text())['records'] if record_path.is_file() else []
                if not record_path.is_file() and (directory / 'model_use.json').exists():
                    self.generation_attempts = None
                if (not isinstance(records, list) or len(records) > 1
                        or any(type(r.get('generation_call_count')) is not int
                               or r['generation_call_count'] not in (0, 1) for r in records)):
                    raise ValueError('The generation count record is incomplete or malformed.')
                if success and (len(records) != 1 or records[0]['generation_call_count'] != 1):
                    raise ValueError('A returned response has no complete generation-count receipt.')
            except BaseException:
                self.generation_attempts = None
                raise
            count = sum(r['generation_call_count'] for r in records)
            self.generation_attempts_lower_bound += count
            if self.generation_attempts is not None:
                self.generation_attempts += count
            if success:
                if not acquired or len(records) != 1:
                    raise RuntimeError('One complete execution record is required.')
                record = records[0]
                if (record['trial_id'] != trial['trial_id'] or record['status'] != 'completed'
                        or record['generation_call_count'] != 1 or record['retry_count'] != 0
                        or not record['weights_unchanged'] or not record['hooks_restored']
                        or record['weights_before']['entries_sha256'] != self.expected_weights
                        or record['weights_after']['entries_sha256'] != self.expected_weights
                        or identity(record['contract']) != identity(asdict(self.contract))
                        or record['generation_call'] != self.contract.generation_kwargs()):
                    raise RuntimeError('The completed trial does not preserve the admitted model and cache policy.')
                if any(a or b for a, b in _hooks(self.model).values()):
                    raise RuntimeError('A hook remains after generation.')
                self.completed_trials += 1
            else:
                self.failed = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = None
            try:
                save_new(directory / 'model_session.json', self.summary())
            except BaseException:
                self.failed = True
                raise

    def summary(self):
        return {'model_load_attempts': self.loads, 'attempted_trials': self.attempted_trials,
            'generation_attempts': self.generation_attempts,
            'generation_attempts_lower_bound': self.generation_attempts_lower_bound,
            'generation_count_verified': self.generation_attempts is not None,
            'completed_trials': self.completed_trials, 'failed': self.failed,
            'automatic_retry': False, 'concurrent_generation': False,
            'reuse_policy': 'same_admitted_weights_new_pipeline_and_dynamic_cache_per_trial'}


class ReusedInspectGenerator(InspectPhaseGenerator):
    def __init__(self, plan, contract, *, model_session, direction_loader):
        self.session = model_session
        super().__init__(plan, contract, model_factory=model_session.acquire, direction_loader=direction_loader)

    def __call__(self, trial, directory):
        # InspectPhaseGenerator admits the exact trial order before acquisition.
        if self.failed or self.next_index >= len(self.plan['trials']) or identity(trial) != identity(self.plan['trials'][self.next_index]):
            raise RuntimeError('Trial order differs or this connection already failed.')
        self.session.begin(trial, directory)
        success = False
        try:
            output = super().__call__(trial, directory)
            success = True
            return output
        finally:
            try:
                self.session.finish(success)
            except BaseException:
                self.failed = True
                raise
