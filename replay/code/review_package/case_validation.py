"""Prospective case checks and review of exact model-visible text. No inference.

Structural checks cannot establish semantic equivalence or detect every disclosure
in free text. A separately reviewed, pinned plan must retain the human assessment.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
from .reuse import legacy
from .inputs import render_messages
from .outcomes import require_supplier_feasible_standing_offer


def _identity(value):
    return hashlib.sha256(legacy.canonical(value).encode()).hexdigest()


def validate_setup(manifest, cases, spec, spec_hash):
    """The fixed prospective study must be checked before tokenization or analysis."""
    if manifest.get('protocol_sha256') != spec_hash:
        raise legacy.AnalysisError('Comparison does not bind the reviewed protocol bytes')
    if manifest.get('phase') != 'untouched' or type(manifest.get('fixture_only')) is not bool:
        raise legacy.AnalysisError('Comparison phase or fixture identity differs')
    for key in ('model_id', 'model_revision', 'prompt_text'):
        if manifest.get(key) != spec[key]:
            raise legacy.AnalysisError('Fixed prospective setting differs: ' + key)
    if manifest.get('analysis_contract_sha256') != legacy.PINS['study/manifests/behavioral_analysis_contract.json']:
        raise legacy.AnalysisError('Underlying analysis contract differs')
    if not isinstance(cases, list) or not cases:
        raise legacy.AnalysisError('A nonempty case bank is required')
    report = validate_case_bank(cases, fixture_only=manifest['fixture_only'])
    if any(c['partition'] != 'untouched' or c.get('output_contract_version') != 2 for c in cases):
        raise legacy.AnalysisError('Untouched decision-only cases are required')
    if not manifest['fixture_only']:
        if report['family_count'] != spec['proposed_family_count']:
            raise legacy.AnalysisError('Case count differs from the reviewed scope')
        for family in report['families']:
            if sorted(c['variant_kind'] for c in cases if c['family_id'] == family) != sorted(spec['proposed_variants']):
                raise legacy.AnalysisError('Each family needs the prespecified variants')
    return report


def validate_case_bank(cases, *, fixture_only):
    scoring = legacy.scorer()
    report = scoring.validate_family_partitions(cases)
    groups = defaultdict(list)
    for case in cases:
        if not fixture_only:
            require_supplier_feasible_standing_offer(case, scoring)
        groups[case['family_id']].append(case)
    for family, values in groups.items():
        bases = [c for c in values if c['variant_kind'] == 'base']
        if len(bases) != 1:
            raise legacy.AnalysisError(f'{family}: exactly one base in the same family is required')
        base = bases[0]
        for case in values:
            if case['variant_of'] != base['case_id']:
                raise legacy.AnalysisError(f'{family}: variant references another family')
            expected = deepcopy(base)
            if case['variant_kind'] == 'paraphrase':
                normalized = lambda text: ' '.join(text.split()).casefold()
                if not fixture_only and normalized(case['scenario']) == normalized(base['scenario']):
                    raise legacy.AnalysisError('Prospective scenario paraphrase must change scenario wording')
                expected.update(scenario=case['scenario'], prompt_style='paraphrase')
            elif case['variant_kind'] == 'issue_order':
                order = [i['issue_id'] for i in case['issues']]
                if order == [i['issue_id'] for i in base['issues']]:
                    raise legacy.AnalysisError('Issue-order variant must actually reorder issues')
                source = {i['issue_id']: i for i in base['issues']}
                expected['issues'] = [source[i] for i in order]
            rendered = scoring.render_prompt(case)
            if rendered != scoring.render_prompt(expected):
                raise legacy.AnalysisError('Variant changes model-visible content outside its declared operation')
            if case['variant_kind'] != 'base' and rendered == scoring.render_prompt(base):
                raise legacy.AnalysisError('Variant does not change the model-visible input')
    return report


REVIEW_CHECKS = ('no_private_scoring_information', 'variant_meaning_preserved')


def make_prompt_review(cases, spec):
    """Prepare pending review, grouping conditions with identical visible inputs."""
    inputs = []
    for case in cases:
        grouped = {}
        for condition in spec['conditions']:
            messages = render_messages(case, condition, spec)
            key = _identity(messages)
            entry = grouped.setdefault(key, {'case_id': case['case_id'], 'conditions': [], 'messages': messages})
            entry['conditions'].append(condition)
        inputs.extend(grouped.values())
    return {'artifact_type': 'MODEL_VISIBLE_TEXT_REVIEW', 'status': 'REVIEW_REQUIRED',
        'reviewer': None, 'checks': {k: False for k in REVIEW_CHECKS},
        'case_bank_sha256': _identity(cases), 'model_visible_sha256': _identity(inputs),
        'inputs': inputs,
        'scope': 'Human review of disclosure and preserved meaning; not approval to execute or proof of construct validity'}


def require_prompt_review(review, cases, spec):
    """Reject missing or stale review. A hash does not authenticate its reviewer."""
    expected = make_prompt_review(cases, spec)
    if (not isinstance(review, dict) or review.get('artifact_type') != expected['artifact_type']
            or review.get('status') != 'APPROVED'
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
            or any((review.get('checks') or {}).get(k) is not True for k in REVIEW_CHECKS)
            or any(review.get(k) != expected[k] for k in ('case_bank_sha256', 'model_visible_sha256', 'inputs'))):
        raise legacy.AnalysisError('Human review of exact model-visible text is missing, incomplete or stale')
