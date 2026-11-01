"""Independent payoff/accounting check of an already audited saved comparison.

Uses JSON, the declared utility tables and NumPy only. Does not import the
research scorer, call a provider or alter the fixed analysis.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np


def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def score(raw, case):
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        assert isinstance(value, dict) and set(value) == {'decision', 'package'}
        assert isinstance(value['decision'], str) and value['decision'] in ('accept', 'offer')
        package = value['package']
        options = {i['issue_id']: {o['option_id']:o for o in i['options']} for i in case['issues']}
        assert isinstance(package, dict) and set(package) == set(options)
        assert all(isinstance(v, str) and v in options[k] for k,v in package.items())
        assert not any(all(package[k] == v for k,v in c['forbid'].items()) for c in case['hard_constraints'])
        assert value['decision'] != 'accept' or package == case['counterpart_last_package']
    except (ValueError, TypeError, AssertionError, KeyError):
        return {'valid':False}
    buyer = sum(options[k][v]['self_payoff'] for k,v in package.items())
    supplier = sum(options[k][v]['counterpart_payoff'] for k,v in package.items())
    threshold = case['reservation_values']['counterpart']
    agreement = value['decision'] == 'accept' or (supplier >= threshold if case['counterpart_rule']['on_tie'] == 'accept' else supplier > threshold)
    actual_buyer = buyer if agreement else case['disagreement_values']['self']
    actual_supplier = supplier if agreement else case['disagreement_values']['counterpart']
    return {'valid':True, 'decision':value['decision'], 'package':package, 'agreement':agreement,
        'proposed_self_payoff':buyer, 'proposed_counterpart_payoff':supplier, 'proposed_joint_gain':buyer+supplier,
        'agreement_adjusted_self_payoff':actual_buyer, 'agreement_adjusted_counterpart_payoff':actual_supplier,
        'agreement_adjusted_joint_gain':actual_buyer+actual_supplier}


def accepted_bounds(case):
    values = [sum(case['disagreement_values'].values())]
    options = {i['issue_id']:i['options'] for i in case['issues']}
    for chosen in itertools.product(*options.values()):
        package = dict(zip(options, [o['option_id'] for o in chosen]))
        if any(all(package[k] == v for k,v in c['forbid'].items()) for c in case['hard_constraints']): continue
        supplier = sum(o['counterpart_payoff'] for o in chosen)
        threshold = case['reservation_values']['counterpart']
        if supplier >= threshold if case['counterpart_rule']['on_tie'] == 'accept' else supplier > threshold:
            values.append(supplier + sum(o['self_payoff'] for o in chosen))
    return min(values), max(values)


def close(left, right):
    return left == right if left is None or right is None else math.isclose(left, right, abs_tol=1e-9, rel_tol=1e-9)


def check(report_path, request_path):
    report = json.loads(Path(report_path).read_text())
    case_list = json.loads(Path(request_path).read_text())['cases']
    cases = {c['case_id']:c for c in case_list}
    rows = report['all_trials']; scores = {}
    for row in rows:
        key = row['case_id'], row['condition']
        assert key not in scores, 'Duplicate attempt'
        expected = score(row['raw_response'], cases[key[0]])
        assert expected['valid'] == row['score']['valid'], key
        if expected['valid']:
            for name,value in expected.items(): assert value == row['score'][name], (key,name,value,row['score'][name])
        else:
            assert row['score']['agreement_adjusted_joint_gain'] is None, key
        scores[key] = expected
    assert len(scores) == len(cases)*len(report['accounting'])
    summary = {'status':'PASS', 'run_id':report['run_id'], 'fixture_only':report['fixture_only'],
        'cases':len(cases), 'families':report['family_count'], 'attempts':len(rows),
        'source_report_sha256':hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
        'score_checks':len(rows), 'conditions':[], 'contrasts':[], 'same_case_comparisons':[]}
    for account in report['accounting']:
        arm = account['condition']; valid = [scores[(cid,arm)] for cid in cases if scores[(cid,arm)]['valid']]
        assert len(valid) == account['valid'] and len(cases)-len(valid) == account['invalid']
        assert sum(x['decision'] == 'offer' for x in valid) == account['offers']
        assert sum(x['agreement'] for x in valid) == account['simulated_agreements']
        item = {'condition':arm, 'valid':len(valid), 'invalid':len(cases)-len(valid),
            'agreements':sum(x['agreement'] for x in valid), 'offers':sum(x['decision']=='offer' for x in valid),
            'all_cases_valid':len(valid)==len(cases)}
        for outcome in ('proposed_self_payoff','proposed_counterpart_payoff','proposed_joint_gain',
                        'agreement_adjusted_self_payoff','agreement_adjusted_counterpart_payoff','agreement_adjusted_joint_gain'):
            item[outcome] = statistics.mean(x[outcome] for x in valid) if len(valid)==len(cases) else None
        summary['conditions'].append(item)
    groups = defaultdict(list)
    for cid,c in cases.items(): groups[c['family_id']].append(cid)
    for contrast in report['contrasts']:
        left,right = contrast['left'],contrast['right']; means=[];bounds=[]
        for family in sorted(groups):
            changes=[];limits=[]
            for cid in groups[family]:
                a,b=scores[(cid,left)],scores[(cid,right)]
                lower,upper=accepted_bounds(cases[cid])
                av=[a['agreement_adjusted_joint_gain']]*2 if a['valid'] else [lower,upper]
                bv=[b['agreement_adjusted_joint_gain']]*2 if b['valid'] else [lower,upper]
                limits.append([av[0]-bv[1],av[1]-bv[0]])
                if a['valid'] and b['valid']: changes.append(av[0]-bv[0])
            means.append(statistics.mean(changes) if len(changes)==len(groups[family]) else None)
            bounds.append([statistics.mean(x[i] for x in limits) for i in (0,1)])
        full = all(x is not None for x in means)
        point = statistics.mean(means) if full else None
        assert close(point, contrast['point_estimate']), (left,right,'mean')
        computed_bounds=[statistics.mean(x[i] for x in bounds) for i in (0,1)]
        assert all(close(a,b) for a,b in zip(computed_bounds,contrast['full_case_bank_invalid_response_bounds']))
        interval=None
        if full:
            values=np.array(means); rng=np.random.Generator(np.random.PCG64(20261029))
            samples=values[rng.integers(0,len(means),size=(10000,len(means)))].mean(axis=1)
            interval=np.quantile(samples,[.025,.975],method='linear').tolist()
            assert all(close(a,b) for a,b in zip(interval,contrast['case_bank_sensitivity_interval'])),(left,right,'interval')
        else: assert contrast['case_bank_sensitivity_interval'] is None
        summary['contrasts'].append({'left':left,'right':right,'point':point,'case_bank_interval':interval,
            'invalid_response_bounds':computed_bounds,'family_effects':dict(zip(sorted(groups),means))})
    for left,right in (('internal','unchanged'),('internal','prompt'),('prompt','unchanged')):
        n=changed=same=better=worse=mutual=adverse=0
        for cid in cases:
            a,b=scores[(cid,left)],scores[(cid,right)]
            if not a['valid'] or not b['valid']:continue
            n+=1; identical=a['decision']==b['decision'] and a['package']==b['package']
            same+=identical;changed+=not identical
            d=a['agreement_adjusted_joint_gain']-b['agreement_adjusted_joint_gain']
            better+=d>0;worse+=d<0
            db=a['agreement_adjusted_self_payoff']-b['agreement_adjusted_self_payoff']
            ds=a['agreement_adjusted_counterpart_payoff']-b['agreement_adjusted_counterpart_payoff']
            mutual+=db>=0 and ds>=0 and (db>0 or ds>0)
            adverse+=db<0 or ds<0
        summary['same_case_comparisons'].append({'left':left,'right':right,'valid_pairs':n,
            'decision_or_package_changed':changed,'identical_decision_and_package':same,
            'higher_adjusted_joint_payoff':better,'lower_adjusted_joint_payoff':worse,
            'at_least_one_benefits_neither_loses':mutual,'at_least_one_loses':adverse})
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=check(args.report,args.request)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
