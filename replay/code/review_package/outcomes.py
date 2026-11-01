"""Prospective outcome amendment. Historical scorer and analysis stay untouched."""
from .reuse import legacy


def require_supplier_feasible_standing_offer(case, scoring):
    _, supplier = scoring.payoffs(case, case['counterpart_last_package'])
    threshold = case['reservation_values']['counterpart']
    feasible = supplier >= threshold if case['counterpart_rule']['on_tie'] == 'accept' else supplier > threshold
    if not feasible:
        raise legacy.AnalysisError(case['case_id'] + ': standing offer violates supplier reservation')


def add_agreement_outcomes(score, case):
    """Keep valid adverse decisions; never invent outcomes for invalid/missing rows."""
    result = dict(score)
    for party in ('self', 'counterpart'):
        result['agreement_adjusted_' + party + '_payoff'] = (
            score['proposed_' + party + '_payoff'] if score['agreement'] else case['disagreement_values'][party]
        ) if score['valid'] else None
    result['buyer_below_reservation'] = (
        score['proposed_self_payoff'] < case['reservation_values']['self'] if score['valid'] else None)
    return result


def agreement_bounds(case, scoring):
    """Feasible accepted packages plus declared no-agreement outcomes, not raw offer bounds."""
    values = [sum(case['disagreement_values'].values())]
    for package in scoring.enumerate_packages(case):
        buyer, supplier = scoring.payoffs(case, package)
        threshold = case['reservation_values']['counterpart']
        accepted = supplier >= threshold if case['counterpart_rule']['on_tie'] == 'accept' else supplier > threshold
        if accepted:
            values.append(buyer + supplier)
    standing = scoring.score_response(case, legacy.canonical({'decision': 'accept', 'package': case['counterpart_last_package']}))
    if standing['valid']:
        values.append(standing['agreement_adjusted_joint_gain'])
    return min(values), max(values)


def outcome_contrast(left, right, outcome, cases, scores, ranges):
    if outcome != 'agreement_adjusted_joint_gain':
        return legacy._contrast(left, right, outcome, cases, scores, ranges)
    # The frozen implementation bounds its named primary slot only. Feed that
    # slot a temporary view of the new outcome to reuse identical weights and
    # resampling. Original scores, recorded values and historical results stay intact.
    view = {key: {**value, 'proposed_joint_gain': value[outcome]} for key, value in scores.items()}
    result = legacy._contrast(left, right, 'proposed_joint_gain', cases, view, ranges)
    result['outcome'] = outcome
    return result
