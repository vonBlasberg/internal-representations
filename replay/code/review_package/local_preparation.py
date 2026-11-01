"""Author a prospective review bank and an assumption-based precision rehearsal.

No model outputs are consulted. These new inputs are not an approved evaluation
bank; the original protocol, historical cases and all selections stay unchanged.
"""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
from .reuse import PACKAGE, legacy, protocol
from .recording import save_new
from .admission import identity
from .case_validation import validate_case_bank


# Scenario, payoff priorities, thresholds and standing offers are authored here,
# not chosen by inspecting generated outcomes. Each party's maximum is 100 points.
PROFILES = [
    ("pumps", "A water utility is buying replacement pumps for a scheduled maintenance shutdown.", "Replacement pumps are needed for the water utility's planned shutdown.", [40,30,20,10], [40,15,35,10], 40,35, [1,1,1,1]),
    ("scanners", "A regional warehouse is buying handheld scanners before a new picking system launches.", "Before introducing a new picking system, a regional warehouse must purchase handheld scanners.", [30,40,20,10], [35,20,35,10], 45,40, [2,0,1,0]),
    ("freezers", "A food distributor is buying freezers to expand its cold-storage capacity.", "A food distributor needs additional freezers for an expansion of its cold-storage operation.", [35,20,35,10], [45,25,20,10], 35,45, [1,2,0,1]),
    ("printers", "A small publisher is purchasing label printers for a new distribution contract.", "A new distribution contract requires a small publisher to acquire label printers.", [45,15,30,10], [30,30,30,10], 40,30, [0,1,1,2]),
    ("sensors", "A building operator is purchasing environmental sensors for an occupied office complex.", "An occupied office complex requires environmental sensors, which its building operator is purchasing.", [25,30,35,10], [40,25,25,10], 30,40, [1,0,1,1]),
    ("cabinets", "A laboratory is buying safety cabinets for a planned refurbishment.", "As part of a planned refurbishment, a laboratory needs to purchase safety cabinets.", [30,25,35,10], [35,35,20,10], 45,35, [2,1,2,0]),
    ("chargers", "A delivery company is buying depot chargers for a new electric van fleet.", "A delivery company's new electric van fleet requires chargers at its depot.", [35,35,20,10], [25,30,35,10], 40,40, [1,1,0,2]),
    ("filters", "A manufacturer is purchasing air filtration units for a production line.", "Air filtration units must be purchased for a manufacturer's production line.", [40,20,30,10], [30,25,35,10], 35,35, [0,2,1,0]),
    ("terminals", "A retailer is buying payment terminals for a store refit.", "A store refit requires a retailer to purchase new payment terminals.", [25,40,25,10], [45,20,25,10], 40,45, [2,0,0,1]),
    ("compressors", "A repair business is purchasing compressors to replace aging workshop equipment.", "A repair workshop needs new compressors to replace its aging equipment.", [45,25,20,10], [35,15,40,10], 45,40, [1,2,1,2]),
    ("routers", "A hotel group is buying network routers for a planned connectivity upgrade.", "A planned connectivity upgrade requires a hotel group to acquire network routers.", [30,30,30,10], [40,30,20,10], 30,35, [0,2,2,0]),
    ("meters", "A facilities contractor is purchasing energy meters for a portfolio of buildings.", "Energy meters are needed across a portfolio of buildings managed by a facilities contractor.", [35,25,30,10], [30,35,25,10], 35,30, [2,2,1,1]),
]
ISSUES = ("price", "delivery", "warranty", "payment")
LABELS = ("Unit price", "Delivery", "Warranty", "Payment")
OPTIONS = (("Low price", "Medium price", "High price"), ("One week", "Three weeks", "Six weeks"),
           ("One year", "Two years", "Three years"), ("Upfront", "30 days", "60 days"))


def economic_identity(case):
    """Exclude names and wording so relabeling cannot masquerade as a new payoff case."""
    return identity({"payoffs": {issue["issue_id"]: [(o["self_payoff"], o["counterpart_payoff"]) for o in issue["options"]]
                     for issue in sorted(case["issues"], key=lambda i: i["issue_id"])},
                     "reservation": case["reservation_values"], "disagreement": case["disagreement_values"]})


def draft_cases():
    scorer = legacy.scorer()
    bank = []
    for index, (name, text, paraphrase, buyer_weights, supplier_weights, br, sr, standing) in enumerate(PROFILES):
        family = 'prospective-20261031-' + name
        issues = []
        for i, (issue, label, options) in enumerate(zip(ISSUES, LABELS, OPTIONS)):
            b, s = buyer_weights[i], supplier_weights[i]
            # Negotiated trade-offs differ in relative importance and middle-option curvature.
            bm = max(1, b // 2 + (index % 3 - 1) * 2)
            sm = max(1, s // 2 + ((index + 1) % 3 - 1) * 2)
            bp = [0, bm, b] if issue == 'warranty' else [b, bm, 0]
            sp = [0, sm, s] if issue in ('price', 'delivery') else [s, sm, 0]
            issues.append({"issue_id": issue, "label": label, "options": [
                {"option_id": issue + str(j), "label": option, "self_payoff": bp[j], "counterpart_payoff": sp[j]}
                for j, option in enumerate(options)]})
        base = {"schema_version": 1, "output_contract_version": 2, "family_id": family,
            "partition": "untouched", "domain": "procurement", "role": "buyer", "scenario": text,
            "issues": issues, "distributive_issue": "price", "integrative_issues": ["delivery", "warranty"],
            "compatible_issue": "payment", "reservation_values": {"self": br, "counterpart": sr},
            "disagreement_values": {"self": 0, "counterpart": 0},
            "counterpart_rule": {"type": "reservation_rule", "on_tie": "accept"}, "hard_constraints": [],
            "counterpart_last_package": {issue: issue + str(j) for issue, j in zip(ISSUES, standing)},
            "counterpart_stated_priorities": ["Later delivery helps our production schedule.",
                "We prefer a shorter warranty commitment.", "Upfront payment is useful to us."],
            "integrative_opportunity_required": True, "normalization_required": True,
            "sentinel_unrestricted_concession": False,
            "source": {"rights_status": "synthetic_project_authored", "authoring_note":
                "Authored locally on October 31 for prospective owner review, without consulting new model outputs. Not approved or executed. No personal or third-party organizational data."}}
        for variant in ("base", "paraphrase", "issue_order"):
            case = deepcopy(base)
            case.update(case_id=family + '-' + variant.replace('_', '-'), variant_id=variant.replace('_', '-'), variant_of=family + '-base',
                        variant_kind=variant, prompt_style='paraphrase' if variant == 'paraphrase' else 'base')
            if variant == 'paraphrase': case['scenario'] = paraphrase
            if variant == 'issue_order': case['issues'] = list(reversed(case['issues']))
            case['system_text'], case['user_text'] = scorer.render_prompt(case)
            case['case_content_sha256'] = scorer.case_content_sha256(case)
            scorer.validate_case(case)
            bank.append(case)
    validate_case_bank(bank, fixture_only=False)
    return bank


def verify_arithmetic(cases):
    """Independently sum every legal package, then compare the retained scorer."""
    scorer = legacy.scorer()
    report = []
    for case in cases:
        offers = []
        for combination in itertools.product(*(i['options'] for i in case['issues'])):
            package = {i['issue_id']: o['option_id'] for i, o in zip(case['issues'], combination)}
            expected_buyer = sum(o['self_payoff'] for o in combination)
            expected_supplier = sum(o['counterpart_payoff'] for o in combination)
            raw = legacy.canonical({"decision": "offer", "package": package})
            scored = scorer.score_response(case, raw)
            if (not scored['valid'] or scored['proposed_self_payoff'] != expected_buyer
                    or scored['proposed_counterpart_payoff'] != expected_supplier
                    or scored['proposed_joint_gain'] != expected_buyer + expected_supplier):
                raise legacy.AnalysisError("Independent draft-case arithmetic differs")
            offers.append((expected_buyer, expected_supplier))
        report.append({"case_id": case['case_id'], "packages_checked": len(offers),
            "buyer_max": max(b for b, s in offers), "supplier_max": max(s for b, s in offers),
            "joint_range": [min(b+s for b, s in offers), max(b+s for b, s in offers)],
            "economic_identity": economic_identity(case), "case_sha256": identity(case),
            "standing_offer": scorer.case_summary(case), "model_visible_sha256": identity([case['system_text'], case['user_text']])})
    return report


def precision_rehearsal():
    patterns = {"constant_zero": [0]*12, "constant_plus_ten": [10]*12,
        "mixed_small": [-10,-8,-6,-4,-2,0,2,4,6,8,10,12],
        "mixed_large": [-40,-30,-20,-10,0,10,20,30,40,50,60,70],
        "one_influential_family": [0]*11+[60]}
    rows = []
    for name, values in patterns.items():
        interval = legacy.family_bootstrap(values)
        rows.append({"pattern": name, "authored_family_differences": values,
            "mean": sum(values)/len(values), "case_bank_sensitivity_interval": interval,
            "interval_half_width": (interval[1]-interval[0])/2,
            "empirical_evidence": False})
    return {"artifact_type": "ASSUMPTION_BASED_PRECISION_REHEARSAL", "fixture_only": True,
        "family_count": 12, "variant_count_per_family": 3, "rows": rows,
        "method": "Existing 10,000-draw paired-family resampling, seed 20261029; not power or population coverage",
        "conclusion": "Twelve families do not guarantee narrow intervals. Width depends on heterogeneity. Constant patterns give zero width by construction, not evidence of certainty.",
        "negligible_effect_threshold": None, "equivalence_claim_authorized": False}


def check_declared_prior_inputs(cases, prototype):
    """Read declared case banks and saved requests only, never generation outputs."""
    paths = sorted(set((prototype / 'negotiation-steering/study/materials').glob('*.jsonl')) |
                   set((prototype / 'records').glob('*/*/request.json')) |
                   set((prototype / 'records').glob('*/*cases*.json')))
    sources, old_cases = [], []
    def visit(value):
        if isinstance(value, dict):
            if {'case_id', 'issues', 'reservation_values', 'disagreement_values', 'scenario'} <= set(value):
                old_cases.append(value)
            else:
                for nested in value.values(): visit(nested)
        elif isinstance(value, list):
            for nested in value: visit(nested)
    for path in paths:
        before = len(old_cases)
        text = path.read_text(encoding='utf-8')
        if path.suffix == '.jsonl':
            for line in text.splitlines():
                if line.strip(): visit(json.loads(line))
        else: visit(json.loads(text))
        sources.append({'path': str(path.relative_to(prototype)), 'sha256': legacy.sha(path),
                        'case_objects': len(old_cases)-before})
    old_economics = {economic_identity(c) for c in old_cases}
    old_scenarios = {c['scenario'].strip().casefold() for c in old_cases}
    duplicates = [{'case_id': c['case_id'], 'economic_duplicate': economic_identity(c) in old_economics,
        'exact_scenario_duplicate': c['scenario'].strip().casefold() in old_scenarios} for c in cases
        if economic_identity(c) in old_economics or c['scenario'].strip().casefold() in old_scenarios]
    return {'status': 'DECLARED_INPUT_SCAN_ONLY', 'source_files': sources, 'case_objects_read': len(old_cases),
        'distinct_prior_economic_structures': len(old_economics), 'duplicates': duplicates,
        'complete_exposure_review': False,
        'limitations': ['Code-generated amendments absent from saved requests need separate review',
            'Numerically distinct families are not automatically substantively independent',
            'Pretraining exposure and external case reuse are not established by local hashes']}


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cases = draft_cases()
    arithmetic = verify_arithmetic(cases)
    if len({economic_identity(c) for c in cases}) != 12:
        raise legacy.AnalysisError("Draft family payoff structures duplicate one another")
    save_new(directory / 'draft-cases.json', cases)
    save_new(directory / 'arithmetic.json', arithmetic)
    save_new(directory / 'precision.json', precision_rehearsal())
    exposure = check_declared_prior_inputs(cases, PACKAGE.parent / 'STAGE1_INTERVENTION_PROTOTYPE_2026-10-25')
    save_new(directory / 'declared-prior-input-check.json', exposure)
    spec, digest = protocol()
    grid = [{"trial_id": c['case_id'] + ':' + condition, "case_id": c['case_id'],
             "family_id": c['family_id'], "condition": condition, **settings}
            for c in cases for condition, settings in spec['conditions'].items()]
    save_new(directory / 'proposed-trial-grid.json', grid)
    result = {"status": "PROSPECTIVE_DRAFT_FOR_REVIEW_NOT_EXECUTION", "execution_enabled": False,
        "case_count": len(cases), "family_count": 12, "planned_responses_if_approved": len(grid),
        "packages_independently_scored": sum(r['packages_checked'] for r in arithmetic),
        "model_runs": 0, "protocol_sha256_unchanged": digest,
        "draft_case_bank_sha256": legacy.sha(directory / 'draft-cases.json'),
        "declared_prior_inputs_scanned": len(exposure['source_files']),
        "exact_prior_duplicates_found": len(exposure['duplicates']),
        "comparable_scale": "Each party has an unconstrained maximum of 100 and disagreement payoff zero",
        "rights": "New synthetic project-authored text and numbers; no third-party records or personal data",
        "not_established": ["Independent substantive case review", "Complete prior-exposure review, including programmatically amended cases",
            "Real tokenizer IDs and exact runtime/direction binding", "Owner approval of ninth condition, case bank and any negligible-effect threshold"],
        "scientific_boundary": "Authored transfer scenarios, not independent draws from an organizational population; no new outputs or baseline-headroom filtering"}
    save_new(directory / 'preparation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(legacy.canonical(prepare(args.output)))


if __name__ == '__main__': main()
