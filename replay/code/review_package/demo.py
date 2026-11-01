"""Build a local saved-results review. Real history and synthetic checks stay separate."""
import argparse
from pathlib import Path
from .reuse import legacy
from .history import read_calibration
from .comparison import analyze_saved
from .admission import verify_inputs_and_attempts
from .fixtures import write_fixture, comparison_parts
from .cli import write_new
from .presentation import shell
from .workflow import Study, NegotiationScorer, AllocationScorer
from .inspect_bridge import run_synthetic_inspect, candidate_configurations, no_network


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    histories = []
    with no_network():
        for attempt in ('a05', 'a06', 'a09', 'a10'):
            history = read_calibration(attempt)
            histories.append(history)
            write_new(directory / attempt, history)
        manifest = write_fixture(directory / 'synthetic-inputs')
        write_new(directory / 'synthetic-comparison', analyze_saved(manifest))
        plan = manifest.parent / 'original-plan.json'
        admission = verify_inputs_and_attempts(manifest, plan, legacy.sha(plan))
        (directory / 'synthetic-input-check.json').write_text(legacy.canonical(admission) + '\n')
        write_new(directory / 'synthetic-input-check', {**admission, 'artifact_type': 'INPUT_ADMISSION_REVIEW'})
        parts = comparison_parts()
        parts[2].pop()
        missing = write_fixture(directory / 'incomplete-inputs', parts)
        write_new(directory / 'incomplete-comparison', analyze_saved(missing))
        case = parts[1][0]
        conditions = ('unchanged', 'internal', 'prompt')
        study = Study('Synthetic negotiation integration', (case,), conditions, True)
        answers = {(case['case_id'], c): next(r['raw_response'] for r in parts[2] if r['case_id'] == case['case_id'] and r['condition'] == c) for c in conditions}
        answers[(case['case_id'], 'prompt')] = '{"decision":[],"package":{}}'
        negotiation = run_synthetic_inspect(study, NegotiationScorer(), answers, directory / 'negotiation-inspect-records')
        write_new(directory / 'synthetic-negotiation-reuse', negotiation)
        allocation_cases = ({'case_id': 'allocation-a', 'budget': 100}, {'case_id': 'allocation-b', 'budget': 80})
        study = Study('Synthetic resource-allocation integration', allocation_cases, conditions, True)
        answers = {(c['case_id'], condition): legacy.canonical({'team_a': c['budget'] // 2, 'team_b': c['budget'] // 2}) for c in allocation_cases for condition in conditions}
        answers[('allocation-a', 'internal')] = '{"team_a":70,"team_b":30}'
        answers[('allocation-b', 'prompt')] = '{"team_a":80,"team_b":80}'
        allocation = run_synthetic_inspect(study, AllocationScorer(), answers, directory / 'allocation-inspect-records')
        write_new(directory / 'synthetic-allocation-reuse', allocation)
        configs = candidate_configurations()
        (directory / 'steerability-configurations.json').write_text(legacy.canonical(configs) + '\n')
    body = '''<header><p>Beyond the Answer · internal review</p><h1>What does an internal intervention change?</h1><p>Review the saved evidence, then check how the tool handles selection, consequences and robustness. No model is running here.</p></header>
<section><h2>1. Review the actual result</h2><p><a href="a10/index.html">Open A10: decisions, payoffs and candidate selection</a></p><p>The selected edit changed a decision on one calibration case. It tied prompting, raised buyer payoff and reduced supplier payoff. New-case robustness is not established.</p><p>Earlier no-eligible-candidate outcomes: <a href="a05/index.html">A05</a>, <a href="a06/index.html">A06</a>, <a href="a09/index.html">A09</a>. A04’s ceiling and A07’s technical failure remain in the package’s evidence/history folder, not relabeled as successful trials.</p></section>
<section><h2>2. Review what the next comparison will report</h2><p class="status">Synthetic software tests only. These are authored responses, not new research results.</p><p><a href="synthetic-comparison/index.html">Nine conditions: outcomes, uncertainty and wording sensitivity</a></p><p><a href="incomplete-comparison/index.html">Incomplete delivery: missing attempts stay visible</a></p><p>The proposed comparison holds A10’s settings fixed. The actual new-case bank and execution still require review and approval.</p></section>
<section><h2>3. Check that the workflow can be reused</h2><p>Both synthetic tasks use Steerability’s provider conversion and Inspect’s logging and scoring with authored responses. They share the result, display and export code. These checks do not exercise the research model’s state-recording and editing path.</p><p><a href="synthetic-negotiation-reuse/index.html">Synthetic negotiation task</a> · <a href="synthetic-allocation-reuse/index.html">Synthetic resource-allocation task</a></p><p>Inspect’s own record viewer: <a href="negotiation-inspect-records/inspect-view/index.html">negotiation</a> · <a href="allocation-inspect-records/inspect-view/index.html">resource allocation</a>. These viewers contain synthetic logs only and need a local HTTP preview.</p><p><a href="steerability-configurations.json">Steerability candidate configuration expansion</a> · <a href="synthetic-input-check.json">Original-input and all-attempt check</a></p><p>This tests a reusable interface, not arbitrary-model compatibility or a second empirical application.</p></section>
<section><h2>Questions to answer while reviewing</h2><ol><li>Why did the selected A10 candidate qualify?</li><li>Did it beat prompting, and did both parties benefit?</li><li>Can you find a failed candidate and its original response?</li><li>Can you distinguish missing delivery, invalid output and an estimated zero effect?</li><li>Can you download the exact attempts behind the displayed result?</li></ol><p>The package README lists the remaining empirical, usability and release steps. This is a local review build, not an approved public release or live-generation service.</p></section>'''
    body += '<p><a href="synthetic-input-check/index.html">Open the original-input verification report</a> to see the verified attempt count and the checks still required.</p>'
    (directory / 'index.html').write_text(shell('Beyond the Answer: local review', body))
    return {'historical_attempts_replayed': sum(len(result['rows']) for result in histories),
            'synthetic_inspect_dispatches': sum(result['integration']['synthetic_dispatches'] for result in (negotiation, allocation)),
            'candidate_configurations_expanded': len(configs), 'model_loads': 0, 'provider_calls': 0,
            'zero_call_basis': 'Authored-response pipeline only; outbound sockets blocked during build',
            'network_disabled': True, 'deployment': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output)
    (args.output / 'build-checks.json').write_text(legacy.canonical(result) + '\n')
    print(legacy.canonical(result))


if __name__ == '__main__':
    main()
