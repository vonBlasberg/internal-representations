"""Offline display adapter. Every number comes from the accompanying review JSON."""
from html import escape
from .reuse import legacy

def text(value):
    return escape("Not available" if value is None else str(value))

def table(headings, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th scope="col">' + text(h) + '</th>' for h in headings) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + text(v) + '</td>' for v in r) + '</tr>' for r in rows) + '</tbody></table></div>'

def details(row):
    score = row.get("score", row)
    return '<details><summary>' + text(row.get("trial_id", str(row.get("case_id")) + ' / ' + str(row.get("condition")))) + '</summary><h4>Original response</h4><pre>' + text(row.get("raw_response")) + '</pre><h4>Score and source</h4><pre>' + text(legacy.canonical({"score": score, "source": row.get("source", {k: row.get(k) for k in ("source_log_path", "source_log_sha256", "source_sample_id")}), "selection": {k: row.get(k) for k in ("eligibility_checks", "historical_rank", "ranking_rule", "selection_reasons")}})) + '</pre></details>'


def selected_change_summary(rows):
    selected = [r for r in rows if r['kind'] == 'internal' and r['historically_selected']]
    if not selected and any(r['kind'] == 'internal' for r in rows):
        return 'No internal candidate qualified under the recorded rule.'
    if len(selected) != 1 or not selected[0]['score']['valid']:
        return ''
    chosen = selected[0]
    delta = chosen['decision_and_payoffs']
    sentence = 'Selected internal edit versus baseline: buyer {:+g}, supplier {:+g}, combined {:+g} points.'.format(delta['buyer_delta'], delta['supplier_delta'], delta['joint_delta'])
    prompts = [r for r in rows if r['kind'] == 'prompt']
    if prompts and all(r['score']['valid'] and r['score']['package'] == chosen['score']['package']
                       and r['score']['decision'] == chosen['score']['decision'] for r in prompts):
        sentence += ' All ' + str(len(prompts)) + ' tested prompts produced the same decision and package.'
    if delta['buyer_gains_supplier_loses']:
        sentence += ' The buyer gained at the supplier\'s expense; this is not a mutual gain.'
    return sentence

def page(result):
    if result["artifact_type"] == "INPUT_ADMISSION_REVIEW":
        title = 'Original inputs and attempt coverage'
        body = '<header><p>Beyond the Answer · local review</p><h1>' + title + '</h1><p class="status">' + text(result['status']) + '</p><nav><a href="review.json" download>Full verification JSON</a></nav></header>'
        body += '<section><h2>What was verified?</h2><p>' + text(result['attempt_count']) + ' attempts verified against the pinned original plan and declared source logs.</p>'
        body += '<p>' + ('Synthetic software test. These are authored inputs and responses, not research findings.' if result['fixture_only'] else 'Saved execution records. This check alone does not admit the research result.') + '</p>'
        body += table(['Record', 'SHA256'], [['Original plan', result['original_plan_sha256']], ['Outer completion record', result['outer_record_sha256']], *[[name, digest] for name, digest in result['input_sha256'].items()]]) + '</section>'
        body += '<section><h2>Checks still required</h2><ul>' + ''.join('<li>' + text(item) + '</li>' for item in result['remaining']) + '</ul><p>No model run, runtime qualification, case-independence approval or research-result admission is performed by this report.</p></section>'
        return shell(title, body)
    historical = result["artifact_type"] == "HISTORICAL_SELECTION_REVIEW"
    reusable = result["artifact_type"] == "REUSABLE_SYNTHETIC_WORKFLOW_CHECK"
    rows = result.get("all_trials", result.get("rows", []))
    title = (result.get("attempt", "").upper() + ': candidate comparison') if historical else result.get("name", "Fixed-candidate comparison")
    status = result.get("evidence_label", result.get("scope", "Unverified saved records"))
    body = '<header><p>Beyond the Answer · local review</p><h1>' + text(title) + '</h1><p class="status">' + text(status) + '</p><p>This page cannot start a model run, change the selection or deploy anything.</p><nav><a href="../index.html">All examples</a><a href="review.json" download>Full JSON</a><a href="all_attempts.csv" download>All attempts CSV</a></nav></header>'
    if historical:
        body += '<section><h2>What happened?</h2><p>These are saved calibration results, not a test on new cases. Eligibility follows the original rule: a valid answer, higher combined payoff and no lower buyer payoff. Supplier preservation and beating prompting were not selection requirements.</p>'
        selected = [r for r in rows if r["kind"] == "unchanged" or r["historically_selected"]]
        body += table(["Condition", "Decision", "Proposed buyer points", "Proposed supplier points", "Proposed combined points", "Agreement-adjusted combined points"], [[r["arm_id"], r["score"]["decision"], r["score"]["proposed_self_payoff"], r["score"]["proposed_counterpart_payoff"], r["score"]["proposed_joint_gain"], r["score"]["agreement_adjusted_joint_gain"]] for r in selected])
        body += '<p>' + text(selected_change_summary(rows)) + '</p>'
        body += '<p>Agreement-adjusted outcomes apply the case’s simulated acceptance rule. They are not observed agreements with human participants.</p></section><section><h2>Which candidates qualified?</h2>'
        body += table(["Candidate", "Layer", "Strength", "Valid", "Joint gain > baseline", "Buyer preserved", "Rank", "Saved selection"], [[r["arm_id"], (r["edit"] or {}).get("layer"), (r["edit"] or {}).get("coefficient"), r["eligibility_checks"]["valid_response"], r["eligibility_checks"]["strict_joint_gain"], r["eligibility_checks"]["buyer_payoff_preserved"], r["historical_rank"], r["historically_selected"]] for r in rows if r["candidate_for_selection"]])
        body += '<p>Eligible internal edits are ordered by combined payoff, then smaller absolute strength, then lower layer. Eligible prompts use payoff, then declared order. A passing candidate is not necessarily robust or specific to the intended construct.</p></section>'
        controls = [r for r in rows if not r['candidate_for_selection'] and r['kind'] != 'unchanged']
        groups = []
        for kind in dict.fromkeys(r['kind'] for r in controls):
            group = [r for r in controls if r['kind'] == kind]
            groups.append([kind, len(group), sum(r['score']['valid'] for r in group),
                sum(r['score']['valid'] and bool(r['decision_and_payoffs']['decision_changed'] or r['decision_and_payoffs']['package_changed']) for r in group),
                sum(not r['score']['valid'] for r in group)])
        body += '<section><h2>What did the controls do?</h2><p>Counts below describe tested settings on this one case, not independent observations. Changes include a different decision or negotiated terms. Inspect each original response below.</p>' + table(['Control', 'Settings tested', 'Valid', 'Changed decision or terms', 'Invalid'], groups) + '</section>'
        geometry = result.get('post_hoc_geometry')
        if geometry:
            body += '<section><h2>Direction alignment and observed decisions</h2><p class="status">Post hoc analysis of the same calibration case, not new-case evidence.</p><p>' + text(geometry['interpretation']) + '</p><p>Rows are ordered within layer by the intended displacement along the target direction. A positive value moves along it; zero means no component along it. Realized L2 is the minimum and maximum over recorded edit applications, not the summed effect over a response. All values and source hashes are exported.</p>'
            body += '<p>Cosine describes the underlying vector before applying strength. Reversal uses the same vector with negative strength, so its signed displacement is negative even though its vector cosine is positive.</p>'
            body += table(['Layer / arm', 'Strength', 'Direction norm', 'Underlying vector cosine', 'Intended along target', 'Realized L2 range', 'Decision'],
                [[str(r['layer']) + ' / ' + r['arm_id'], r['strength'], round(r['direction_norm'], 4), round(r['cosine_with_intended'], 4), round(r['intended_projection_on_target'], 4),
                  str(round(r['realized_displacement_l2_min'], 4)) + ' to ' + str(round(r['realized_displacement_l2_max'], 4)), r['decision']] for r in geometry['rows']]) + '</section>'
    elif reusable:
        names = result["outcome_names"]
        body += '<section><h2>Shared workflow, different scorer</h2><p>All responses below were authored for a software test. They are not model outputs or empirical findings. The same recording, display and export code accepts different outcome names.</p>'
        body += table(["Case", "Condition", "Execution", "Valid", *names], [[r["case_id"], r["condition"], r["execution_status"], r["valid"], *[r["outcomes"].get(k) for k in names]] for r in rows]) + '</section>'
    else:
        if result.get('primary_outcome') == 'agreement_adjusted_joint_gain':
            body += '<section><h2>What counts as an improvement?</h2><p>The primary outcome is combined payoff after the simulated supplier acceptance rule. If an offer is rejected, each party receives its declared disagreement payoff. Proposed terms and both parties’ outcomes remain visible. A valid decision below the buyer’s stated minimum is reported as adverse, not discarded.</p></section>'
        body += '<section><h2>All conditions and attempts</h2><p>Settings are fixed before evaluation. This view does not select again. Synthetic data test the analysis, not the intervention.</p>'
        body += table(["Condition", "Planned", "Completed", "Valid", "Invalid", "Technical", "Unresolved"], [[r[k] for k in ("condition", "planned", "completed", "valid", "invalid", "technical_failure", "indeterminate")] for r in result.get("accounting", [])])
        if result.get("missing_attempts"):
            body += '<p class="status">' + text(result["missing_attempts"]) + ' planned attempts have no delivered record. No aggregate effect is reported.</p>'
        body += '</section><section><h2>Effects and uncertainty</h2><p>Points and intervals use the declared equal-family analysis. Intervals describe sensitivity to resampling this case bank, not a population. Missing or invalid outcomes are never silently assigned zero.</p>'
        body += table(["Comparison", "Outcome", "Estimate", "Case-bank sensitivity interval", "Invalid-response bounds", "Interpretation"], [[r["left"] + ' versus ' + r["right"], r["outcome"], r["point_estimate"], r["case_bank_sensitivity_interval"], r["full_case_bank_invalid_response_bounds"], r['interval_note']] for r in result.get("contrasts", []) + result.get("secondary_contrasts", [])]) + '</section>'
        body += '<section><h2>What actions produced these outcomes?</h2>' + table(['Condition', 'Offers', 'Accepted offers', 'Accepted standing packages'], [[r[k] for k in ('condition', 'offers', 'accepted_offers', 'accepted_counterpart_packages')] for r in result.get('accounting', [])]) + '<p>Counts retain the original analysis contract. They use valid decisions and the simulated acceptance rule, not human agreements.</p></section>'
        body += '<section><h2>Does wording change the result?</h2><p>Each row keeps related variants together. A sign reversal means observed effects run in opposite directions. Missing outcomes prevent a complete comparison. No automatic “robust” label is assigned.</p>'
        body += table(["Comparison", "Related scenarios", "Complete", "Observed sign reversal", "Validity changed", "Joint effect range"], [[r["left"] + ' versus ' + r["right"], r["family_id"], r["all_variants_comparable"], r["observed_sign_reversal"], r["validity_changes_across_variants"], r["delta_range"]] for r in result.get("variant_sensitivity", [])]) + '</section>'
        body += '<section><h2>Decisions and both parties’ payoffs</h2><p>Agreement-adjusted payoffs apply the simulated acceptance rule and disagreement outcome. They are not observed human bargaining outcomes.</p>'
        body += table(['Case', 'Condition', 'Valid', 'Decision', 'Agreement', 'Buyer outcome', 'Supplier outcome', 'Combined outcome', 'Below buyer minimum'], [[r['case_id'], r['condition'], r['score']['valid'], r['score']['decision'], r['score']['agreement'], r['score'].get('agreement_adjusted_self_payoff'), r['score'].get('agreement_adjusted_counterpart_payoff'), r['score']['agreement_adjusted_joint_gain'], r['score'].get('buyer_below_reservation')] for r in rows])
        body += '<details><summary>Compare proposed payoffs before acceptance</summary><p>Proposed payoffs describe the returned terms, including rejected offers.</p>' + table(['Case', 'Condition', 'Proposed buyer payoff', 'Proposed supplier payoff', 'Proposed combined payoff'], [[r['case_id'], r['condition'], r['score']['proposed_self_payoff'], r['score']['proposed_counterpart_payoff'], r['score']['proposed_joint_gain']] for r in rows]) + '</details></section>'
    body += '<section><h2>Inspect every response and score</h2><p>' + text(len(rows)) + ' attempts retained, including controls, invalid answers and unresolved records. Open a row to inspect its response, score, selection checks and source reference. Full settings and analysis details are in the JSON download.</p>' + ''.join(details(r) for r in rows) + '</section>'
    return shell(title, body)

def shell(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + text(title) + '</title><style>body{margin:0;color:#111;background:#fff;font:17px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px 80px}h1{font-size:36px;line-height:1.2;max-width:850px}h2{font-size:25px;line-height:1.3}h4{margin-bottom:8px}p{max-width:900px}nav{display:flex;gap:24px;flex-wrap:wrap}a{color:inherit;text-underline-offset:4px}section{border-top:1px solid #111;margin-top:36px;padding-top:16px}.status{border-left:4px solid #111;padding:10px 16px;font-weight:650}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px 10px;text-align:left;border-bottom:1px solid #ccc;vertical-align:top}th{border-bottom:2px solid #111}details{border-top:1px solid #ccc;padding:12px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;max-width:100%}a:focus,summary:focus{outline:3px solid #111;outline-offset:4px}@media(max-width:640px){main{padding:24px 18px}h1{font-size:28px}h2{font-size:22px}}</style><main>' + body + '</main></html>'
