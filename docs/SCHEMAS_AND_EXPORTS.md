# Cases, scoring and exports

## Where to make a study-specific change

| Part | Current file or function |
|---|---|
| Authored negotiation cases | [cases.json](../replay/comparison/delivery/cases.json): `case_id`, `family_id`, `variant_kind`, issue options and utilities, standing package, reservations, disagreement values, and messages. |
| Case contract and deterministic payoffs | [negotiation.py](../replay/code/reused/negotiation-steering/analysis/negotiation.py): `validate_case`, `payoffs`, and response scoring. |
| Response contract | Exactly `decision` (`accept` or `offer`) and `package`, with one allowed option per issue. Acceptance repeats the standing package. The parser retains other labels as invalid. |
| Saved-trial structure | [behavioral_trial.schema.json](../replay/code/reused/negotiation-steering/contracts/behavioral_trial.schema.json); fixed messages are in [original-plan.json](../replay/comparison/delivery/original-plan.json). |
| Reusable synthetic interface | [workflow.py](../replay/code/review_package/workflow.py): `Study`, `OutcomeScorer.score(case, raw)`, `AllocationScorer`, `NegotiationScorer`, `evaluate_saved`. The shared interface is currently qualified for model-free examples. |
| Outcome definitions | [prospective_validation_v2.json](../replay/protocol/prospective_validation_v2.json) records the pre-generation outcome amendment. [scoring.py](../replay/code/review_package/scoring.py) computes valid negotiation outcomes. |
| Notebook presentation | `notebook_support.py` supplies tables, figures, exact saved prompts and the editable example. |

Copy cases and define a scorer for an adaptation; leave the empirical example intact. A new intervention study also needs compatible model access, qualification, development-case selection and a fixed evaluation. Measurement and bias-auditing examples are planned uses, not supported upload formats here.

## Reading `exports/evaluation_attempts.csv`

One row is one attempted case/condition response. All 324 rows are included, including nine invalid answers.

| Fields | Meaning |
|---|---|
| `trial_id`, `case_id`, `family_id`, `condition` | Attempt identity, scenario, dependent family and condition. |
| `execution_status`, `valid`, `raw_response` | Technical completion, scoring validity and original answer. Invalid outcomes remain missing, not zero. |
| `decision` | Parsed `accept` or `offer`; the package and parsing errors are in the corresponding `score` object in `findings.json`. |
| `buyer_payoff`, `supplier_payoff`, `joint_payoff` | Proposed-package values before the supplier acceptance rule. |
| `agreement_adjusted_buyer_payoff`, `agreement_adjusted_supplier_payoff`, `agreement_adjusted_joint_payoff` | Values after the rule; rejection uses the case's disagreement values. |
| `buyer_below_reservation` | Whether the proposed package's buyer payoff falls below the stated minimum, before applying the supplier acceptance rule. This is not a flag for rejection-adjusted payoff. |
| `site`, `coefficient`, `direction_sha256` | Edited location, multiplier and saved direction identity. |
| `source_log_path`, `source_sample_id`, `source_log_sha256` | Original record and sample. Paths are relative to the declared evidence bundle. |
| `csv_escaped_fields` | Fields escaped for spreadsheet viewing; original text remains in JSON. |

`calibration_attempts.csv` additionally reports eligibility, rejection reasons, tie-breaking, direction norms, cosine similarity, intended displacement and measured displacement summaries. These are historical selection results, not a new search. `party_payoff_differences.csv` reports equal-family means over valid paired subsets, separately from full-bank estimates and missing-outcome bounds.

`findings.json` contains full scores, contrasts, missing-outcome bounds and family-bootstrap percentile intervals where defined. Its admission record links the completed audit. `repeatability_outputs.csv` keeps all 72 outputs and separates behavioral, textual, token and recorded-logit agreement.

## Complete field mapping

The CSV contains 51 columns. The exporter in [cli.py](../replay/code/review_package/cli.py), function `write_new`, is the exact CSV-to-JSON mapping. The groups not expanded above are:

| Fields | Source and meaning |
|---|---|
| `arm_id`, `phase`, `fixture_only`, `evidence_label`, `record_origin` | Trial/analysis metadata. All empirical evaluation rows have `fixture_only` false. See historical status meanings below. |
| `model_id`, `model_revision`, `publication_sha256`, `protocol_sha256` | Model identity and source bindings recorded with the analysis. |
| `historically_selected`, `historical_rule_qualified`, `historical_rank` | Historical candidate-selection fields. They do not select candidates on evaluation cases. |
| `selection_status`, `selection_rule_json`, `selection_scope`, `selection_reasons`, `eligibility_checks`, `ranking_rule` | Selection metadata and JSON-encoded checks. Evaluation rows report no new selection. |
| `direction_norm`, `cosine_with_intended`, `intended_displacement_l2`, `intended_projection_on_target` | Direction geometry from the recorded candidate definition. |
| `realized_displacement_l2_min`, `realized_displacement_l2_max`, `realized_displacement_observations`, `realized_displacement_source`, `realized_displacement_source_sha256`, `geometry_status` | Observed edit magnitudes and their source, or an explicit unavailable status. |
| `primary_outcome` | The pre-generation amended outcome, `agreement_adjusted_joint_gain`. |

In JSON, these values come from each trial, its `score`, `source`, `edit` or `geometry` object, or the top-level analysis metadata. The payoff names in CSV map `buyer` to `self` and `supplier` to `counterpart`. JSON retains the exact package, parsing errors and full response. Fields listed in `csv_escaped_fields` received a leading apostrophe for safe spreadsheet viewing; use JSON for original strings.

`descriptive_comparisons.json` reports prompting minus baseline on agreement-adjusted joint payoff: equal weights across families and across variants within a family, with invalid responses bounded over feasible accepted-package and disagreement payoffs. This is descriptive and leaves the original primary and secondary contrasts unchanged.

## Historical status meanings

The 324-row CSV retains `SAVED_RESULTS_PENDING_INDEPENDENT_ADMISSION` from the original analysis export. The current evidence status is `FIXED_COMPARISON_INDEPENDENTLY_VERIFIED`, linked to [the completed audit](../replay/comparison/audit.json) in `findings.json`.

The inherited JSON type `REVIEW_DRAFT_NINE_CONDITION_ANALYSIS` identifies the analysis schema. `qualification_verified: false` means that analysis routine did not itself verify model qualification. `release_approved: false` means it did not grant publication approval. Neither reverses the completed execution audit. `case_independence_verified: false` is retained because variants within a family are dependent. These historical fields are explained rather than changed to manufacture a different execution record.
