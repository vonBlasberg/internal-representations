# Retained analysis and integration code

Use the repository README for installation and replay. These modules were developed for the completed study; their historical preparation labels do not supersede the completed audits linked in `../../ARTIFACT_STATUS.md`.

- `review_package/history.py` recalculates saved calibration eligibility and ranking.
- `comparison.py`, `outcomes.py` and `sensitivity.py` account for every condition and outcome.
- `scoring.py` applies the retained decision-only parser and payoff rules.
- `workflow.py` supplies the synthetic study/scorer interface used in the notebook.
- `inputs.py`, `admission.py` and `recording.py` bind study inputs and recorded attempts.
- `inspect_bridge.py` and `recording_check.py` tested Steerability provider conversion and Inspect logging/scoring with authored responses. CPU notebook replay does not run them.

Scientific scoring and selection logic is retained. Anonymous-copy hash bindings are documented in `../../ANONYMIZATION.md`. The original execution used the separately retained GPU source; this analysis directory is not a general-purpose model launcher.
