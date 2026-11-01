# Reproducing and adapting the notebook

## Current evidence and capabilities

The prototype supports candidate comparison, saved-response inspection, recalculation, complete exports and an editable synthetic allocation exercise. The notebook's roadmap covers planned own-study execution and measurement.

- **Main comparison:** 324 completed responses across 36 cases and nine conditions, independently verified. [Execution audit](../replay/comparison/audit.json).
- **Repeatability:** two complete fresh sessions, 72 responses and 24 baseline-versus-edit contrasts, all matching the originals. [Combined audit](../replay/repeatability/combined/summary.json). The earlier timeout remains a failed attempt.
- **Automated checks:** source integrity, candidate selection, every attempted response, scoring, controls, exports and full CPU replay. [Local check record](../verification/LOCAL_CHECKS.md) and [release execution receipt](../verification/verified_notebook.json).
- **Further development:** own-study GPU execution, selection across development families, validated measurement and independent researcher reuse tests. Pilot selection remains open.

These completed audits establish the evidence status. Historical preparation labels remain part of the archived record.

## Saved-data analysis

Follow the installation, replay and test commands in the [README](../README.md). The notebook verifies `source_manifest.json`, then uses the retained analysis code to:

1. rescore calibration responses and reproduce eligibility and ranking;
2. reconcile all 324 new-case attempts with the declared conditions and saved logs;
3. reproduce validity, decisions, payoffs, control comparisons and case-bank sensitivity;
4. compare scientific results with the verified saved report and recalculate payoffs with a separate implementation;
5. recount both complete repeat sessions from their responses and audited comparisons;
6. regenerate the figures and exports.

Research inputs under `replay/` stay fixed. Each replay retains its calculations in a new `outputs/notebook-*` directory. Reader downloads under `exports/`, the figures and the executed notebook are regenerated. Tests use temporary export locations and leave checked-in results unchanged.

`exports/findings.json` adds the completed admission status and its source to the historical report. The earlier pending label remains in `historical_evidence_label`, the retained CSV and the saved report. The [export guide](SCHEMAS_AND_EXPORTS.md) explains inherited status flags. `exports/descriptive_comparisons.json` adds the descriptive prompting-minus-baseline calculation without changing the frozen contrast hierarchy.

Run verification in normal Python mode. The active replay entry points reject `python -O` and `PYTHONOPTIMIZE`, which disable required checks. Each fresh replay writes `verification/notebook_execution*.json`; `verification/verified_notebook.json` retains the release execution receipt. These are local software checks. An outside researcher reproduction and usability walkthrough are separate checks.

## Environment and platform notes

The tested environment is Python 3.12.10 on macOS. `requirements-lock.txt` pins the library versions; `requirements.txt` lists the direct dependencies. Installation was tested in a separate environment on the same computer, not by an independent researcher.

Linux/WSL uses the README commands but has not received an independent platform test. On Windows, use a short WSL extraction location because the retained provenance includes long paths. Native Windows installation and long-path behavior remain untested.

## What the evidence establishes

The comparison covers 36 authored cases in 12 families, one model, one selected direction and fixed settings. The edit changes decisions with the same input prompt; reversal and random controls test direction dependence. Both parties' requested and agreement-adjusted payoffs show that the intended shared gains did not transfer. Missing-outcome bounds retain invalid answers; family-bootstrap percentile intervals describe case-bank sensitivity, not population precision.

Two fresh sessions reproduce all 72 responses and 24 baseline-versus-edit contrasts for identical prompts and greedy decoding. The notebook recounts behavior, text and token agreement and the recorded logit-match indicators. This checks execution stability; it adds no new independent scenarios. The earlier timed-out session and its completed-prefix records remain archived and excluded from complete-repeat totals. Compact provenance records are included under `replay/repeatability/provenance/`.

## Generation and reuse

Steerability, Inspect AI and the qualified Transformers/PyTorch execution path produced the original research records. CPU replay reads those records; it does not invoke the model-generation pipeline. The two earlier synthetic integration tasks exercised provider conversion and Inspect logging/scoring through authored responses, not through a new model run.

The request, settings and source derivative under `replay/generation_inputs/` document the completed execution. [GPU setup](GPU_SETUP.md) distinguishes that source record from a portable, newly qualified GPU launcher. Recorded execution verdicts describe audits of the private originals; `source_manifest.json` checks the distributed derivatives. Identifier redactions and updated checksum references are documented in [anonymization](ANONYMIZATION.md).

For adaptation, start with the editable allocation cell and the [case and scoring guide](SCHEMAS_AND_EXPORTS.md). Stage 2 extends this to own-study execution and measurement. An independent researcher walkthrough remains the practical reuse test, separate from the completed automated analysis checks.

For maintenance, `python -B scripts/build_notebook.py` rebuilds the notebook's cells from its source. Then run `python -B execute_notebook.py` to recalculate the saved evidence and refresh the reading copy. Reading or running the notebook does not require this build step.
