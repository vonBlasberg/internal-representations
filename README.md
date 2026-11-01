# internal-representations

## Beyond the Answer

**Does an internal edit change an AI's decision, who benefits, and does prompting do the same thing?**

Open [START_HERE.ipynb](START_HERE.ipynb), or download and open [PREVIEW.html](PREVIEW.html). The notebook uses independently verified responses from a negotiation study. Its figures and tables recalculate saved evidence on a CPU; no model download, API key or GPU is needed.

The selected edit changed 13 of 33 valid decisions from acceptance to counteroffering with the same input prompt. Reversal shifted decisions back; random edits changed little. The edited offers requested more for the buyer and less for the supplier, who rejected them. The notebook distinguishes directional behavioral influence from payoff improvement, wording robustness and execution stability.

### A short route through the notebook

1. Inspect the construction pair and candidate comparison, including the selected and rejected settings.
2. Follow one case from prompt and response to both parties' payoffs, then examine all cases and controls.
3. Check wording sensitivity and same-prompt repetition.
4. Change the budget or authored responses in the synthetic allocation exercise and rerun its cell.

This is a Stage 1 proof of concept for a broader practical guide and research tool. Stage 2 will support researchers' own cases and scorers, selection across development cases, held-out evaluation and measurement from recorded states. Negotiation is the first example, not the intended limit. The notebook's roadmap gives the development steps and reuse tests.

## Run locally

Install **Python 3.12** from [python.org](https://www.python.org/downloads/) or your usual environment manager. Select that interpreter in your shell. On macOS or Linux, check `python3 --version` before creating the environment; it should report 3.12.x.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m jupyterlab START_HERE.ipynb
```

Choose **Run → Run All Cells**. To replay and test from a terminal:

```sh
.venv/bin/python -B execute_notebook.py
.venv/bin/python -B -m unittest discover -s tests -v
```

The tested route is macOS with Python 3.12.10. Linux/WSL uses the commands above but has not received an independent platform test. The retained provenance contains long paths: on Windows use a short WSL extraction location rather than Windows Explorer's default ZIP extraction. Native Windows installation and long-path behavior remain untested.

Installation downloads ordinary notebook libraries; replay uses bundled records only. `requirements-lock.txt` pins the tested environment and `requirements.txt` lists its direct dependencies. Every replay retains a new calculation folder under `outputs/`, then refreshes the reader-facing `exports/` and figures. Original empirical inputs stay unchanged.

## Contents and adaptation

- Executed notebook, five figures, every evaluation attempt and the historical candidate comparisons.
- Scoring/analysis code, original response records, settings and completed audits.
- An editable model-free allocation example using the shared saved-response scoring interface.
- [Case, response and export guide](SCHEMAS_AND_EXPORTS.md), [reproducibility guide](REPRODUCIBILITY.md), and [current artifact status](ARTIFACT_STATUS.md).

The proposal, technical appendix and cover letter are submitted separately. The repository contains the notebook, code and supporting evidence.

## Credentials and license

CPU replay needs no credentials. [GPU_SETUP.md](GPU_SETUP.md) explains the separate generation route; [.env.example](.env.example) supplies empty placeholders. Keep real credentials in provider configuration or an ignored local environment, outside notebook cells and outputs.

Original project code and documentation use the [MIT license](LICENSE). Third-party code retains its [own notices](replay/third_party/README.md); model weights are not bundled and retain their model license. The public copy's identifier redactions and checksum scope are described in `ANONYMIZATION.md`. The research archive preserves the originals.

