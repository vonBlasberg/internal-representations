# internal-representations

## Beyond the Answer

**Does an internal edit change an AI's decision, who benefits, and does prompting do the same thing?**

Open the [notebook](START_HERE.ipynb) to follow the study, or download and open the [HTML preview](PREVIEW.html). Both include the results and figures, so you can read them without installing anything. To recalculate the analysis, follow **Run locally** below. No model download, API key or GPU is needed.

The selected edit changed 13 of 33 valid decisions from acceptance to counteroffering while keeping the complete starting prompt unchanged. Prompting changed 12 of 31 valid decisions in the same direction. An opposite-direction edit shifted baseline decisions toward acceptance, while random edits changed little. The edited offers requested more for the buyer and less for the supplier, who rejected them. The prototype shows how to distinguish behavioral influence from payoff improvement, wording robustness and execution stability.

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

The tested route is macOS with Python 3.12.10. See the [reproducibility guide](docs/REPRODUCIBILITY.md) for platform notes, completed checks and how replay preserves the evidence. Installation downloads notebook libraries; the analysis itself uses bundled records only.

## Supporting files

- [exports/](exports/): downloadable results, including every attempted response.
- [replay/](replay/): cases, original responses, analysis code, settings and verification records. These support reproduction and include unsuccessful attempts.
- [docs/](docs/REPRODUCIBILITY.md): reproducibility, [cases and scoring](docs/SCHEMAS_AND_EXPORTS.md), [advanced GPU setup](docs/GPU_SETUP.md) and [anonymization](docs/ANONYMIZATION.md).

The notebook includes an editable allocation exercise and the development roadmap. Its build script is under `scripts/`; reviewers can read or rerun the notebook without rebuilding it.

The proposal, Technical Appendix and cover letter are submitted separately. The repository contains the notebook, code and supporting evidence.

## Credentials and license

CPU replay needs no credentials. [Advanced GPU setup](docs/GPU_SETUP.md) explains the separate generation route; [.env.example](.env.example) supplies empty placeholders. Keep real credentials outside notebook cells, outputs and Git.

Original project code and documentation use the [MIT license](LICENSE). Third-party code retains its [own notices](replay/third_party/README.md); model weights are obtained from their licensed source.

