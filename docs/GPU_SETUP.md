# Credentials and model generation

## Reading and rerunning the notebook

No API key, account or GPU is required. Follow the [README](../README.md) and install `requirements-lock.txt` for the tested CPU environment.

## A separate model-generation run

The original study used Modal compute and an open-weight Qwen checkpoint, with Steerability/Inspect recording and the qualified Transformers/PyTorch state-editing path. A key for an ordinary chat-completion API is not a substitute: that service generally does not expose this internal state.

The [.env.example](../.env.example) file contains empty placeholders:

| Variable | Purpose |
|---|---|
| `MODAL_TOKEN_ID` | Identifies the Modal credential used for separately approved compute. |
| `MODAL_TOKEN_SECRET` | Authenticates that credential. Treat it as a password. |
| `HF_TOKEN` | Optional authentication if checkpoint access requires it. CPU replay never needs it. |

Use the provider's normal local authentication or environment variables. If you make a local `.env` copy, it is excluded from Git. The notebook intentionally does not load it. Do not paste secrets into notebook source or output, and never send them to another researcher with the repository.

The execution source and exact request in `replay/generation_inputs/` document the completed run. The distributed source ZIP is an identifier-redacted derivative, separate from the frozen executable payload in the private archive. To generate again, configure a new request and qualify the model, state edits and restoration in the intended environment. Retain the model revision, directions, prompts, scoring and decoding for a replication; specify GPU memory, time and cost limits for that execution. The original one-use requests are historical records.

The local notebook supports complete analysis replay. A portable, one-command GPU launcher and arbitrary own-question support are not claimed as finished features. The roadmap separates those extensions from Stage 1.
