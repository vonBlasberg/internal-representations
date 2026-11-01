# Identical-prompt repeatability

The two complete sessions reproduce all 72 responses and 24 baseline-versus-edit contrasts from the original evaluation. They use the same 12 original-wording cases, three conditions, model, prompts, state edit and greedy decoding. `combined/summary.json` identifies the independent audits and their source hashes. `combined/outcomes.json` contains the raw responses, token IDs, scores and source-log hashes; the complete original deliveries and audit outputs are in the two session directories.

The notebook recounts these saved comparisons without a model or network. Text, tokens and recorded processed logits match separately from the behavioral outcome. These repetitions do not create independent scenarios or resolve the main study's wording sensitivity and negative agreement-adjusted payoff findings.

R01 session 1 timed out after 32 completed responses. Its terminal record and original file manifest are in `failed-R01-repeat-1`; its full partial delivery is preserved in the authors' research archive. It is not counted as a complete repeat. The two complete sessions are retained R01 session 2 and the separately approved R02 completion.

The display's **Fresh session 1** is retained R01 session 2. **Fresh session 2** is the R02 completion. Both passing audits used [source/R02/audit.py](source/R02/audit.py), SHA256 `c615885b5cc0343df04626f0c8eb58007bcb698ac275a62c0cee4999a19176a4`. The R01 audit source is the earlier two-session auditor, retained for provenance. A separate implementation checked these records locally; this is distinct from replication or a usability test by an outside researcher.

`source/` preserves the execution, preparation and audit source for inspection, not a standalone launch command. These files refer to the original research archive layout. The default notebook never imports or executes the GPU adapter. A new generation run requires a separately configured and qualified environment.
