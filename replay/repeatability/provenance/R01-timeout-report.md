# R01 terminal status: first session timed out

October 31, 2026, 21:32 UTC / 22:32 Europe/Madrid. Same-call retrieval and independent local checks only. No model was loaded or generation launched, retried or cancelled during closeout.

The planned two-session repeatability check is incomplete. Session 1 reached its configured 2,400-second child-process limit before completing 36 responses. The containing function had a separate 2,700-second limit. Session 2 completed all 36 responses.

| Record | Session 1 | Session 2 |
| --- | --- | --- |
| Call | `fc-01M69VE74MZA67N7VWWDCQYYVA` | `fc-01M69VE8N1ZQ3YFCEXR54A5YS6` |
| Process return code | -15 | 0 |
| Timed out | true | false |
| Committed files independently hash-checked | 272 | 301 |
| Completed trial receipts / planned responses | 32 / 36 | 36 / 36 |
| Original Inspect log statuses | 32 success, 1 started | 36 success |
| Attempt records | 33 | 36 |
| Final worker result and outer summary | Absent | Present |

The interrupted attempt has index 32, the 33rd attempt: `negotiation-same-prompt-repeatability-2026-10-31-R01-repeat-1--prospective-20261031-routers-base--internal`. Its Inspect status remains `started` and it has no completed verification receipt. The three final meters-case conditions were not started. Incomplete responses receive no admitted outcome.

Exact session 1 terminal record:

```json
{"automatic_retry":false,"plan_sha256":"a3ca63626cf9db48c26f9756eb94956970ad09384306eaf09981e212bace359b","process":{"returncode":-15,"timed_out":true},"run_id":"negotiation-same-prompt-repeatability-2026-10-31-R01-repeat-1","status":"FAILED_OR_INDETERMINATE_NO_RETRY"}
```

Both deliveries are retained in their exclusive-create `execution/R01/launch/<session>/returned` directories. A separate standard-library check verified all 573 file sizes and SHA256 hashes, exact manifest membership, call identities, frozen transport and plan identities, matching returned/manifest/terminal records, retrieval manifest hashes, attempt counts and original Inspect log statuses. This check performed no behavioral comparison.

Manifest SHA256 values:

- Session 1: `7138ada5acc82e9ad026f2f5b5780fd1971d497747cb34bf8c153fe3cdf06529`
- Session 2: `68aef281dea934e40f415d9089e874d9d3239470140610d8538a6fc0151fbcb6`

The unchanged preparation check passed for all 454 archived A03 payload files, 36 reference rows and 72 exact planned inputs. The frozen adapter, auditor, plan and transport identities matched before retrieval. The source-archive and preparation checks also passed on entry to the unchanged independent auditor.

That auditor ran once offline with the documented snapshot and fresh target `verification/R01-terminal-A01`. It stopped at `audit.py:112` with exit code 1:

```text
ValueError: Failed or incomplete outer execution; preserve diagnostics, no result admission
```

No passing audit directory or comparison result was created. Session 2's delivery and outer completion are verified; its full per-response scientific audit has not yet been performed separately. No R01 repeatability percentage is admitted.

The original stderr is preserved, including reference PyTorch fallback warnings for `causal_conv1d` and `flash-linear-attention`. This closeout does not establish why session 1 took longer than session 2 and changes no runtime component.

This is a time-limit failure, not a negative repeatability finding. The original 324-response results, including wording sensitivity and negative agreement-adjusted payoff findings, remain unchanged. The owner subsequently authorized one fresh full 36-response session with a longer operational time allowance. That distinct completion must retain this failure, use unchanged scientific inputs/settings, and receive separate source-bound verification. Filling only the four unfinished responses would not reproduce an uninterrupted fresh session. No automatic retry or publication is authorized.
