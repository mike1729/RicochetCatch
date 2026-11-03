# Runner review resolution — 2026-11-03

The review's three required fixes are implemented. The original freeze and all 31 original source files have been preserved locally. No runtime preflight, model request, login, credential access, or held-out scoring was performed during this revision.

## Changes

| Finding | Resolution |
|---|---|
| Real Code Mode notice invalidates an otherwise valid log | Allow only the exact published message as an `item.completed` error before `turn.started`. Late occurrences, changed text, other envelopes, reconnecting errors, tool calls, unknown events and compaction remain fatal. Startup notices never prove execution denial. |
| Feature flags leave six sub-agent tools exposed | Explicit `agents.enabled = false`. Offline inspection confirms only `exec`, `wait`, `request_user_input`, and `request_user_input_async` remain declared; no multi-agent blocks, skills, or personal instructions are present. |
| Empty assignment transport does not exercise the schema | Use a prescribed all-zero document with range and unsorted-list selectors, scalar and list values, and two overlapping writes. Both schema and parser are checked, including zero final tensors. |
| Neutral instruction mentions tools | Restored exactly `Answer the user's request.` followed by a newline. |
| Tests rely on unfrozen simulator/fixtures | Include `ricochet_sim.py` and every non-cache file below `tests/`, including the real log and attribution. |
| HEAD does not contain the implementation | Commit implementation before the replacement freeze. Record committed-byte comparisons, dirty status and mismatches; never label an unrelated HEAD as the source revision. |
| Replacement freeze might change hidden episodes | Verify the original artifact hashes and reuse both seed files byte-for-byte; link the prior manifest digest. |
| Login instructions are too vague | README now gives the isolated login command, configuration comparison, reboot caveat and mismatch behavior. Login remains an interactive user action. |

The event auditor additionally requires one started and completed turn and rejects duplicate starts/completions or events after completion.

The complete authors' 10,977-byte [Breakout event log](tests/fixtures/authors-breakout/events.jsonl) is pinned to commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Its SHA-256 is `39ab8479e5f2afac77688634ea3fc54bea6a1f0d41a59eeb97d39b5b65ea3270`; [provenance and CC BY 4.0 attribution](tests/fixtures/authors-breakout/provenance.json) are included. Both the auditor and generation validator accept this real tool-free stream in offline tests. Negative fixtures demonstrate that the exception cannot swallow execution attempts or errors occurring during the turn.

## Context and compaction decision

The [official OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) distinguishes the model context budget from the automatic compaction threshold. The [API model page](https://developers.openai.com/api/docs/models/gpt-6-astra) describes API limits; those do not establish the local CLI's effective configuration.

The installed `codex-cli 0.164.0` bundled Astra metadata reports a default context of 272,000 tokens, a maximum of 872,000 and an effective-context percentage of 95. Neither `debug prompt-input` nor the bundled metadata exposes the effective compaction default or clamping behavior. A threshold of 1,000,000 exceeds the advertised CLI maximum. We therefore retain the model defaults rather than assume that override is safe. Any observed compaction still invalidates the planned uncompacted condition.

[Offline review](runs/review-offline-v2/OFFLINE_REVIEW.txt), [final tool inventory](runs/review-offline-v2/current-final.inventory.json), and [selected bundled metadata](runs/review-offline-v2/bundled-model-catalog.json) archive the evidence. The preview route differs from real `exec`; it is not runtime execution-denial validation. Runtime preflight remains pending.

## Verification and scope

The complete suite passes **235 tests**. Tests include the real authors' stream, exact startup-notice timing/envelope checks, all-zero nonempty transport, suppression settings, single-turn lifecycle, source provenance against real temporary Git repositories, and preservation of the original hidden seed commitments.

Game mechanics, tensor grammar, network dimensions, baselines, calibration results, held-out distribution and primary success criterion are unchanged. The claim remains useful play. We have not added a bounce-prediction claim or a new secondary success requirement. The optional float64 sensitivity analysis remains unimplemented. Reconnecting errors remain explicit audit failures and never trigger automatic generation retries.

The active replacement is [runs/frozen-v2/manifest.json](runs/frozen-v2/manifest.json). Its hash and source revision are recorded in [FREEZE_COMMITMENT.json](FREEZE_COMMITMENT.json) after the source commit. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) contains the current handoff and pending commands. Private seed files remain under ignored `runs/`; no files were pushed remotely.
