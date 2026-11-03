# Weight-space programming: implementation brief

Revised 2026-11-03 after independent review. Status: the production game, network harness, evaluator, renderer, and guarded isolated runner are implemented and tested; production development calibration reproduces the review results. Runtime preflight and the Astra submission remain pending. No experimental Astra request has been made. See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for the implementation handoff.

## Objective and scope

Test whether GPT-6 Astra can construct the numerical weights of a useful neural policy for a newly implemented game, from its specification and source, without executing code, playing the game, training, or receiving performance feedback.

The first experiment is **one game, one fresh Astra submission, one frozen network, and 256 held-out evaluation episodes**. A successful artifact is evidence of capability on this task. Repeated submissions would measure reliability; a difficulty ladder would investigate limits. Neither is required for this first experiment.

Implement **Ricochet Catch**, specified in [GAME_DESIGN.md](GAME_DESIGN.md): a paddle catches gems moving vertically or diagonally, with wall reflections and bombs to avoid. This selects the most involved falling-object version discussed. Color-changing catches, multiple levels, PPO training, and a Python-controller-to-weights condition are outside the initial scope.

The implementing agent should build and validate the local game, renderer, parser, evaluation harness, and isolated generation runner. The runner must default to a dry run; a separate explicit invocation starts an Astra request. Do not include an automatic retry, repair, or optimization loop.

## What the experiment can establish

The intended conclusion is: “Astra produced a fixed neural network that achieved useful performance on this specified game, without tools or execution feedback.”

The conclusion concerns the resulting network across episodes, not one lucky playthrough. It does not imply reliable success across independent submissions, superiority to RL, or a general ability to write arbitrary neural programs.

Success also does not establish that the network computes eventual landing positions, predicts complete bounce trajectories, or plans several steps ahead. A one-step controller already clears the proposed performance threshold in review simulations. The experiment tests construction of a useful neural policy; identifying its internal algorithm would require separate analysis.

Create the exact implementation privately and record its provenance. Do not claim that the mechanics are unprecedented or absent from training. A model alias is not an immutable checkpoint. Only a verifiably fixed snapshot predating creation supports the stronger claim that these exact source bytes could not have been in that checkpoint's training data. Record the requested and any returned model identifiers; never invent a snapshot date.

## Decisions fixed for the initial experiment

| Item | Decision |
|---|---|
| Game | One Ricochet Catch environment |
| Policy | `449 -> 32 -> 32 -> 3`, tanh hidden layers; 15,555 parameters |
| Initialization | Every weight and bias is zero before applying the submission |
| Actions | Deterministic argmax; index order stay, left, right |
| Model | Explicitly request `gpt-6-astra`; no fallback to another model |
| Reasoning effort | `xhigh` |
| Reasoning summaries | Request `detailed`; preserve returned summaries as supplementary provenance |
| Generation | One fresh conversation, one final response, no feedback |
| Tools | No execution, search, file tools, connectors, delegation, or other tools |
| Output | Declarative JSON tensor assignments; no executable program |
| Evaluation | 256 hidden episode tapes, at most 256 ticks each |
| Development | Separate 256 episode tapes; calibration happens before generation |
| Primary outcome | Mean undiscounted gem catches per episode |

This adapts the direct-construction idea of the authors' [Experiment 5](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp5_write_from_scratch/README.md). It is not a numerical replication: our game, architecture, episode budget, and deterministic actions differ. Do not add their stochastic action sampling or sticky actions to this first version.

## Development calibration and baselines

All baselines receive only information present in the network observation. None may read upcoming spawn-tape entries, access the environment RNG, or receive evaluator-only reflection labels. Their implementations and results stay out of Astra's prompt.

Implement these baselines:

1. **Stay:** action 0 forever; also verifies the all-zero network's behavior.
2. **Random:** independently sample each of the three actions uniformly. Use a separate recorded RNG from the scene generator.
3. **Reactive chase:** target the visible gem with smallest vertical distance to the bottom. This target is unique because objects occupy distinct rows. Move toward its current column. Stay if there is no gem. Ignore bombs.
4. **Reactive chase with safety:** veto any action whose next paddle cell is hit by a bomb on the next tick. Of the safe actions choose one minimizing distance to the reactive chase target's current column; ties follow action index order. With no gem, choose the smallest safe action index. Assert that a safe action exists: at least two distinct next paddle cells are reachable and at most one bomb can arrive per tick.
5. **One-step lead with safety:** select the same lowest gem, but target `clip(column + vx, 0, 7)`. Choose the safe action minimizing distance to that target, ties by action index; with no gem choose the smallest safe action index. Intentionally clip the gem target rather than reflecting it. Compute imminent bomb collisions with the actual one-step reflection rule, using only bombs on row 6. Do not compute eventual gem landing positions or use a landing lookup table.
6. **Predictive greedy with safety:** predict each visible gem's arrival time `k = 7 - row` and landing column from the movement rules. Retain gems with `abs(landing_column - paddle_column) <= k`. Select by `(k, horizontal_distance, landing_column)` and choose the safe action minimizing distance to its landing column, ties by action index. With no eligible gem, choose the smallest safe action index. This comparison measures how far a simple predictive heuristic gets.
7. **Predictive reference:** a receding-horizon dynamic program over the next seven ticks and eight paddle positions. Forecast bottom arrivals of currently visible objects using the movement rules. Maximize gem catches while forbidding bomb collisions; new, unseen spawns are excluded from its forecast. Its first action is executed, then the calculation is repeated next tick. Terminal value is zero, horizon is `min(7, remaining_ticks)`, and ties use the smallest first-action index.

The reference is a competent heuristic, not an omniscient or proven optimal policy. Its horizon covers all currently visible objects that can arrive before timeout. Review simulations put predictive greedy close to the reference; do not assume dynamic programming supplies a large advantage. Prediction helper functions belong to the baseline module, not the source given to Astra.

Before asking Astra:

- Pass the mechanics fixtures in GAME_DESIGN.md.
- Run all baselines on development tapes and inspect representative replays.
- The reference should average at least 10 catches per episode and beat reactive chase with safety by at least 20% in mean catches. These are engineering checks for scoring opportunities and a useful performance range; they do not establish that all passing policies require multi-step prediction.
- The reference should also beat the strongest of stay, random, and reactive chase by at least 10 mean catches. Otherwise the proposed success threshold below will not distinguish useful competence clearly.
- If these checks fail, inspect the implementation and tune only the disclosed spawn probability, bomb fraction, and velocity mixture. Retain nonzero probability for bombs and all three velocities, with at least half of spawns moving diagonally. Document every calibration change. Do not run Astra to tune difficulty. If mechanics or architecture must change, update both briefs and version the protocol first.
- Once calibration passes, freeze the constants and all baseline implementations. Select the **trivial comparator** as the best of stay, random, and reactive chase by development mean return; ties use the listed order. Do not reselect it on test results.

Optional PPO comparisons can be added later as a separately documented study. A weak or unfinished PPO run is not evidence of superior weight programming.

### Review calibration already performed

[ricochet_sim.py](ricochet_sim.py) is an independent pure-Python review aid. Its mechanics fixtures pass. Using its default constants and seeds 0–255 gives these means:

| Controller | Mean catches |
|---|---:|
| Reactive chase | 6.1914 |
| Reactive chase with safety | 21.8125 |
| One-step lead with safety | 44.0781 |
| Predictive greedy with safety | 56.7539 |
| Predictive reference | 57.5703 |

The one-step controller was evaluated separately against the simulator with its landing table cleared; it had zero bomb deaths. The halfway threshold was 31.8809, and the one-sided 95% bootstrap lower bound on its paired margin was 11.7578 (10,000 resamples; review bootstrap seed 12345). These are development findings, not Astra results or held-out results for the production implementation. Implement the exact baseline above and reproduce the comparison when validating production mechanics. The review script does not yet include this additional controller.

Retain the initial spawn and bomb probabilities unless production calibration reveals a bug or fails the declared checks. Keep the original trivial-comparator selection and primary threshold; the new one-step and predictive-greedy policies are descriptive comparisons, not additional passing requirements. Review seeds already inspected must not become the hidden test set.

The simulator's object tuples include a diagnostic reflection flag. Its supplied controllers ignore that field, but it must be excluded from the production policy interface. [robustness.py](robustness.py) corrupts a controller's landing lookup table, including bomb avoidance; it does not measure robustness to neural-weight errors. Both scripts and their tables must stay out of the model prompt.

## Evaluation and the meaning of “working”

Use the same 256 held-out spawn tapes for Astra and every frozen baseline. The primary comparisons remain the development-selected trivial comparator and the reference; do not reselect comparators from test results. Early termination is allowed; the unused tail of a tape remains unchanged. Use distinct recorded action seeds for random policies.

Adopt this practical, precommitted success criterion: Astra must close more than half the performance gap from the trivial comparator to the reference. For episode `i`, define:

`D_i = R_astra_i - 0.5 * (R_reference_i + R_trivial_i)`.

Resample the 256 episode indices jointly across policies 10,000 times, using a bootstrap RNG seed recorded before generation. The one-sided 95% lower percentile bound is the 5th percentile of the resampled mean `D`. **Declare the intended demonstration successful if that bound is greater than zero**, and the submission and no-tools audit are valid.

Also require a positive one-sided 95% lower bound for the paired mean `R_astra - R_trivial`. If the reference fails to outperform the trivial comparator in held-out mean return, flag the reference calibration as failed and withhold the planned success declaration; retain the raw outcomes without retuning or replacing episodes.

The halfway threshold is an operational definition of useful play, not a universal scientific boundary or a measure of compilation fidelity. Publish raw scores and uncertainty even if it is missed. Failure to clear it does not prove Astra cannot write weights; modest performance can support a narrower claim. Confidence-interval width depends on the submitted policy's observed variability; 256 episodes does not guarantee a particular margin of error.

Report:

- mean, median, standard deviation, and a paired comparison with the two frozen comparators;
- a two-sided 95% bootstrap interval for each mean return;
- catches, missed gems, fraction of episodes ending in a bomb collision, and mean episode length;
- catch fractions separately for vertical gems, diagonal gems with no wall reflection, and gems whose full fall includes a reflection. Use full-tape eligible gems as denominators: those that would reach row 7 by the horizon if the episode continued; count opportunities after death as uncaught. Compute these evaluator-only labels independently of policy termination and exclude spawns that cannot arrive before timeout;
- optionally, catches divided by gems actually reaching the bottom before termination, explicitly labeled survival-conditional; report zero-denominator episodes;
- parameter count, nonzero count, tensor magnitude summaries, parse status, and tool-audit status;
- the requested/returned model identity, CLI version, reasoning effort, generation time, token usage if returned, and number of attempted submissions.

Do not use survival alone as success, choose the best test episodes, or interpret the 256 episodes as 256 independent model-generation attempts.

Freeze a small set of single-object diagnostic probes before generation, with reachable states and expected outcomes defined from the mechanics. Include wall approaches, imminent gem catches, and imminent bomb avoidance; label deliberately synthetic states separately. Respect the one-object-per-row invariant. Probe scores and bounce catch fractions describe behavior, not proof of a particular internal algorithm, and cannot rescue a failed primary result.

Keep float32 as the official evaluation. Record CPU architecture, NumPy version and BLAS implementation/thread settings. Log the count of decisions with top-two logit gap at most `1e-6 * max(1, max(abs(logits)))`; freeze this diagnostic tolerance in advance. After the official evaluation, optionally run a separately labeled float64 sensitivity check using the same stored float32 weights cast to float64. Report action disagreements on recorded float32 observations and return changes on the same tapes. Never select the more favorable arithmetic as the primary result.

## Episode seeds and freeze procedure

Generate disjoint development and test seed lists locally. Development seeds can be used freely during implementation. Create and commit to a private test-seed manifest before generation; keep it outside the model working directory and prompt bundle. The test generator distribution is public; individual seeds and tapes are hidden.

Before generation, record SHA-256 hashes of:

- environment and observation source, frozen game config, game specification;
- network inference code, assignment parser, JSON Schema, baselines;
- prompt, neutral instructions, CLI overrides and their version checks;
- development/test manifests and bootstrap RNG seed;
- dependency lockfile, CPU/BLAS metadata, diagnostic probes, the independent simulator and every test fixture, and the implementation revision when version control is available. Record a source revision only when its committed bytes match every frozen file; record dirty state separately. Track the final manifest hash without committing private seed data.

Run the frozen test suite once after the submitted tensors have been parsed and hashed. Do not inspect test outcomes during calibration. If a real implementation bug requires changes after generation, preserve the original artifacts and label the changed protocol and rerun explicitly.

## Astra prompt bundle

Generate a self-contained text prompt containing only:

1. the objective: maximize expected gem catches under the given rules;
2. the frozen game rules and constants;
3. exact environment transition, reset/spawn, reward, and observation source;
4. the network equation, parameter shapes, dtype, input indexing, actions and tie rule;
5. the output JSON contract and neutral serialization examples;
6. evaluation distribution, horizon, deterministic action selection, and the fact that seeds are hidden;
7. the explicit restriction to reasoning and numerical assignments, with no tools, execution, training, rollouts, search, or feedback.

Exclude baseline source, intercept formulas or planning helpers, review scripts, diagnostic labels/probes, calibrated scores, test seeds/tapes, our conversations, and this experiment plan. In particular, **do not simply paste GAME_DESIGN.md wholesale**: its calibration fixtures are for the implementing agent. Author a clean game-rules section from sections 1–6 of that document, plus the actual source, and audit the resulting bundle for solution leakage. Serialization hints are allowed: for row `r` and channel `ch`, the eight column indices are `56*r + ch + 7*c` for `c = 0..7`, or a range with start `56*r + ch`, stop `56*r + 56`, step `7`. These describe layout only.

Prompt skeleton:

> Construct the weights of the fixed neural policy below to maximize expected gem catches in this game. All tensors initially contain zeros. Reason from the supplied mechanics and source. You have no tools and will receive no rollout or performance feedback. Do not provide a training algorithm or a runtime controller. Return exactly the JSON object required by the assignment schema; its numerical assignments will be applied once, and only the stated network forward pass will choose actions. Include a short explanation in the JSON explanation field.

No Python-policy-first stage is included. If explored later, label it as a different condition.

## Tensor assignment format

Use a small declarative format, `tensor-assignments-v2`. This revision allows ordered overwrites and unsorted index lists. The JSON object has exactly three fields: `format`, `explanation`, and `assignments`. Explanation is text only and never executed. Each assignment has exactly `tensor`, `selectors`, and `values`.

- `tensor`: one of `W1`, `b1`, `W2`, `b2`, `W3`, `b3`.
- `selectors`: one selector per tensor dimension. A selector is either a nonempty list of distinct integer indices in any order, or an object with exactly the three keys `{ "start": i, "stop": j, "step": k }` and half-open range semantics. Preserve list order; do not sort it. For axis size `n`, require integer values `0 <= i < j <= n` and `1 <= k <= n`; list indices must be in `[0,n)`. Booleans are not integers for this format. No negative indices, empty selections, duplicate indices within a selector, or implicit slices.
- `values`: a finite JSON number broadcast to every selected element, or a flat list of finite numbers whose length equals the Cartesian product of the selected axis lengths. Iterate each axis in its selector's supplied order, with the last axis varying fastest. For selectors `[[2,0],[7,0]]`, four values map to `(2,7), (2,0), (0,7), (0,0)` in that order.
- Tensors start at zero. Apply assignments in array order; later assignments overwrite earlier values at overlapping entries. Omitted entries remain zero. Record the number of overwritten scalar entries.

Illustration of syntax, not a working policy:

```json
{
  "format": "tensor-assignments-v2",
  "explanation": "Example of assignment syntax only.",
  "assignments": [
    {"tensor":"W1","selectors":[[0],[0,7,14]],"values":[1.0,2.0,3.0]},
    {"tensor":"b1","selectors":[{"start":0,"stop":32,"step":1}],"values":0.0}
  ]
}
```

The actual prompt's syntax example must use a separate toy tensor with an explicitly different schema, or this example with an explicit warning; it must not supply a useful circuit.

The parser must reject duplicate JSON keys, unknown fields, nonfinite values, booleans masquerading as numbers, wrong tensor rank, invalid selectors, incorrect value counts, magnitudes above `10,000`, and values not finite after float32 conversion. Validate every supplied value, even one later overwritten. Limit the response file to 5 MiB, explanation to 10,000 characters, assignments to 10,000, and total expanded scalar writes to 1,000,000 including overwrites. Final unique entries cannot exceed the 15,555 tensor parameters. Check bounds and resource limits before expanding ranges. Empty assignments are syntactically valid and produce the zero network.

Request bare JSON. The parser accepts either one JSON document or that same document enclosed by exactly one outer Markdown fence whose opening line is three backticks followed by `json` and whose closing line is three backticks. Only whitespace may surround the document or fence; reject trailing prose, multiple blocks, or any other envelope. This fixed envelope rule is part of the precommitted contract, not a repair heuristic. Preserve the original response bytes.

Do not use `eval`, execute submitted code, extract a preferred block from commentary, round to nicer numbers, clip weights, add fallbacks, or repair an invalid submission. Reject the entire submission atomically if any assignment is invalid; never evaluate a partially applied artifact. JSON Schema checks structure and basic types; the authoritative parser additionally checks tensor-dependent bounds, expanded value counts, numeric conversion and resource limits. Test both layers before generation; do not assume schema-valid means semantically valid. Include realistic whitespace, the optional envelope, unsorted selectors, ordered overwrites, duplicate keys, bad values, and boundary/resource cases. Save tensors using a non-executable numeric format such as `.npz` loaded with `allow_pickle=False`.

## Local Codex runner

Use the local Codex CLI as the generation route, explicitly requesting `gpt-6-astra` with reasoning effort `xhigh`. Do not substitute another model or switch to an API route. The runner defaults to a dry run; only a separate explicit invocation may issue a model request.

The planning audit inspected `codex-cli 0.164.0` offline. The preview still exposed personal instructions and tool declarations under some proposed overrides. Its `debug prompt-input` path does not accept `--ignore-user-config`, so that preview does not establish the exact behavior of `codex exec` with that flag. No tool execution or execution-denial behavior has been tested. **The runner is not ready for the game until the validation gate below passes.**

Verify the installed CLI against the [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode). The authors' [no-tools configuration](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/harness/codex_config.toml) is a reference with placeholder paths and version-dependent settings, not a validated configuration for this machine.

### Isolation and configuration

- Create a dedicated private Codex home/configuration context and a separate empty working directory outside the implementation repository. Do not populate them with personal `AGENTS.md`, skills, memories, plugins, project files, baseline code, test seeds, or previous conversations. Supply the game prompt on stdin in a new session; never resume or fork a conversation.
- Preserve the user's normal Codex home, settings and authentication. If the isolated home requires authentication, use the supported login flow for that home; do not copy credential files or tokens. Credential material must never enter experiment artifacts. Keep the isolated directory private and outside version control.
- Use the neutral instruction `Answer the user's request.` and archive its exact bytes. Inspect both loaded instructions and effective tool declarations; a clean working directory or `--ignore-user-config` alone is insufficient. Record any managed configuration that affects the run.
- Starting from the installed version's supported settings, disable execution, editing, browsing, search, computer/image tools, apps/MCP/plugins, skills, memory, hooks and delegation. Set `agents.enabled = false` explicitly; disabling feature flags alone leaves delegation tools visible in CLI 0.164.0. Remove obsolete settings such as `collaboration_modes` and `worktrees` when the installed version marks them removed. Do not treat accepted configuration keys or `features list` alone as proof that a capability is unavailable.
- Keep read-only sandboxing and approval policy `never`. They are additional restrictions, not evidence that computation is disabled. Use strict configuration validation and suppress rule/instruction discovery through supported mechanisms; inspect the resulting context rather than assuming suppression worked.
- Request `model_reasoning_summary="detailed"` when supported and archive returned summaries. These are supplementary design notes, not a complete reasoning trace or proof of how the network was constructed. Use verified current context/output limits and defaults; do not copy the original experiment's oversized context and compaction overrides. Record the settings and audit any compaction event.

### Required validation gate

Complete this gate before freezing and sending the game prompt. Offline checks issue no model requests. Smoke tests are separate explicit commands, never automatic consequences of installation, calibration, dry runs or failed checks.

1. Save the CLI version, requested model, supported relevant settings, effective configuration, neutral instructions and an offline preview of instruction/tool declarations. Use the exact isolated home, working directory and overrides intended for generation. Document any difference between the preview command and the real `exec` path; do not represent a non-equivalent preview as verification of the real path.
2. Run a **required, content-free execution-denial smoke test** through the actual `codex exec` path and exact isolation settings. Ask for a harmless calculation such as `2+2` specifically through the advertised execution tool. The test must demonstrate that execution is unavailable or that a corresponding attempted call is explicitly denied without execution. An ordinary answer or a refusal to attempt the tool does not validate a reachable execution stub. Inspect other advertised callable capabilities as well; do not accept an unverified alternative execution or delegation path.
3. If tool schemas remain as inert stubs, accept that configuration only after nonexecution has been validated. A “Code Mode unavailable” or “fails closed” notice is supporting evidence, not sufficient proof on its own; absence of that notice is likewise not the sole verdict. Distinguish a startup notice from an actual tool-call event. The exact Code Mode unavailable/fail-closed message observed as an `item.completed` error before `turn.started` is recorded as a startup notice; other envelopes/messages, the same message during a turn, and reconnecting errors remain audit failures. The source event log and its hash are frozen as a regression fixture. A valid stream must contain one started and completed turn. If the gate cannot establish that execution is unavailable, stop before the game request and report the unresolved capability.
4. Validate `--output-schema` with the actual schema and a content-free fixture request, through the same route. Prescribe exactly `{"format":"tensor-assignments-v2","explanation":"Transport check only.","assignments":[{"tensor":"W1","selectors":[{"start":0,"stop":2,"step":1},[5,3]],"values":[0,0,0,0]},{"tensor":"W1","selectors":[[1],{"start":3,"stop":6,"step":2}],"values":0}]}`; do not ask the model to invent weights. This all-zero document exercises both selector forms, unsorted list order, both value forms and two overwritten scalar entries. Confirm that the saved final response is accepted by both the structural schema and the semantic parser and produces only zero tensors. Exercise the same coverage with offline fixtures. This may share a smoke invocation with the execution-denial check only if both outcomes are unambiguous. Test the event auditor on the resulting events and on fixtures containing successful, denied and unknown tool events.
5. Archive smoke prompts, responses, event logs, configuration, conclusions and hashes. Record smoke requests separately from experimental policy attempts. They must contain no game rules, source, baseline, landing table, seed or proposed weights, and their conversation must never be resumed for generation.

After the gate passes, freeze the prompt and configuration, then make one explicit game-generation request. Revalidate the gate after any change to the CLI version, model, isolation settings or relevant tool configuration. If effective declarations contain an unverified capability, do not proceed on the strength of a prior smoke test.

Invocation shape, to be completed with the validated isolated configuration—not a standalone no-tools command:

```text
codex exec --model gpt-6-astra --ephemeral --json
  --strict-config --skip-git-repo-check
  --sandbox read-only --cd <clean-working-directory>
  -c approval_policy="never"
  -c model_reasoning_effort="xhigh"
  -c model_reasoning_summary="detailed"
  <validated neutral-instructions and isolation settings>
  --output-schema <absolute-schema-path>
  --output-last-message <absolute-response-path>
  -
```

Launch with an argument vector, an explicitly selected isolated Codex home in the child process environment, and explicit stdin/stdout/stderr handles. Do not use `--ignore-user-config` if doing so would discard the isolated configuration being validated. Never invoke a shell with interpolated model output. The configurable transport timeout starts at 45 minutes; preserve interrupted streams and never automatically launch another generation after a timeout.

Save raw JSONL events, stderr, the exact final response, exit status, timestamps, resolved settings and available model/usage metadata. Audit every event against the installed version's event schema. Unknown events require inspection before a valid audit can be declared. A substantive tool-call attempt in the game run invalidates the primary no-tools condition even when denied; inert declarations and documented startup notices alone are not calls. Compaction is a protocol deviation that must be reported and prevents declaring the planned uncompacted single-response condition valid. Describe log evidence as observed evidence, not cryptographic proof of backend internals.

## Attempts and failure handling

Count every substantive game-generation attempt, including malformed, incomplete, interrupted or weak responses. Keep authentication failures before substantive generation and content-free smoke requests in separate manifest categories. Never erase an attempt or start an automatic retry because its output is inconvenient.

An actual tool-call attempt invalidates the primary no-tools condition, including an explicitly denied attempt. Invalid JSON or assignments under the frozen output contract are failed artifact submissions. Do not repair the model's numbers, relax the grammar, skip invalid statements, apply a valid prefix, select a preferred block or substitute fallback weights. Invalid submissions must not reach scoring through a partially populated network.

Distinguish a nonconforming response from an implementation bug. If the response conforms to the frozen grammar but the parser rejects or misinterprets it, preserve the original parser and result, add a regression fixture demonstrating the defect, and transparently correct the parser against the **same unchanged raw response**. Record the correction, hashes and original/corrected parsing or scoring outcomes. This is a harness correction, not a new model attempt. It must not change the frozen grammar or choose an interpretation based on policy performance. If the intended meaning is ambiguous under that grammar, the response remains invalid. Broader environment or scoring bugs require a documented protocol revision and preservation of the original outcomes.

Additional game attempts may be authorized later, but retain the first attempt and label every change to prompts, parser contract, feedback, model or settings. Diagnostic float64 evaluations, isolated-state probes and alternate reports cannot replace the primary frozen evaluation.

## Implementation deliverables and acceptance order

Suggested lightweight Python layout:

```text
src/ricochet_catch/
  env.py             # small standalone game; no baseline helpers
  observations.py    # public indexing and observation construction
  policy.py          # pure NumPy forward pass
  assignments.py     # declarative parser and semantic validation
  baselines.py       # excluded from model bundle
  evaluate.py        # seeded tape evaluation and metrics
  render.py          # local playback / optional human play
  run_astra.py       # dry-run by default; isolated CLI invocation and audit
  cli.py
schemas/tensor-assignments-v2.json
configs/game.json
prompts/neutral.txt
tests/
runs/<run-id>/
```

Use NumPy for inference; avoid PyTorch, a GPU, or a heavyweight game engine as requirements. A Gymnasium adapter is optional. Pin chosen dependency versions. The renderer is a local artifact, not a hosted website, and must use the same transition code and recorded trajectories as evaluation.

Implement in this order:

1. Environment, deterministic tape replay, observation mapping, and meaningful mechanics fixtures.
2. Renderer and human controls; inspect the prediction and bomb-conflict fixtures.
3. Baselines and development calibration; finalize game constants.
4. Network inference, parser/schema, realistic valid and malformed-response tests, ordered-overwrite fixtures and numeric fixture verification.
5. Evaluator and report generation using synthetic/manual test submissions, clearly labeled as such.
6. CLI isolation, dry-run prompt preview, logging, and event-audit fixtures; separately initiated content-free runtime preflight before any experimental generation.
7. Freeze manifest, clean prompt bundle and readiness report. Do not auto-launch Astra during installation, tests, or calibration.

Expose commands equivalent to `play`, `baseline`, `calibrate`, `freeze`, `preflight --dry-run`, `preflight --execute`, `generate --dry-run`, `generate --execute`, `evaluate`, and `replay`. Exact command spelling is an implementation choice; document it. Both preflight and experimental model requests require their explicit execute command; installing or testing the project must never issue them.

For each real run preserve: manifest and hashes; prompt and schema; CLI configuration; raw events/stderr/response; audit and parse reports; immutable tensors; per-episode metrics; replay trajectories; aggregate statistics and conclusion. Keep secret material out of these artifacts.

Show three replays selected by a fixed rule: the lower median episode after sorting by `(return, episode_index)` (position 127 with zero-based indexing for 256 episodes), the highest-score episode (tie: lowest episode index), and the earliest-index bomb failure if any. Label selection and overlay policy name, score, tick, action, and logits for Astra. Visual evidence supplements the full evaluation table.

The implementing agent's first handoff should include calibration results, test status, runtime-preflight status, the exact pending generation invocation, and all frozen hashes. Mark runtime preflight pending if its explicit execute command has not been run. Clearly distinguish implemented behavior from an Astra result that has not yet been obtained.
