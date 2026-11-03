# Ricochet Catch

A deterministic local game and experiment harness for testing whether GPT-6 Astra can write useful neural-policy weights from rules and source, without tools or performance feedback.

The game, local browser display, seven baselines, float32 network, strict assignment parser, evaluation reports, diagnostic probes, and guarded generation runner are implemented. **No experimental Astra submission has been requested.** Runtime preflight remains pending; hidden test episodes have not been scored.

## Install and play

Python 3.11–3.14 is supported; development was verified on Python 3.13.15 with NumPy 2.3.4. Dependencies are pinned in `uv.lock` and `requirements.txt`.

```sh
uv sync --locked --extra test
uv run --locked ricochet play
```

The game opens a browser connected only to a local server. Left/right arrows move the paddle; no key means stay. Space pauses, N advances one tick, and R restarts the same scene. The initial speed is 5 ticks/second. Close the server with Ctrl+C. The current human trajectory is saved to `runs/human/replay.json`.

The board shows gems as diamonds, bombs as crosses, and the paddle as a bar. Direction markers distinguish left, stationary, and right motion. Catch gems for +1; a bomb ends the episode. The horizon is 256 ticks.

## Commands

After installation, `.venv/bin/ricochet` and `.venv/bin/ricochet-catch` are equivalent. Use `--help` for all options. Output directories for evaluations, freezes, baselines, and runner attempts must be new; existing results are preserved.

```sh
# Correctness, parser, policy, evaluation, and UI tests. No model calls.
uv run --locked --extra test pytest -q

# Baselines and clearly labeled selected replays on development scenes.
uv run --locked ricochet baseline --policy all --episodes 256 --out runs/my-baselines

# Fixed development seeds 0–255 and predeclared calibration checks.
uv run --locked ricochet calibrate --out runs/my-calibration.json

# Export prediction and bomb-conflict correctness fixtures.
uv run --locked ricochet fixtures --out runs/my-fixtures

# Full evaluator check using a manually supplied zero network, on development scenes.
uv run --locked ricochet evaluate --submission examples/zero-network.json --out runs/my-zero-check

# Verify recorded physics and export standalone HTML; no policy or RNG rerun.
uv run --locked ricochet replay runs/my-zero-check/lower_median.json --open

# Commit source/dependency hashes, probes, private hidden seeds and readiness.
uv run --locked ricochet freeze --calibration runs/my-calibration.json --out runs/my-freeze

# Only prepare prompts/configuration/invocations. These issue no CLI/model requests.
uv run --locked ricochet preflight --dry-run --run-dir runs/my-preflight-preview
uv run --locked ricochet generate --dry-run --run-dir runs/my-generation-preview
```

JSON reports preserve episode scores, catch fractions with full-tape denominators, uncertainty, paired comparisons, arithmetic metadata, and near-tie counts. Network replay overlays include all three logits. Fixed selections are lower median by `(score,index)`, highest score with lowest-index tie break, and earliest-index bomb failure if present. They are selected examples, not random representative scenes.

The 2,496 single-object diagnostic states have legal eight-tick construction witnesses. A separate two-object probe checks the bomb/gem conflict. Reflection labels remain evaluator-only: policies receive only immutable four-field objects/paddle/time or the exact 449-element observation.

## Development validation

Production calibration exactly reproduces the independent review simulation on development seeds 0–255:

| Policy | Mean catches |
|---|---:|
| Stay / all-zero network | 2.7227 |
| Random | 2.9766 |
| Reactive chase | 6.1914 |
| Reactive chase with safety | 21.8125 |
| One-step lead with safety | 44.0781 |
| Predictive greedy with safety | 56.7539 |
| Predictive reference | 57.5703 |

All calibration checks pass with unchanged game constants. The development-selected trivial comparator is reactive chase; the halfway threshold is 31.8809. These are development findings, not Astra results.

## Explicit model requests

Installation, tests, calibration, freeze, and dry runs never launch a model request. Each actual request stage requires its own explicit `--execute` command. The runner never retries generation or repairs submissions.

The candidate isolated configuration still needs runtime verification. The required preflight makes two content-free requests, using the actual CLI route, to check execution denial and schema transport. Its exact all-zero transport document covers range and unsorted-list selectors, scalar and list values, and ordered overwrites. It preserves logs and an incomplete gate-review template. An ordinary answer, refusal, or startup notice cannot certify a reachable execution stub. Inspect the exact instructions, every callable capability, denied-call evidence, and managed configuration before completing the hash-bound gate. There is no command that automatically approves it.

The configuration explicitly sets `agents.enabled = false`, in addition to disabling execution and other capabilities. The audit recognizes the authors' exact pre-turn Code Mode startup error item as a notice. The same text after `turn.started`, other errors (including reconnecting errors), unknown events, tool attempts, and compaction still invalidate the primary run. The notice alone never passes the execution-denial gate.

Before runtime preflight, authenticate the dedicated home through the supported login flow. Run this interactively yourself; no login or credential access is performed by installation, tests, or dry runs:

```sh
mkdir -p /private/tmp/ricochet-astra/home
chmod 700 /private/tmp/ricochet-astra/home
CODEX_HOME=/private/tmp/ricochet-astra/home codex login
```

Check whether login created `config.toml`. If it exists, compare it with the candidate `config.toml` saved by your preflight dry run. Do not silently replace a different existing configuration: the runner checks this and stops before any request. If no configuration exists, the explicit preflight command installs the archived candidate. `/private/tmp` may be cleared by a reboot, in which case this isolated login must be repeated. Never copy credentials from your usual Codex home.

```sh
# Separate explicit content-free requests. Not run during implementation.
uv run --locked ricochet preflight --execute --run-dir runs/preflight \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work

# Only after preflight evidence is reviewed and every gate check passes:
uv run --locked ricochet generate --execute --run-dir runs/astra-001 \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work \
  --gate runs/preflight/gate-review-template.json --freeze runs/frozen-v2/manifest.json

# Only after that unchanged response has been archived and audited:
uv run --locked ricochet evaluate --submission runs/astra-001/response.json \
  --generation-run runs/astra-001 --freeze runs/frozen-v2/manifest.json --out runs/astra-evaluation
```

The harness never copies or archives credentials. Normal Codex settings are preserved. Generation refuses missing, changed, or incomplete preflight evidence and frozen artifacts. Official scoring also verifies generation provenance and reruns the frozen correctness suite after parsing and hashing the submission. Manual artifacts use development scenes and can never produce an Astra success declaration.

The review revision preserves `runs/frozen/` and creates `runs/frozen-v2/`, reusing exactly the original hidden seeds. To make a later revision without changing the episode commitment, pass `freeze --previous-freeze runs/frozen-v2/manifest.json --out runs/frozen-v3`. Source revision metadata identifies a Git commit only when every frozen source/test byte matches that commit; the simulator and all test fixtures are included. `FREEZE_COMMITMENT.json` records the active local manifest's hash in version control, while private seed data remain excluded.

The clean prompt includes authored game rules, exact environment/observation/inference source, and assignment contract. Baselines, planning helpers, review scripts, diagnostic probes, calibration scores, and hidden seeds are excluded. `runs/` is ignored by Git and includes private held-out commitments.

See [GAME_DESIGN.md](GAME_DESIGN.md) for mechanics and [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the research protocol. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) records this implementation handoff. `ricochet_sim.py` and `robustness.py` remain independent review aids and are excluded from the prompt.
