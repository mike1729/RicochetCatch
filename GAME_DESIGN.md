# Ricochet Catch: game and policy specification

Prepared 2026-11-03. Game version `0.1`, reviewed specification. The production implementation passes the mechanics, neural-policy, parser, renderer, and evaluation tests, and reproduces the independent review calibration with the constants below. Runtime preflight and the Astra experiment remain pending. See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for validation evidence and [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the protocol.

This is one game containing rewards, diagonal motion, wall reflections, and bombs. There are no color flips, inventory, levels, upgrades, hidden timers, or persistent policy memory. The experiment tests whether Astra can write weights that produce useful play; the task does not prescribe an internal policy algorithm or require explicit multi-step landing prediction.

## 1. Board, actions and objective

- Board: 8 columns × 8 rows. Column 0 is left; row 0 is top.
- A one-cell paddle occupies row 7, initially column 3.
- Three discrete actions: `0 = stay`, `1 = left`, `2 = right`.
- An action moves the paddle by 0, -1, or +1 columns, clipped to `[0,7]`.
- Gem caught at the paddle: reward `+1`.
- Bomb caught at the paddle: episode terminates, reward `0` for that step.
- Missed objects: reward `0`. No reward for survival and no separate movement cost.
- The score is total gem catches before a bomb collision or the 256-tick time limit.

A bomb ends future scoring. It does not subtract an arbitrary shaped reward. All episode returns are nonnegative.

## 2. State and episode generation

Policy-facing state consists only of paddle column, all active objects' type/row/column/horizontal direction, and remaining ticks. An object is `(kind, row, column, vx)`, where kind is gem or bomb and `vx` is -1, 0 or +1. Its vertical speed is always +1 row per tick. Policies receive the observation in section 4, or an equivalent decoding of exactly those fields for the baselines.

Evaluator-only metadata must remain separate. For example, whether an object's complete fall includes a wall reflection may be useful for reporting, but must not be attached to a policy-visible object tuple or exposed through an environment handle. Define each diagnostic label explicitly, including whether it describes a past reflection or the complete trajectory. Such labels never become policy inputs.

Initial state: no active objects, paddle column 3, `remaining_ticks = 256`.

Retain this configuration, which passed the review simulation. Verify the production implementation against that simulation and complete the recorded calibration/freeze procedure before generation:

| Constant | Value |
|---|---:|
| Spawn probability per tick | 0.45 |
| Probability a spawned object is a bomb | 0.25 |
| Spawn column | Uniform over 0–7 |
| Horizontal velocity | Uniform over -1, 0, +1 |
| Episode horizon | 256 ticks |

All mechanics and probabilities are supplied to Astra. The per-episode seed and future random outcomes are not.

Before each episode, generate a spawn tape for all 256 ticks independently of the policy. For each tick, draw a fixed number of RNG values in a fixed order: spawn indicator, column, velocity and type, even if no object will spawn. Record the chosen RNG algorithm and dependency version. A tape entry is `null` or an object type/column/velocity. Replaying the tape must not depend on actions, observations, renderer activity, or early termination.

At most one object spawns per tick. Objects spawn at row 0 **after** the current tick's movement/collisions. They first move on the following tick. Consequently at most one object occupies each row in every reachable state, and there are at most seven active objects in a nonterminal observation. No arbitrary active-object cap is needed.

Do not expose the spawn seed, tape, RNG state or future arrivals to any policy. A pre-generated tape is an implementation device for common random scenes, not an observation. From the policy's perspective spawning is independent stochastic input; no memory is needed to infer a hidden spawn clock.

## 3. Exact step order

`step(action)` on a finished episode is an error. Validate action as one of the three integer action indices; do not coerce floats or booleans.

For each tick:

1. Move the paddle according to the action and clip its column to the board.
2. Simultaneously advance every currently active object:
   - Start with `next_column = column + vx`.
   - If `next_column < 0` or `next_column > 7`, replace `vx` by `-vx`, then set `next_column = column + vx`.
   - Set `column = next_column` and `row = row + 1`.
   - The reflected direction is persistent and appears in the next observation.
3. For any object whose updated row is 7, compare its updated column with the **new** paddle column. Award +1 for a matching gem, terminate for a matching bomb, or count it as missed. Remove the object in every case. No object remains on row 7 in the returned observation.
4. Increment elapsed ticks / decrement remaining ticks once, including on a collision step.
5. A bomb collision takes precedence over the time limit: `terminated=true`, `truncated=false`. Return immediately, with no spawn on that tick.
6. If remaining ticks is now zero, return `terminated=false`, `truncated=true`, without spawning.
7. Otherwise append the current tick's spawn-tape entry at row 0, if present, and return the new observation.

There is no collision between falling objects, no collision above row 7, and no swept-path collision. For example, a paddle moving away from a bomb's landing cell on its arrival tick escapes it. Reaching the wall at column 7 with `vx=+1` does not flip direction until the next attempted outward movement; direction channels reflect exactly this rule.

Since at most one object is born per tick and all have identical vertical speed, simultaneous gem/bomb arrivals cannot occur in reachable states. State-loading utilities must validate field bounds and the unique-row invariant rather than inventing an order for an unreachable collision.

At least one action always avoids an immediate bomb collision in a reachable state: the paddle can reach at least two distinct columns in one tick, while at most one bomb can arrive. Safety-aware baselines should assert that their immediate-safe-action set is nonempty. This guarantees an escape on the next tick, but does not guarantee that the desired gem remains catchable.

## 4. Observation encoding

Return a `float32` vector of length **449**. The first 448 entries encode an `8 × 8 × 7` binary grid with channel last. Keep the full layout, including entries that are always zero in reachable play; do not remove padding or change the network dimensions. The channels are:

| Channel | Meaning |
|---:|---|
| 0 | Paddle occupancy |
| 1 | Gem with `vx = -1` |
| 2 | Gem with `vx = 0` |
| 3 | Gem with `vx = +1` |
| 4 | Bomb with `vx = -1` |
| 5 | Bomb with `vx = 0` |
| 6 | Bomb with `vx = +1` |

Grid index: `((row * 8 + column) * 7 + channel)`. Each active object sets exactly one entry to 1. The paddle sets `(row=7, column=paddle_column, channel=0)` to 1. All other grid entries are zero.

Entry **448** is `remaining_ticks / 256`. There is no learned normalization, frame stacking, cropping, target selection, computed landing position, danger score, or other preprocessing. Raw direction information is intentional: the network is being tested on computing a policy, not on guessing unobserved velocity.

Observations after terminal steps use the updated remaining time and object list. Evaluation will never request another action from such an observation.

## 5. Fixed neural policy

Use column-vector convention:

```text
x      : (449,)
W1, b1 : (32,449), (32,)
W2, b2 : (32,32),  (32,)
W3, b3 : (3,32),   (3,)

h1 = tanh(W1 @ x + b1)
h2 = tanh(W2 @ h1 + b2)
z  = W3 @ h2 + b3
action = argmax(z)
```

All stored tensors and inference arithmetic are float32. Use the standard NumPy tanh, with pinned NumPy version. A tied argmax selects the smallest action index; all-zero tensors therefore always select stay. Reject nonfinite observations or logits as implementation/artifact errors instead of silently choosing an action.

Parameter count:

`(449*32 + 32) + (32*32 + 32) + (32*3 + 3) = 15,555`.

No recurrence, controller code, external rule lookup, action masking, sampling, or weight changes are permitted at inference time. The environment's clipping of the paddle at a wall is part of game physics, not a policy-side action correction.

## 6. Local display and playback

The visual board is square and displays the exact discrete state. Show gems as diamonds, bombs as crosses, and the paddle as a bar. Pair color with shape so color is not the only distinction. Indicate horizontal direction with a small arrow or trail, distinguishing stationary-horizontal objects. These indicators are purely display features derived from the state.

Show score, tick/horizon, policy name and last action. A network replay additionally shows its three logits. Human play uses left/right keys and no key for stay, with pause, single-step and restart. Keep simulation ticks independent of display frame rate; an initial display speed of 5 ticks/second is suitable for inspecting trajectories. Rendering must consume recorded states and never advance RNGs itself.

Display a clear game-over cause: bomb hit or time limit. Save event/state/action trajectories so the same episode can be replayed identically without running the policy again. Report any maximum-score or failure replay as selected; do not present it as a random representative episode.

## 7. Mechanics fixtures for the implementing agent

These fixtures and their explanations are **not part of Astra's prompt**. Disable spawning during each fixture and use sufficient remaining time.

### Boundary and scoring fixtures

- Object `(gem, row=3, col=7, vx=+1)` becomes `(gem,4,6,-1)` after one tick.
- Object `(bomb,3,0,-1)` becomes `(bomb,4,1,+1)`.
- Object at `(3,6,+1)` first moves to `(4,7,+1)`; only the following step reflects.
- A vertical gem at `(6,3,0)` with paddle column 3 is caught by stay; score increases by exactly one and the object disappears.
- Replace that gem by a bomb: stay terminates with zero reward; left escapes and the bomb disappears.
- Left at column 0 and right at column 7 leave the paddle in bounds.
- With one tick remaining, process a gem catch before truncating. A bomb hit on that same last tick is termination, not truncation.
- A tape spawn appended at row 0 after a step is not advanced in that same step.
- Same seed/tape and action sequence gives byte-identical observations and identical events, with or without rendering.

### Prediction fixture

Paddle column 3, one gem at `(row=4, col=6, vx=+1)`. It lands in three ticks at column 5, after a wall reflection.

- Actions right, right, stay catch it.
- Following its current column at each tick produces right, right, right and misses it.

This verifies movement and a concrete failure of current-position tracking. It does not establish that successful overall play requires explicit landing prediction, multi-step planning, or an internal simulation. The reviewed one-step-lead controller already clears the proposed performance threshold; retain it as a baseline in EXPERIMENT_PLAN.md. Observed success supports the weight-writing capability claim without identifying the network's internal method.

### Conflicting objective fixture

Paddle column 3. A vertical bomb at `(6,4,0)` lands next tick; a vertical gem at `(5,5,0)` lands one tick later.

Reaching the gem requires moving right twice, but the first move hits the bomb. A safe controller must forgo that gem. This fixture checks that the game can require sacrificing a reward, without introducing an impossible-to-avoid hazard.

## 8. Required correctness checks

Before calibration, verify action/update ordering, wall reflection direction, terminal precedence, score accounting, and observation indexing against the fixtures. Independently check the inference equation against a manually calculable small tensor example; a test that simply calls the same inference helper twice is insufficient.

Test tape independence using different action sequences on the same tape. Compare spawn entries on ticks when both episodes actually execute the spawn stage. If one episode terminates on a tick, its skipped spawn must not be treated as a tape mismatch. Also check that a shared tape remains unchanged after either episode finishes. Confirm that the random-policy RNG does not perturb spawning. Check all reachable observations respect binary occupancy and velocity channels, and that the remaining-time feature reaches zero exactly at the horizon.

Check that removing or changing evaluator-only reflection labels cannot change any policy observation or baseline action. Enumerate reachable immediate-collision cases to verify the nonempty-safe-action invariant.

Use tiny fixtures and baseline development runs to detect impossible collision states, reward exploits, or a reference controller accidentally reading future spawns. Freeze the final implementation and calibrated configuration according to EXPERIMENT_PLAN.md before any experimental Astra submission.

## 9. Frozen diagnostic probes after generation

These probes are evaluation-only and stay out of the Astra prompt. Freeze their generator and reporting rules before generation, then run them on the submitted network without changing its weights or selecting a replacement. They supplement the episode scores.

- Generate distinct reachable single-object states from legal spawn-and-movement prefixes, rather than treating every row/column/direction combination as reachable. Use elapsed tick 8 (`remaining_ticks = 248`), both object kinds, each reachable row/column/direction state, and every paddle column realizable by a legal action prefix. Keep all other spawn slots empty.
- From each state, suppress subsequent spawns and roll forward until that object arrives. Report gem catches separately for states where a catch was physically reachable, and bomb avoidance for bomb states. Record actions and logits; stratify by velocity, time to arrival, and whether a reflection occurs before arrival. These strata are evaluator labels only.
- Include the fixed two-object conflict fixture above as a separate diagnostic. Any additional multi-object probes must preserve unique rows and have a legal construction.
- Label deliberately unreachable or malformed states as synthetic validation tests, and exclude them from reachable-state diagnostic rates and gameplay evaluation.

Probe outcomes describe behavior under controlled states. They cannot establish that the network explicitly represents landing columns, simulates future motion, or uses any particular internal mechanism.
