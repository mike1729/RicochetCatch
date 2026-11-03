# Ricochet Catch

A local experiment testing whether GPT-6 Astra can write the weights of a useful neural policy from a custom game's rules and source, without tools or performance feedback.

The initial study uses one game, one Astra submission, a fixed `449 → 32 → 32 → 3` network, and 256 held-out evaluation episodes.

- [Experiment plan](EXPERIMENT_PLAN.md): generation protocol, baselines, success criterion, assignment format, runner validation, and implementation handoff.
- [Game design](GAME_DESIGN.md): mechanics, observations, network, display, and correctness fixtures.

These are the current briefs, revised after review. The production implementation and Astra experiment are pending. The runner must default to dry run; installation, calibration, and tests must never issue model requests. Content-free runner preflight and the actual experiment each require a separate explicit execution command.

[ricochet_sim.py](ricochet_sim.py) and [robustness.py](robustness.py) are review aids for cross-checking mechanics and baseline behavior. Keep both out of the Astra prompt: they contain predictive controllers and lookup tables. Their results are development evidence, not Astra results. The simulator also carries evaluator-only reflection metadata that must be excluded from the production policy interface.
