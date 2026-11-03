> Historical v1 handoff below; superseded by [the review resolution](REVIEW_RESOLUTION.md). The current handoff is regenerated after the source commit and replacement freeze.

# Implementation handoff — 2026-11-03

Ricochet Catch v0.1 is implemented and tested. No experimental Astra submission has been requested. Runtime execution-denial/schema preflight remains pending, and no hidden test episodes have been scored.

## Implemented

- Deterministic game physics, immutable spawn tapes and policy state, exact 449-element float32 observations.
- Local browser human play with independent simulation ticks, shape/direction markers, pause, step, restart and recorded trajectories.
- Seven state-only baselines, calibration, paired bootstrap reporting, full-tape catch denominators, and deterministic replay selections.
- The fixed 15,555-parameter NumPy network, strict atomic tensor assignment parser, JSON Schema, and non-executable numeric storage.
- 2,496 reachable single-object diagnostic states with legal construction witnesses and a separate two-object conflict probe.
- Dry-run-first isolated CLI runner, conservative event auditing, preserved attempt artifacts, and hash-bound preflight/generation gates.

## Validation

Final correctness suite: **207 passed in 2.13s**, with no warnings or skipped tests. See [saved test output](runs/development/tests-final.txt).

Mechanics tests include byte-identical comparisons with the independent review simulator, terminal precedence, scoring, reflections, tape independence, all observation channels, and malformed-state rejection. Neural inference is checked against independently calculated values; parser tests cover malformed envelopes, numeric/resource limits, overwrites and ordered selectors. UI tests execute the shipped JavaScript and cover controls, replay isolation, and rapid restart races.

Browser verification exercised the final local display, single-step, timed play, pause, restart and game-over cause. The saved preview is a manually observed stay episode, not a representative or maximum-score selection.

[Game preview](runs/ui-test/game-preview.png) · [prediction catch](runs/fixtures/prediction_catch.html) · [prediction chase miss](runs/fixtures/prediction_chase_miss.html) · [bomb conflict](runs/fixtures/conflict_bomb_hit.html)

## Development calibration

All four declared checks pass with the original constants, using seeds 0–255. Production means exactly reproduce the review simulation.

| Policy | Mean catches |
|---|---:|
| stay | 2.72265625 |
| random | 2.97656250 |
| chase | 6.19140625 |
| chase_safe | 21.81250000 |
| one_step_lead_safe | 44.07812500 |
| predict_greedy_safe | 56.75390625 |
| reference | 57.57031250 |

The trivial comparator is **chase**. The halfway threshold is **31.88085938**. The one-step lead paired margin has one-sided 95% lower bound **11.75781250**, using 10,000 resamples and bootstrap seed 12345.

[Full calibration and per-episode results](runs/development/calibration-final.json). These are development findings, not Astra or held-out results.

A manually supplied all-zero network was evaluated through the full evaluator on all 256 development tapes. Its mean was 2.72265625, exactly matching stay. Its report explicitly marks it manual/synthetic and withholds an Astra success declaration. [Full synthetic report](runs/zero-network-check/report.json). Selected replay files contain logits, and the 2,496 probes were exercised.

## Frozen commitments

[Freeze manifest with every source/artifact hash](runs/frozen/manifest.json) · [readiness](runs/frozen/readiness.json) · [clean prompt](runs/frozen/prompt.txt)

Manifest SHA-256: `4247cdc076141710ba60a95f8c7f57052ce78a6c7c019e55b667781e765d3ec8`.

The private held-out seed manifest contains 256 distinct seeds disjoint from development/review seeds. It is stored under a private directory, excluded from Git, and has not been scored. Development random actions use seed base 10,000,019; held-out random actions use 20,000,019. Float32 arithmetic, NumPy 2.3.4, interpreter/CPU/BLAS details, bootstrap seed/rank convention and probe definitions are archived.

The Git HEAD identifies the starting revision; local implementation bytes are committed by their hashes. No Git commit or remote publication was performed.

## Runtime preflight and exact pending invocation

Installed CLI observed offline: `codex-cli 0.164.0`. The candidate configuration and dry-run invocation are prepared. Accepted settings and an offline preview alone do not establish execution denial. The actual isolated runtime path, denial behavior and schema transport have not been tested. Readiness therefore remains **not ready for generation**.

[Preflight dry-run manifest](runs/preflight-final-preview/manifest.json) · [generation dry-run manifest](runs/generate-final-preview/manifest.json). Both record zero model requests.

The preflight requires its separate explicit execution command:

```sh
.venv/bin/ricochet-catch preflight --execute --run-dir runs/preflight \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work
```

After its exact instruction/tool context and denial/schema evidence pass review, the pending generation invocation is:

```sh
.venv/bin/ricochet-catch generate --execute --run-dir runs/astra-001 \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work \
  --gate runs/preflight/gate-review-template.json --freeze runs/frozen/manifest.json
```

Neither command was executed. Generation requests GPT-6 Astra with xhigh reasoning, detailed summaries and a 45-minute transport timeout. It has no automatic retry or artifact repair. Missing or changed gate/freeze evidence prevents generation. Official evaluation checks unchanged response provenance, parses/hashes tensors, reruns the frozen suite, and only then scores the hidden tapes.

## All frozen SHA-256 hashes

| File | SHA-256 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | `62fd0db46e1164e974bb272954c32fb24d3568aaec5d20c5aa5b60bc4caf0d71` |
| [GAME_DESIGN.md](GAME_DESIGN.md) | `8eb99f9aa4dd6dec76fda92b7b87f4408273b2a75568e00105a4a04807b7ee64` |
| [configs/game.json](configs/game.json) | `c948eb095c3bd95f4c539cf5ffc11fabcb4e319d06fff9368ab47a5fe5004845` |
| [prompts/neutral.txt](prompts/neutral.txt) | `5c5defc1e76e42bc8d8dd7bd734400d30ef42d0ed7ef9bcbd3d106978c0ad718` |
| [pyproject.toml](pyproject.toml) | `f1a4d15025ca9b51a50cc068f35ebafa76d9cda77c4eb4e21996f51f8bfeb00a` |
| [requirements.txt](requirements.txt) | `ffa1e545cccc2839c9044c0e0046adc869049dd673dc601ecf14e21788eabffb` |
| [schemas/tensor-assignments-v2.json](schemas/tensor-assignments-v2.json) | `8339b48933dabce22fb043d17fefa4b6dd0c1d513889ef2859117a963f7cc0c7` |
| [src/ricochet_catch/__init__.py](src/ricochet_catch/__init__.py) | `9e39a1b569861f678fad0471ecc8cee2e72e6a61e35620170414a4708f632acc` |
| [src/ricochet_catch/__main__.py](src/ricochet_catch/__main__.py) | `3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a` |
| [src/ricochet_catch/assignments.py](src/ricochet_catch/assignments.py) | `94c23259c9206443362ee5830571ba4b78eb09db47fa77c1d9955740d5c9fc7a` |
| [src/ricochet_catch/baselines.py](src/ricochet_catch/baselines.py) | `345b7ef921b8ca49c51ba9464b5aa6a140a85ecf210da015693c5451452b0fb2` |
| [src/ricochet_catch/cli.py](src/ricochet_catch/cli.py) | `0825845ce19fdc2f2140ee005796312f328c761c75f2c6cb0e83fd47c6722a53` |
| [src/ricochet_catch/env.py](src/ricochet_catch/env.py) | `7643ae3b6b64565ce47f77427acd61312bd6280b706f3e69b97e0d2df6f024ab` |
| [src/ricochet_catch/evaluate.py](src/ricochet_catch/evaluate.py) | `c2aa0a0adf124104dff4f2b61345723ae38d8b94b20792747cb9ed049d1a4eb0` |
| [src/ricochet_catch/freeze.py](src/ricochet_catch/freeze.py) | `5ddea2adb50173ae8c502a849d733314a131e364c1a2fa6ca66d14492448caa2` |
| [src/ricochet_catch/observations.py](src/ricochet_catch/observations.py) | `a1126ae62dabe0885d6ca8e2adf07ecf1014ec52ddc5e04dfbf4d9d4ad9e861b` |
| [src/ricochet_catch/policy.py](src/ricochet_catch/policy.py) | `63a5cb91f16ad1b750b801ab499ec90157ff763b3d0bc4f2e5f6d1fc20027a1f` |
| [src/ricochet_catch/probes.py](src/ricochet_catch/probes.py) | `a44dfdcbfc2103ffff067c421a2e15e1d154d362846ab2b94f0055e93751ce46` |
| [src/ricochet_catch/render.py](src/ricochet_catch/render.py) | `74a8597478d6476fc96715ca805602c21f28fe607dc174c074a67391eb04d2f7` |
| [src/ricochet_catch/run_astra.py](src/ricochet_catch/run_astra.py) | `c77b0a393c7a365afd2ae896fed1aa57a004683521a9e63aa39aad2fb739aad6` |
| [tests/test_assignments.py](tests/test_assignments.py) | `48c6b46dc4c8f1cd36ab5324012bdd8f6231d4efed3f03130ad560632c152d4a` |
| [tests/test_baselines.py](tests/test_baselines.py) | `106cb4d2d644d9cf52e0864963090e0570b6a2dc9421728d7f9ec2181ace8b67` |
| [tests/test_cli.py](tests/test_cli.py) | `c43eb0000685c8b1e49c6e840f0d8a5b2fe0c882a037cd3560ba8e20ffd5ec6e` |
| [tests/test_env.py](tests/test_env.py) | `da53695422e6261f0ed9a015c58c405d29c43f172a8ca0787e5e7b19a76ea995` |
| [tests/test_evaluate.py](tests/test_evaluate.py) | `a7f3e3a0a74f8af42a64fb4277f437ecc2250fac111c6d94296015db4a1472ca` |
| [tests/test_policy.py](tests/test_policy.py) | `0a05723d24f44ca043673d0e02c87b59e4c975ce4d32ca6ab4be9aaccbbab226` |
| [tests/test_probes.py](tests/test_probes.py) | `4ab35de0f3a639874ff7b04e35d6936da7e1a8a487c746cef2a34863019eff32` |
| [tests/test_render.py](tests/test_render.py) | `720c9a2975e69e28e413c68121e6dad726e8daa96d49619cdedde4df109abc90` |
| [tests/test_render_ui.py](tests/test_render_ui.py) | `9f6605f4becf8220325790b1efa521cfcc84507d9ad483e53306980f3b50f0a6` |
| [tests/test_runner.py](tests/test_runner.py) | `0a8d0fbc35d60dc5ada7070bcf7b515d180fad2ca1eb8bdb0ea9729e547f02e1` |
| [uv.lock](uv.lock) | `ec1a271180670b054648314db3b0e43619910541234e92b3b553c220e5cfc535` |
| [runs/frozen/arithmetic.json](runs/frozen/arithmetic.json) | `f0975ad5c1d69349376f4b999f444644661dc228a6b7cc8dd69352059e6218bf` |
| [runs/frozen/calibration.json](runs/frozen/calibration.json) | `77cb969441c73dd471c028a911626d0916b9c980ce3284f0f49768a98fc5fae3` |
| [runs/frozen/cli-config.candidate.toml](runs/frozen/cli-config.candidate.toml) | `147e6e8b74e8c3654bba6c653b0200b01eea4f86e0e5ab23e6779e1f19afff65` |
| [runs/frozen/cli-validation-status.json](runs/frozen/cli-validation-status.json) | `0f35a903420e693fd37a8ffe8e9e846c7188230e6a4c4f91a5384474ca9fe674` |
| [runs/frozen/development-seeds.json](runs/frozen/development-seeds.json) | `20043feaaaf129ea0f5f5d09c3ce6da5c6af574d40bb1afac88c9df8d94381b8` |
| [runs/frozen/diagnostic-probes.json](runs/frozen/diagnostic-probes.json) | `869f7392b0a02f5e567e8ddee56047bf4d3bb1518bc71460fd6eb4e1152a1269` |
| [runs/frozen/diagnostic-rules.json](runs/frozen/diagnostic-rules.json) | `40266591594b94652a6438d249f385a41395381f4678ab1a862b5543fe84b204` |
| [runs/frozen/prompt.txt](runs/frozen/prompt.txt) | `a8d81fe5c1f69b507447dbd9dbe45fb8b2a75d9db9fb733c3367261e15e28d69` |
| [runs/frozen/readiness.json](runs/frozen/readiness.json) | `48fc274df03c8d436f5a736a8eaf8fe17a83a36e05dcefa8fd73029eb616ba68` |
| [runs/frozen/test-seeds.private.json](runs/frozen/test-seeds.private.json) | `182a1448c3750780208c3ccba2b37d9029693b469f3eec2df9c9de3cb03942aa` |
