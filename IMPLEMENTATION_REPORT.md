# Implementation handoff — review revision 2

Prepared 2026-11-03. The game and experiment harness are implemented and tested. The runner review fixes are complete. **Runtime preflight remains pending; no experimental Astra request, login or hidden test scoring has been performed.**

## Review changes and tests

The suite passes **235 tests**. [Saved output](runs/development/tests-review-v2.txt) · [Detailed review resolution](REVIEW_RESOLUTION.md).

- The event auditor accepts only the exact published pre-turn Code Mode notice; both the auditor and generation validator are tested against the entire pinned authors’ event log. Tool attempts, later/nonmatching errors, reconnecting errors, unknown events and compaction remain failures. A notice never proves execution denial.
- The isolated configuration explicitly disables sub-agents. Offline inspection confirms only four function tools remain declared, with no multi-agent, skills or personal-instruction blocks. The neutral instruction is exactly `Answer the user's request.`.
- The content-free transport document now exercises both selector forms, unsorted list order, scalar/list values and overwrites, with every tensor still zero.
- The freeze includes the independent simulator and every test fixture, and records a Git source revision only when all frozen bytes match that commit.
- The source is committed on `codex/implement-and-review-game`; a separate documentation commitment records the replacement manifest hash without committing private data.

[Offline inspection](runs/review-offline-v2/OFFLINE_REVIEW.txt) · [Final declarations](runs/review-offline-v2/current-final.inventory.json) · [Authors’ log provenance](tests/fixtures/authors-breakout/provenance.json). These are offline checks, not proof that the real execution route denies tool calls.

## Context decision

The installed CLI reports a bundled Astra default context of 272,000 tokens and a maximum of 872,000. Its effective compaction threshold/clamping cannot be verified from the available offline output. We retain model defaults rather than assume a 1,000,000 threshold is safe. This reasoning and the official OpenAI documentation links are recorded in [the review resolution](REVIEW_RESOLUTION.md#context-and-compaction-decision).

## Development calibration

Game mechanics, tensor grammar, network dimensions, baselines and primary success criterion are unchanged. The original calibration is reused; all four checks pass and the results exactly reproduce the independent review simulation on development seeds 0–255.

| Policy | Mean catches |
|---|---:|
| chase | 6.19140625 |
| chase_safe | 21.81250000 |
| one_step_lead_safe | 44.07812500 |
| predict_greedy_safe | 56.75390625 |
| random | 2.97656250 |
| reference | 57.57031250 |
| stay | 2.72265625 |

Comparator: **chase**. Halfway threshold: **31.88085938**. One-step-lead paired-margin lower one-sided 95% bound: **11.75781250**, with 10,000 resamples and bootstrap seed 12345.

[Full calibration](runs/frozen-v2/calibration.json). These remain development findings, not Astra or held-out results. The claim is useful play; no bounce-prediction claim or new secondary success requirement has been introduced. Float64 sensitivity remains optional and unimplemented.

The prior full manual all-zero-network check matched stay at 2.72265625 catches and explicitly withheld an Astra success declaration. [Synthetic evaluator report](runs/zero-network-check/report.json). Human play, state/action recording, selected playback and 2,496 reachable single-object probes remain available.

## Frozen provenance

[Tracked commitment](FREEZE_COMMITMENT.json) · [Replacement manifest](runs/frozen-v2/manifest.json) · [Readiness](runs/frozen-v2/readiness.json) · [Clean prompt](runs/frozen-v2/prompt.txt).

Source revision: `62eb67af7336600a4cb88c5efd992bc99bb6a35a`. Every frozen source/test file matches that commit, and the checkout was clean at freeze time.

Replacement manifest SHA-256: `0e5a012c0f91e3901358603a341726b478240a0908aa2a1acd78b63a91fcef0a`.

Original manifest SHA-256: `4247cdc076141710ba60a95f8c7f57052ce78a6c7c019e55b667781e765d3ec8`. Its 10 artifacts are untouched. All 31 original source files were archived before editing under `runs/review-archive/source-v1/`.

The replacement verifies the original artifact commitments and reuses both development and private held-out seed files byte-for-byte. No new hidden seeds were sampled. The 256 hidden episodes remain unscored. The clean game prompt is also byte-identical to the original freeze; only transport/isolation/audit and provenance changed.

Private runs are ignored by Git. The manifest source revision identifies the implementation commit; a later documentation-only commit records its digest. No remote push was performed.

## Login and pending commands

Login must be completed interactively for the dedicated home, if needed. Run:

```sh
mkdir -p /private/tmp/ricochet-astra/home
chmod 700 /private/tmp/ricochet-astra/home
CODEX_HOME=/private/tmp/ricochet-astra/home codex login
```

Check any `config.toml` created by login against the preflight dry-run candidate. The runner refuses a different existing configuration before any model request. Do not copy credentials from the normal home. `/private/tmp` may be cleared on reboot, requiring a new isolated login.

[Current preflight dry run](runs/preflight-review-v2-preview/manifest.json) · [Current generation dry run](runs/generate-review-v2-preview/manifest.json). Both record zero model requests.

The separate, explicit runtime preflight command remains:

```sh
.venv/bin/ricochet-catch preflight --execute --run-dir runs/preflight \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work
```

Only after reviewing the exact declarations/instructions and denied-call/schema evidence and completing every hash-bound gate check, the pending generation command is:

```sh
.venv/bin/ricochet-catch generate --execute --run-dir runs/astra-001 \
  --isolated-home /private/tmp/ricochet-astra/home --workdir /private/tmp/ricochet-astra/work \
  --gate runs/preflight/gate-review-template.json --freeze runs/frozen-v2/manifest.json
```

Neither command was run. Generation requires the unchanged reviewed gate and frozen files, has no automatic retry or repair, and requests GPT-6 Astra with xhigh reasoning and detailed summaries. Official evaluation parses/hashes the unchanged response, reruns frozen correctness tests and then scores the hidden tapes.

## Every frozen SHA-256 hash

| File | SHA-256 |
|---|---|
| [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) | `6b8b3bf05a4f7ffbbf73fe5ce515bef52f00355514f5d4e6d7a2824e0adde7c6` |
| [GAME_DESIGN.md](GAME_DESIGN.md) | `8eb99f9aa4dd6dec76fda92b7b87f4408273b2a75568e00105a4a04807b7ee64` |
| [configs/game.json](configs/game.json) | `c948eb095c3bd95f4c539cf5ffc11fabcb4e319d06fff9368ab47a5fe5004845` |
| [prompts/neutral.txt](prompts/neutral.txt) | `b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb` |
| [pyproject.toml](pyproject.toml) | `f1a4d15025ca9b51a50cc068f35ebafa76d9cda77c4eb4e21996f51f8bfeb00a` |
| [requirements.txt](requirements.txt) | `ffa1e545cccc2839c9044c0e0046adc869049dd673dc601ecf14e21788eabffb` |
| [ricochet_sim.py](ricochet_sim.py) | `ba3ccd2d1a0d63dc32291e523e1e8d25cc2b0d4f1d0482b7fbf6dac0f7e27686` |
| [schemas/tensor-assignments-v2.json](schemas/tensor-assignments-v2.json) | `8339b48933dabce22fb043d17fefa4b6dd0c1d513889ef2859117a963f7cc0c7` |
| [src/ricochet_catch/__init__.py](src/ricochet_catch/__init__.py) | `9e39a1b569861f678fad0471ecc8cee2e72e6a61e35620170414a4708f632acc` |
| [src/ricochet_catch/__main__.py](src/ricochet_catch/__main__.py) | `3129adff95e0f38b87ab34cddfc27082c7892bcaec267ff5d40f160fdf55b44a` |
| [src/ricochet_catch/assignments.py](src/ricochet_catch/assignments.py) | `94c23259c9206443362ee5830571ba4b78eb09db47fa77c1d9955740d5c9fc7a` |
| [src/ricochet_catch/baselines.py](src/ricochet_catch/baselines.py) | `345b7ef921b8ca49c51ba9464b5aa6a140a85ecf210da015693c5451452b0fb2` |
| [src/ricochet_catch/cli.py](src/ricochet_catch/cli.py) | `b603f253d93806af0a95c8f6618777a2eab9c293e2701ef2cc3a7762e1a9c8db` |
| [src/ricochet_catch/env.py](src/ricochet_catch/env.py) | `7643ae3b6b64565ce47f77427acd61312bd6280b706f3e69b97e0d2df6f024ab` |
| [src/ricochet_catch/evaluate.py](src/ricochet_catch/evaluate.py) | `c2aa0a0adf124104dff4f2b61345723ae38d8b94b20792747cb9ed049d1a4eb0` |
| [src/ricochet_catch/freeze.py](src/ricochet_catch/freeze.py) | `074b4edf07fb27acfdfb8dcbfae6e0fd33a14c1de8bcdde08c39f6e26af95f4d` |
| [src/ricochet_catch/observations.py](src/ricochet_catch/observations.py) | `a1126ae62dabe0885d6ca8e2adf07ecf1014ec52ddc5e04dfbf4d9d4ad9e861b` |
| [src/ricochet_catch/policy.py](src/ricochet_catch/policy.py) | `63a5cb91f16ad1b750b801ab499ec90157ff763b3d0bc4f2e5f6d1fc20027a1f` |
| [src/ricochet_catch/probes.py](src/ricochet_catch/probes.py) | `a44dfdcbfc2103ffff067c421a2e15e1d154d362846ab2b94f0055e93751ce46` |
| [src/ricochet_catch/render.py](src/ricochet_catch/render.py) | `74a8597478d6476fc96715ca805602c21f28fe607dc174c074a67391eb04d2f7` |
| [src/ricochet_catch/run_astra.py](src/ricochet_catch/run_astra.py) | `84d627ae7caf4a5a0073ec2d9d08c80b8b208bb2142f0357bbf8f7be3e8246da` |
| [tests/fixtures/authors-breakout/events.jsonl](tests/fixtures/authors-breakout/events.jsonl) | `39ab8479e5f2afac77688634ea3fc54bea6a1f0d41a59eeb97d39b5b65ea3270` |
| [tests/fixtures/authors-breakout/provenance.json](tests/fixtures/authors-breakout/provenance.json) | `41de219466311e2b10565e2b6c9e70597c5832dba5ba52e89def4cef34d32644` |
| [tests/test_assignments.py](tests/test_assignments.py) | `48c6b46dc4c8f1cd36ab5324012bdd8f6231d4efed3f03130ad560632c152d4a` |
| [tests/test_baselines.py](tests/test_baselines.py) | `106cb4d2d644d9cf52e0864963090e0570b6a2dc9421728d7f9ec2181ace8b67` |
| [tests/test_cli.py](tests/test_cli.py) | `c43eb0000685c8b1e49c6e840f0d8a5b2fe0c882a037cd3560ba8e20ffd5ec6e` |
| [tests/test_env.py](tests/test_env.py) | `da53695422e6261f0ed9a015c58c405d29c43f172a8ca0787e5e7b19a76ea995` |
| [tests/test_evaluate.py](tests/test_evaluate.py) | `a7f3e3a0a74f8af42a64fb4277f437ecc2250fac111c6d94296015db4a1472ca` |
| [tests/test_freeze_revision.py](tests/test_freeze_revision.py) | `69a050d2a72e7dd234e2d9cba8f0aedb6091b250b53c6de6da0c86e319c17885` |
| [tests/test_policy.py](tests/test_policy.py) | `0a05723d24f44ca043673d0e02c87b59e4c975ce4d32ca6ab4be9aaccbbab226` |
| [tests/test_probes.py](tests/test_probes.py) | `4ab35de0f3a639874ff7b04e35d6936da7e1a8a487c746cef2a34863019eff32` |
| [tests/test_render.py](tests/test_render.py) | `720c9a2975e69e28e413c68121e6dad726e8daa96d49619cdedde4df109abc90` |
| [tests/test_render_ui.py](tests/test_render_ui.py) | `9f6605f4becf8220325790b1efa521cfcc84507d9ad483e53306980f3b50f0a6` |
| [tests/test_runner.py](tests/test_runner.py) | `f717f06707feaedeaf0ef5bbfdf6e6728aa9d2f07072c63027087512e602b272` |
| [uv.lock](uv.lock) | `ec1a271180670b054648314db3b0e43619910541234e92b3b553c220e5cfc535` |
| [runs/frozen-v2/arithmetic.json](runs/frozen-v2/arithmetic.json) | `f0975ad5c1d69349376f4b999f444644661dc228a6b7cc8dd69352059e6218bf` |
| [runs/frozen-v2/calibration.json](runs/frozen-v2/calibration.json) | `77cb969441c73dd471c028a911626d0916b9c980ce3284f0f49768a98fc5fae3` |
| [runs/frozen-v2/cli-config.candidate.toml](runs/frozen-v2/cli-config.candidate.toml) | `c351180a210adfeb40e86be632e31e4001a21c1e4e6523b49ac5175cd4b07a99` |
| [runs/frozen-v2/cli-validation-status.json](runs/frozen-v2/cli-validation-status.json) | `0f35a903420e693fd37a8ffe8e9e846c7188230e6a4c4f91a5384474ca9fe674` |
| [runs/frozen-v2/development-seeds.json](runs/frozen-v2/development-seeds.json) | `20043feaaaf129ea0f5f5d09c3ce6da5c6af574d40bb1afac88c9df8d94381b8` |
| [runs/frozen-v2/diagnostic-probes.json](runs/frozen-v2/diagnostic-probes.json) | `869f7392b0a02f5e567e8ddee56047bf4d3bb1518bc71460fd6eb4e1152a1269` |
| [runs/frozen-v2/diagnostic-rules.json](runs/frozen-v2/diagnostic-rules.json) | `40266591594b94652a6438d249f385a41395381f4678ab1a862b5543fe84b204` |
| [runs/frozen-v2/prompt.txt](runs/frozen-v2/prompt.txt) | `a8d81fe5c1f69b507447dbd9dbe45fb8b2a75d9db9fb733c3367261e15e28d69` |
| [runs/frozen-v2/readiness.json](runs/frozen-v2/readiness.json) | `c47aeea32d65a59e54b0569346c6bfe75dfb3a775ea0a9640c1c31ba1c3a8bde` |
| [runs/frozen-v2/test-seeds.private.json](runs/frozen-v2/test-seeds.private.json) | `182a1448c3750780208c3ccba2b37d9029693b469f3eec2df9c9de3cb03942aa` |
