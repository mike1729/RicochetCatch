"""Explicit local commands; imports, installation, and tests never request models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .env import Env, GameConfig, Object, PolicyState, make_tape

ROOT = Path(__file__).resolve().parents[2]


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _config(root):
    data = json.loads((root / 'configs/game.json').read_text())
    if data.pop('version') != '0.1':
        raise ValueError('Unsupported game version')
    return GameConfig(**data)


def _directory(path):
    dest = Path(path).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    return dest


def _run_frozen_tests(root, out):
    """Run the committed suite once after artifact hashing, before hidden scoring."""
    import os
    import subprocess
    child_env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    with (out / 'tests.stdout.txt').open('wb') as stdout, (out / 'tests.stderr.txt').open('wb') as stderr:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root,
                                env=child_env, stdout=stdout, stderr=stderr, timeout=60, check=False)
    report = {'exit_status': result.returncode,
              'stdout_sha256': hashlib.sha256((out / 'tests.stdout.txt').read_bytes()).hexdigest(),
              'stderr_sha256': hashlib.sha256((out / 'tests.stderr.txt').read_bytes()).hexdigest()}
    _json(out / 'tests.json', report)
    if result.returncode:
        raise ValueError('Frozen correctness tests failed; artifact preserved, hidden scoring not started')
    return report


def _save_selected(results, out):
    from .evaluate import save_episode, select_replays
    from .render import write_replay_html
    indexed = {episode.episode_index: episode for episode in results}
    selections = select_replays(results)
    for selection, index in selections.items():
        if index is not None:
            path = out / f'{selection}.json'
            save_episode(indexed[index], path, selection=selection)
            write_replay_html(json.loads(path.read_text()), path.with_suffix('.html'))
    return selections


def _baseline(args, root, config):
    from .baselines import BASELINE_NAMES
    from .evaluate import evaluate_baselines, report_summary
    names = BASELINE_NAMES if args.policy == 'all' else [args.policy]
    out = _directory(args.out)
    results = evaluate_baselines(range(args.seed, args.seed + args.episodes), config, names, record=True)
    report = {'kind': 'development_baseline', 'astra_result': False,
              'seeds': list(range(args.seed, args.seed + args.episodes)),
              'summary': {name: report_summary(episodes) for name, episodes in results.items()}}
    for name, episodes in results.items():
        folder = out / name
        folder.mkdir()
        _save_selected(episodes, folder)
    _json(out / 'report.json', report)
    print(json.dumps({'report': str(out / 'report.json'), 'means': {name: r['mean'] for name, r in report['summary'].items()}}))


def _calibrate(args, root, config):
    from .evaluate import calibrate
    output = Path(args.out).resolve()
    if output.exists():
        raise ValueError('Calibration output already exists; preserve it and choose a new path')
    report = calibrate(config)
    _json(output, report)
    print(json.dumps({'report': str(output), 'passed': report['passed'],
                      'means': {name: row['mean'] for name, row in report['summary'].items()},
                      'trivial_comparator': report['trivial_comparator']}))


def _evaluate(args, root, config):
    import numpy as np
    from .assignments import load_assignments, load_tensors, save_tensors
    from .evaluate import (build_report, evaluate_baselines, evaluate_policy, runtime_metadata)
    from .policy import NeuralPolicy, PARAMETER_COUNT
    from .probes import run_probes

    # Invalid artifacts fail before any hidden seed is opened or scored.
    if args.submission:
        parsed = load_assignments(args.submission)
        tensors, parsing = parsed.tensors, parsed.report()
        raw = Path(args.submission).read_bytes()
    else:
        tensors = load_tensors(args.tensors)
        parsing, raw = {'format': 'validated-numeric-npz', 'parameter_count': PARAMETER_COUNT}, None
    policy = NeuralPolicy(tensors, 'submitted network' if args.generation_run else 'manual/synthetic network')
    audit_valid = False
    if args.freeze:
        if not args.generation_run or not args.submission:
            raise ValueError('Held-out evaluation requires --generation-run and its unchanged --submission')
        from .freeze import verify_freeze
        from .run_astra import validate_generation_run
        frozen = verify_freeze(root, args.freeze)
        validation = validate_generation_run(root, args.generation_run, args.freeze)
        if Path(args.submission).read_bytes() != (Path(args.generation_run) / 'response.json').read_bytes():
            raise ValueError('Submission differs from the archived generation response')
        # Official scoring requires archived, valid provenance and a clean audit.
        audit_valid = validation['audit_valid']
        seed_manifest = json.loads((Path(args.freeze).resolve().parent / 'test-seeds.private.json').read_text())
        seeds, comparator = seed_manifest['seeds'], frozen['trivial_comparator']
        bootstrap_seed = frozen['bootstrap_seed']
        calibration_path = Path(args.freeze).resolve().parent / 'calibration.json'
        calibration = json.loads(calibration_path.read_text())
        random_seed0 = seed_manifest['random_action_seed0']
    else:
        if args.generation_run:
            raise ValueError('--generation-run requires the corresponding --freeze')
        seeds, comparator, bootstrap_seed, random_seed0 = list(range(256)), 'chase', 12345, 10_000_019
        calibration = None
        validation = {'status': 'manual/synthetic development evaluation; no Astra generation'}
    out = _directory(args.out)
    if raw is not None:
        (out / 'response.original.json').write_bytes(raw)
    save_tensors(out / 'tensors.npz', tensors)
    tensor_sha = hashlib.sha256((out / 'tensors.npz').read_bytes()).hexdigest()
    _json(out / 'parse-report.json', {**parsing, 'valid': True, 'tensor_sha256': tensor_sha})
    frozen_tests = _run_frozen_tests(root, out) if args.freeze else None
    results = evaluate_policy(lambda _: policy, seeds, config, policy.name)
    baselines = evaluate_baselines(seeds, config, random_seed0=random_seed0)
    report = build_report(results, baselines, comparator, parse_valid=True, audit_valid=audit_valid,
                          bootstrap_seed=bootstrap_seed, metadata={
                              'runtime': runtime_metadata(), 'generation': validation,
                              'frozen_correctness_tests': frozen_tests,
                              'tensor_sha256': tensor_sha, 'parameter_count': PARAMETER_COUNT,
                              'nonzero_parameters': sum(int(np.count_nonzero(t)) for t in tensors.values()),
                              'tensors': {name: {'nonzero': int(np.count_nonzero(t)),
                                                'min': float(t.min()), 'max': float(t.max()),
                                                'max_absolute': float(np.abs(t).max())} for name, t in tensors.items()}})
    if not args.freeze:
        report.update(kind='manual_synthetic_development_evaluation', astra_result=False,
                      successful_demonstration=False)
    _json(out / 'episodes.json', {name: [r.to_dict(False) for r in rows] for name, rows in {'network': results, **baselines}.items()})
    _save_selected(results, out)
    _json(out / 'probes.json', run_probes(lambda _: policy))
    _json(out / 'report.json', report)
    print(json.dumps({'report': str(out / 'report.json'), 'mean': report['summary']['network']['mean'],
                      'successful_demonstration': report['successful_demonstration'], 'kind': report['kind']}))


def _fixtures(args):
    from .evaluate import state_record
    from .render import write_replay_html
    output = _directory(args.out)
    for name, objects, actions in [
        ('prediction_catch', (Object('gem', 4, 6, 1),), [2, 2, 0]),
        ('prediction_chase_miss', (Object('gem', 4, 6, 1),), [2, 2, 2]),
        ('conflict_safe', (Object('bomb', 6, 4, 0), Object('gem', 5, 5, 0)), [0, 2]),
        ('conflict_bomb_hit', (Object('bomb', 6, 4, 0), Object('gem', 5, 5, 0)), [2]),
    ]:
        env = Env((None,) * 256, initial_state=PolicyState(3, objects, 248), spawning=False)
        trajectory = [{'tick': 8, 'state': state_record(env.state), 'score': 0, 'action': None,
                       'logits': None, 'events': [], 'reward': 0, 'terminated': False, 'truncated': False}]
        for action in actions:
            step = env.step(action)
            trajectory.append({'tick': env.elapsed_ticks, 'state': state_record(env.state), 'score': env.score,
                               'action': action, 'logits': None, 'reward': step.reward,
                               'terminated': step.terminated, 'truncated': step.truncated,
                               'events': [{'kind': e.kind, 'object': e.object._asdict() if e.object else None,
                                           'paddle_column': e.paddle_column} for e in step.events]})
        record = {'format': 'ricochet-replay-v1', 'policy_name': 'Scripted mechanics fixture',
                  'selection': 'Synthetic correctness fixture: ' + name, 'trajectory': trajectory}
        _json(output / f'{name}.json', record)
        write_replay_html(record, output / f'{name}.html')
    print(str(output))


def parser():
    from .baselines import BASELINE_NAMES
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT, help='Implementation repository root')
    sub = p.add_subparsers(dest='command', required=True)
    play = sub.add_parser('play', help='Play locally using the real game transition')
    play.add_argument('--seed', type=int, default=0)
    play.add_argument('--save', default='runs/human/replay.json')
    play.add_argument('--port', type=int, default=0)
    play.add_argument('--no-browser', action='store_true')
    baseline = sub.add_parser('baseline', help='Development baselines with selected replays')
    baseline.add_argument('--policy', choices=['all', *BASELINE_NAMES], default='reference')
    baseline.add_argument('--episodes', type=int, default=256)
    baseline.add_argument('--seed', type=int, default=0)
    baseline.add_argument('--out', default='runs/baseline')
    calibration = sub.add_parser('calibrate', help='Calibrate on the fixed 256 development tapes')
    calibration.add_argument('--out', default='runs/development/calibration.json')
    freeze = sub.add_parser('freeze', help='Commit hashes, hidden seeds and readiness without scoring test tapes')
    freeze.add_argument('--out', default='runs/frozen-v2')
    freeze.add_argument('--calibration', default='runs/development/calibration.json')
    freeze.add_argument('--preflight', default=None)
    freeze.add_argument('--previous-freeze', type=Path, help='Reuse the prior hidden seed commitment in a new freeze')
    for command in ['preflight', 'generate']:
        runner = sub.add_parser(command, help='Prepare a dry run; requests require explicit --execute')
        mode = runner.add_mutually_exclusive_group()
        mode.add_argument('--dry-run', action='store_true')
        mode.add_argument('--execute', action='store_true')
        runner.add_argument('--run-dir', default=f'runs/{command}-dry-run')
        runner.add_argument('--isolated-home', default='/private/tmp/ricochet-astra/home')
        runner.add_argument('--workdir', default='/private/tmp/ricochet-astra/work')
        runner.add_argument('--gate')
        runner.add_argument('--freeze')
        runner.add_argument('--timeout', type=int, default=2700)
    evaluation = sub.add_parser('evaluate', help='Evaluate a manual artifact on development tapes or a frozen submission')
    artifact = evaluation.add_mutually_exclusive_group(required=True)
    artifact.add_argument('--submission', type=Path)
    artifact.add_argument('--tensors', type=Path)
    evaluation.add_argument('--freeze', type=Path)
    evaluation.add_argument('--generation-run', type=Path)
    evaluation.add_argument('--out', default='runs/evaluation')
    replay = sub.add_parser('replay', help='Verify recorded physics and export standalone playback')
    replay.add_argument('record', type=Path)
    replay.add_argument('--out', type=Path)
    replay.add_argument('--open', action='store_true')
    fixtures = sub.add_parser('fixtures', help='Export prediction and bomb-conflict correctness fixtures')
    fixtures.add_argument('--out', default='runs/fixtures')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = args.root.resolve()
        config = _config(root)
        if args.command == 'play':
            from .render import play
            play(args.seed, args.save, args.port, not args.no_browser, config)
        elif args.command == 'baseline':
            if args.episodes < 1:
                raise ValueError('Episode count must be positive')
            _baseline(args, root, config)
        elif args.command == 'calibrate':
            _calibrate(args, root, config)
        elif args.command == 'freeze':
            from .freeze import freeze_project
            result = freeze_project(root, args.out, args.calibration, args.preflight, args.previous_freeze)
            print(json.dumps({'manifest': str(Path(args.out).resolve() / 'manifest.json'),
                              'calibration_passed': result['calibration_passed'], 'runtime_preflight': result['runtime_preflight']}))
        elif args.command in ('preflight', 'generate'):
            from .run_astra import prepare_run
            result = prepare_run(root, args.run_dir, args.isolated_home, args.workdir,
                                 mode=args.command, execute=args.execute, gate_path=args.gate,
                                 freeze_path=args.freeze, timeout=args.timeout)
            print(json.dumps({'manifest': str(Path(args.run_dir).resolve() / 'manifest.json'),
                              'dry_run': result['dry_run'], 'experimental_requests': result['experimental_requests'],
                              'smoke_requests': result['smoke_requests'], 'runtime_preflight': result['runtime_preflight']}))
        elif args.command == 'evaluate':
            _evaluate(args, root, config)
        elif args.command == 'replay':
            from .evaluate import verify_replay
            from .render import write_replay_html
            record = json.loads(args.record.read_text())
            verify_replay(record)
            out = write_replay_html(record, args.out or args.record.with_suffix('.html')).resolve()
            if args.open:
                import webbrowser
                webbrowser.open(out.as_uri())
            print(str(out))
        elif args.command == 'fixtures':
            _fixtures(args)
    except (ValueError, OSError, KeyError) as error:
        print(f'ricochet: {error}', file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == '__main__':
    main()
