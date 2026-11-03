import json
from pathlib import Path

import pytest

from ricochet_catch.cli import main, _run_frozen_tests
from ricochet_catch.evaluate import verify_replay

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_and_fixture_commands_save_verifiable_replays(tmp_path):
    main(['baseline', '--policy', 'reference', '--episodes', '2', '--out', str(tmp_path / 'baseline')])
    report = json.loads((tmp_path / 'baseline/report.json').read_text())
    assert report['astra_result'] is False
    assert report['summary']['reference']['episodes'] == 2
    for record in (tmp_path / 'baseline/reference').glob('*.json'):
        verify_replay(json.loads(record.read_text()))
    main(['fixtures', '--out', str(tmp_path / 'fixtures')])
    for record in (tmp_path / 'fixtures').glob('*.json'):
        verify_replay(json.loads(record.read_text()))
    caught = json.loads((tmp_path / 'fixtures/prediction_catch.json').read_text())
    missed = json.loads((tmp_path / 'fixtures/prediction_chase_miss.json').read_text())
    hit = json.loads((tmp_path / 'fixtures/conflict_bomb_hit.json').read_text())
    assert caught['trajectory'][-1]['score'] == 1
    assert missed['trajectory'][-1]['score'] == 0
    assert hit['trajectory'][-1]['terminated'] is True


@pytest.mark.parametrize('mode', ['generate', 'preflight'])
def test_default_runner_commands_never_start_processes(tmp_path, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError('A dry run must never start a process')
    monkeypatch.setattr('subprocess.run', forbidden)
    out = tmp_path / mode
    main([mode, '--run-dir', str(out), '--isolated-home', str(tmp_path / 'private-home'),
          '--workdir', str(tmp_path / 'empty-work')])
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['dry_run'] is True
    assert manifest['experimental_requests'] == manifest['smoke_requests'] == 0
    assert not (tmp_path / 'private-home').exists()


def test_invalid_submission_rejected_before_hidden_seed_or_output_access(tmp_path):
    submission = tmp_path / 'bad.json'
    submission.write_text('{"format":"tensor-assignments-v2", "explanation":"", "assignments":[false]}')
    with pytest.raises(SystemExit) as error:
        main(['evaluate', '--submission', str(submission), '--freeze', str(tmp_path / 'does-not-exist'),
              '--generation-run', str(tmp_path / 'also-does-not-exist'), '--out', str(tmp_path / 'output')])
    assert error.value.code == 2
    assert not (tmp_path / 'output').exists()


def test_replay_command_rejects_tampering(tmp_path):
    main(['fixtures', '--out', str(tmp_path / 'fixtures')])
    path = tmp_path / 'fixtures/prediction_catch.json'
    record = json.loads(path.read_text())
    record['trajectory'][-1]['score'] = 999
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit):
        main(['replay', str(path), '--out', str(tmp_path / 'bad.html')])
    assert not (tmp_path / 'bad.html').exists()


def test_post_parse_frozen_suite_saves_results_and_stops_on_failure(tmp_path, monkeypatch):
    import subprocess
    def fail(argv, **kwargs):
        assert argv[1:] == ['-m', 'pytest', '-q']
        assert kwargs['cwd'] == ROOT
        assert kwargs['env']['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        kwargs['stdout'].write(b'1 failed\n')
        return subprocess.CompletedProcess(argv, 1)
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(ValueError, match='hidden scoring not started'):
        _run_frozen_tests(ROOT, tmp_path)
    assert json.loads((tmp_path / 'tests.json').read_text())['exit_status'] == 1
    assert (tmp_path / 'tests.stdout.txt').read_text() == '1 failed\n'
