import json

from ricochet_catch.env import Env, Object, PolicyState
from ricochet_catch.render import HumanGame, render_html, write_replay_html


def test_live_controls_record_same_physics_and_restart(tmp_path):
    game = HumanGame(17, tmp_path / 'human.json')
    comparison = Env(game.tape)
    for action in [2, 2, 0, 1, 0, 0, 2, 1]:
        expected = comparison.step(action)
        frame = game.step(action)
        assert frame['score'] == comparison.score
        assert frame['state']['paddle_column'] == comparison.state.paddle_column
        assert frame['state']['objects'] == [o._asdict() for o in comparison.state.objects]
        assert frame['reward'] == expected.reward
    saved = json.loads((tmp_path / 'human.json').read_text())
    assert len(saved['trajectory']) == 9
    assert game.restart()['tick'] == 0
    assert game.env.state == PolicyState(3, (), 256)


def test_playback_embeds_recorded_states_safely(tmp_path):
    game = HumanGame()
    game.record['policy_name'] = '</script><script>alert(1)</script>'
    html = render_html(game.record)
    assert '</script><script>alert(1)</script>' not in html
    assert 'const data=' in html and 'live=false' in html
    assert 'setInterval' in html and 'requestAnimationFrame' not in html
    assert 'Game over · bomb hit' in html and 'Game over · time limit' in html
    assert 'aria-label="Eight by eight game board"' in html
    result = write_replay_html(game.record, tmp_path / 'replay.html')
    assert result.read_text() == html


def test_record_content_cannot_be_replaced_by_template_tokens():
    record = HumanGame().record
    record['selection'] = '__LIVE__ __DATA__'
    assert '"selection": "__LIVE__ __DATA__"' in render_html(record)


def test_renderer_has_no_effect_on_trajectory_or_rng():
    first, second = HumanGame(9), HumanGame(9)
    for action in [0, 2, 1] * 10:
        if first.env.done:
            break
        render_html(first.record)
        assert first.step(action) == second.step(action)
    assert first.tape == second.tape
