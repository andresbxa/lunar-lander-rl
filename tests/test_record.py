from lunar_rl.agents import RandomAgent
from lunar_rl.record import record_episode, save_gif


def test_record_episode_and_save_gif(tmp_path):
    frames, total = record_episode(RandomAgent(seed=0), seed=10_000, label="test", width=200)
    assert len(frames) > 1
    assert frames[0].shape == (133, 200, 3)
    out = tmp_path / "ep.gif"
    save_gif(frames, out)
    assert out.stat().st_size > 0
