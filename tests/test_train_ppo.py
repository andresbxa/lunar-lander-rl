import numpy as np
import pandas as pd

from lunar_rl.agents import load_agent
from lunar_rl.train_ppo import bootstrap_truncations, train_one


def test_bootstrap_only_truncated_not_terminated():
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    terminated = np.array([False, True, False, True])
    truncated = np.array([False, False, True, True])  # env 3: both flags -> treated as terminal
    final_obs = np.array([None, np.zeros(8), np.full(8, 2.0), np.zeros(8)], dtype=object)

    def value_fn(o):
        return o[:, 0] * 10.0  # V = 20 for env 2's final obs

    out = bootstrap_truncations(rewards, terminated, truncated, final_obs, value_fn, gamma=0.5)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0 + 0.5 * 20.0, 1.0])
    np.testing.assert_allclose(rewards, 1.0)  # input not modified


def test_short_ppo_training_produces_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "name": "ppo_test",
        "env": {"enable_wind": False},
        "train": {"total_timesteps": 512, "n_envs": 2},
        "eval": {"freq": 256, "n_episodes": 1, "seed_start": 1000},
        "ppo": {"hidden_sizes": [16], "n_steps": 128, "batch_size": 64, "n_epochs": 2},
    }
    result = train_one(cfg, seed=0, progress_bar=False)

    assert result["seed"] == 0
    curve = pd.read_csv(tmp_path / "results/curves/ppo_test/seed_0.csv")
    assert curve["timesteps"].tolist() == [0, 256, 512]
    for f in ("best_model.pt", "final_model.pt", "config.yaml"):
        assert (tmp_path / "models/ppo_test/seed_0" / f).exists()
    load_agent("ppo", tmp_path / "models/ppo_test/seed_0/best_model.pt")
