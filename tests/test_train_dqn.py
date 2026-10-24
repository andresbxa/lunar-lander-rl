import pandas as pd

from lunar_rl.agents import load_agent
from lunar_rl.train_dqn import train_one


def test_short_dqn_training_produces_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # training writes to relative models/, runs/, results/
    cfg = {
        "name": "dqn_test",
        "env": {"enable_wind": False},
        "train": {"total_timesteps": 2_000, "log_interval": 500},
        "eval": {"freq": 1_000, "n_episodes": 1, "seed_start": 1000},
        "dqn": {"hidden_sizes": [16], "buffer_size": 1_000, "learning_starts": 500,
                "batch_size": 32, "target_update_interval": 100, "eps_decay_steps": 1_000},
    }
    result = train_one(cfg, seed=0, progress_bar=False)

    assert result["seed"] == 0
    curve = pd.read_csv(tmp_path / "results/curves/dqn_test/seed_0.csv")
    assert curve["timesteps"].tolist() == [0, 1_000, 2_000]
    for f in ("best_model.pt", "final_model.pt", "config.yaml"):
        assert (tmp_path / "models/dqn_test/seed_0" / f).exists()
    assert any((tmp_path / "runs/dqn_test/seed_0").iterdir())
    load_agent("dqn", tmp_path / "models/dqn_test/seed_0/best_model.pt")
