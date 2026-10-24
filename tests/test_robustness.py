import pandas as pd

from lunar_rl.agents import RandomAgent
from lunar_rl.robustness import evaluate_conditions, plot_robustness, wind_kwargs


def test_wind_kwargs_zero_means_disabled():
    assert wind_kwargs(0.0, 0.0)["enable_wind"] is False
    assert wind_kwargs(10.0, 1.0) == {"enable_wind": True, "wind_power": 10.0, "turbulence_power": 1.0}


def test_evaluate_conditions_rows_and_files(tmp_path):
    df = evaluate_conditions(RandomAgent(seed=0), [(0.0, 0.0), (20.0, 2.0)], n_episodes=2,
                             episodes_dir=tmp_path, stem="random")
    assert df["condition"].tolist() == ["no_wind", "wind20_turb2"]
    assert (df["n_episodes"] == 2).all()
    assert (tmp_path / "random_wind20_turb2.csv").exists()


def test_plot_robustness_writes_png(tmp_path):
    rows = [{"agent": a, "train_seed": s, "wind_power": w, "turbulence_power": w / 10,
             "mean_return": 250 - 5 * w + s, "success_rate": 1 - w / 40}
            for a in ("dqn", "ppo") for s in (0, 1) for w in (0.0, 10.0, 20.0)]
    out = tmp_path / "fig.png"
    plot_robustness(pd.DataFrame(rows), out)
    assert out.stat().st_size > 0
