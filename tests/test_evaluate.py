import pandas as pd

from lunar_rl.agents import RandomAgent
from lunar_rl.evaluate import (
    EPISODE_COLUMNS,
    SUMMARY_COLUMNS,
    aggregate_over_seeds,
    evaluate,
    summarize,
    upsert_summary,
)


def test_evaluate_returns_expected_format():
    df = evaluate(RandomAgent(seed=0), n_episodes=3, seed_start=10_000)
    assert list(df.columns) == EPISODE_COLUMNS
    assert len(df) == 3
    assert df["env_seed"].tolist() == [10_000, 10_001, 10_002]
    assert (df["length"] > 0).all()
    assert (df["terminated"] | df["truncated"]).all()


def test_evaluate_is_reproducible():
    a = evaluate(RandomAgent(seed=0), n_episodes=3)
    b = evaluate(RandomAgent(seed=0), n_episodes=3)
    pd.testing.assert_frame_equal(a, b)


def test_summarize_and_aggregate():
    df = evaluate(RandomAgent(seed=0), n_episodes=3)
    stats = summarize(df)
    assert stats["n_episodes"] == 3
    assert 0.0 <= stats["success_rate"] <= 1.0
    agg = aggregate_over_seeds([{**stats, "mean_return": 100.0}, {**stats, "mean_return": 200.0}])
    assert agg["mean_return"] == 150.0
    assert agg["n_seeds"] == 2
    assert agg["std_over"] == "train_seeds"


def test_upsert_summary_replaces_same_key(tmp_path):
    path = tmp_path / "summary.csv"
    base = {"agent": "random", "train_seed": "na", "condition": "no_wind", "n_seeds": 1,
            "std_over": "episodes", "n_episodes": 10, "mean_return": -100.0, "std_return": 1.0,
            "success_rate": 0.0, "mean_length": 90.0, "truncated_rate": 0.0}
    upsert_summary(path, base)
    upsert_summary(path, {**base, "mean_return": -50.0})
    upsert_summary(path, {**base, "condition": "wind15_turb1.5"})
    df = pd.read_csv(path)
    assert list(df.columns) == SUMMARY_COLUMNS
    assert len(df) == 2
    assert df.loc[df["condition"] == "no_wind", "mean_return"].item() == -50.0
