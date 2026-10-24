import pytest
import pandas as pd

from lunar_rl.agents import RandomAgent
from lunar_rl.validation import PeriodicValidator


def test_validator_schedule_csv_and_best(tmp_path):
    saves = []
    v = PeriodicValidator(eval_freq=100, n_episodes=2, seed_start=1000, env_kwargs={},
                          curve_path=tmp_path / "curve.csv", save_best=lambda: saves.append(1))
    agent = RandomAgent(seed=0)

    assert v.maybe_validate(agent, 0) is not None   # step 0 is always due
    assert v.maybe_validate(agent, 50) is None       # not due yet
    stats = v.maybe_validate(agent, 130)             # due (>= 100)
    assert stats is not None and "best_mean_return" in stats
    assert v.next_eval == 200                        # schedule stays on the 100-step grid
    assert v.last_step == 130

    df = pd.read_csv(tmp_path / "curve.csv")
    assert df["timesteps"].tolist() == [0, 130]
    assert len(saves) >= 1                           # first evaluation is always a new best
    assert v.best_mean == pytest.approx(df["mean_return"].max(), abs=1e-3)  # CSV keeps 3 decimals
