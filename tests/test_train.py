from pathlib import Path

import pytest

from lunar_rl.train import TRAINERS, train_seed
from lunar_rl.utils import load_config

CONFIGS = sorted(p for p in Path(__file__).parents[1].joinpath("configs").glob("*.yaml")
                 if p.name != "ablations.yaml")  # the ablation suite has its own tests


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg["algo"] in TRAINERS
    assert isinstance(cfg["train"]["total_timesteps"], int)
    assert cfg["eval"]["seed_start"] < 10_000  # validation seeds must not overlap test seeds
    assert all(s < 1_000 for s in cfg["seeds"])  # training seeds must not overlap validation seeds


def test_unknown_algo_raises():
    with pytest.raises(ValueError):
        train_seed({"algo": "nope"}, seed=0)
