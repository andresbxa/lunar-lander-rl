import pytest

from lunar_rl.utils.config import apply_overrides


def test_overrides_parse_types_and_nest():
    cfg = {"train": {"total_timesteps": 1_000_000}, "env": {"enable_wind": False}}
    out = apply_overrides(cfg, ["train.total_timesteps=50000", "env.enable_wind=true", "new.key=0.5"])
    assert out["train"]["total_timesteps"] == 50000
    assert out["env"]["enable_wind"] is True
    assert out["new"]["key"] == 0.5
    # Original is not mutated.
    assert cfg["train"]["total_timesteps"] == 1_000_000


def test_bad_override_raises():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])
