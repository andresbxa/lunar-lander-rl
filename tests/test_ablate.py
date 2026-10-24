from pathlib import Path

import pandas as pd

from lunar_rl.ablate import load_suite, report, variant_config
from lunar_rl.utils import load_config

SUITE = Path(__file__).parents[1] / "configs" / "ablations.yaml"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def test_each_variant_changes_exactly_one_setting(monkeypatch):
    monkeypatch.chdir(SUITE.parents[1])
    for name, spec in load_suite(SUITE).items():
        cfg = variant_config(name, spec)
        base = load_config(spec["base"])
        diff = {k for k, v in _flatten(cfg).items() if _flatten(base).get(k) != v} - {"name"}
        assert diff == set(spec["set"]), name
        assert cfg["name"] == name
        assert cfg["seeds"] == base["seeds"]


def test_variant_override_types(monkeypatch):
    monkeypatch.chdir(SUITE.parents[1])
    suite = load_suite(SUITE)
    assert variant_config("ppo_big_net", suite["ppo_big_net"])["ppo"]["hidden_sizes"] == [256, 256]
    assert variant_config("ppo_lr_1e-4", suite["ppo_lr_1e-4"])["ppo"]["learning_rate"] == 1e-4
    assert variant_config("dqn_no_target", suite["dqn_no_target"])["dqn"]["use_target_network"] is False


def test_report_verdicts(tmp_path, monkeypatch):
    monkeypatch.chdir(SUITE.parents[1])
    suite = {k: v for k, v in load_suite(SUITE).items() if k in ("ppo_big_net", "ppo_gamma_0.99")}
    pd.DataFrame([
        {"variant": "ppo_big_net", "train_seed": "all", "mean_return": 262.0, "std_return": 8.0, "success_rate": 1.0},
        {"variant": "ppo_gamma_0.99", "train_seed": "all", "mean_return": 150.0, "std_return": 20.0,
         "success_rate": 0.5},
    ]).to_csv(tmp_path / "abl.csv", index=False)
    pd.DataFrame([{"agent": "ppo", "train_seed": "all", "condition": "no_wind", "mean_return": 260.0,
                   "std_return": 10.0, "success_rate": 1.0}]).to_csv(tmp_path / "summary.csv", index=False)
    table = report(suite, tmp_path / "abl.csv", tmp_path / "summary.csv")
    assert "| `ppo_big_net` |" in table and "inconclusive" in table
    assert "worse (likely real)" in table
