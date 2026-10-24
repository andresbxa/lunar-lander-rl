from pathlib import Path

import pandas as pd

from lunar_rl.ablate import load_suite, report, variant_config, verdict
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


def test_baselines_are_in_the_suite_without_overrides():
    suite = load_suite(SUITE)
    for name in ("dqn", "ppo"):
        assert suite[name]["baseline"] is True and suite[name]["set"] == {}


def test_verdict_rule():
    assert verdict(+30.0, 10.0, 20.0) == "better (likely real)"
    assert verdict(-30.0, 10.0, 20.0) == "worse (likely real)"
    assert verdict(+15.0, 10.0, 20.0) == "inconclusive"  # smaller than one of the stds


def test_report_compares_both_checkpoints_with_the_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(SUITE.parents[1])
    suite = {k: v for k, v in load_suite(SUITE).items() if k in ("ppo", "ppo_big_net")}
    rows = []
    for variant, best, final in [("ppo", 260.0, 255.0), ("ppo_big_net", 300.0, 200.0)]:
        for ckpt, mean in (("best", best), ("final", final)):
            rows.append({"variant": variant, "checkpoint": ckpt, "train_seed": "all",
                         "mean_return": mean, "std_return": 10.0, "success_rate": 1.0})
    pd.DataFrame(rows).to_csv(tmp_path / "abl.csv", index=False)
    table = report(suite, tmp_path / "abl.csv")
    assert "| **`ppo`** | baseline | 260.0 ± 10.0 | – | 255.0 ± 10.0 | – | – |" in table
    assert "| `ppo_big_net` |" in table
    assert "+40.0" in table and "-55.0" in table
    assert "better (likely real) / worse (likely real)" in table
