from pathlib import Path

from lunar_rl.ablate import load_suite
from lunar_rl.reproduce import ABLATIONS, ALGO_CONFIGS, build_stages

ROOT = Path(__file__).parents[1]


def test_full_run_trains_every_algorithm_before_evaluating_it():
    names = [name for name, _ in build_stages(skip_training=False, workers=3)]
    for algo in ALGO_CONFIGS:
        assert names.index(f"train {algo}") < names.index(f"evaluate {algo}")
    assert names.index("robustness to wind") > names.index("evaluate ppo")


def test_skip_training_has_no_training_stage():
    stages = build_stages(skip_training=True, workers=3)
    assert not any(name.startswith("train ") for name, _ in stages)
    ablate_argv = dict(stages)["ablations"]
    assert "--evaluate-only" in ablate_argv


def test_referenced_configs_and_ablations_exist():
    for config in ALGO_CONFIGS.values():
        assert (ROOT / config).exists()
    assert set(ABLATIONS) == set(load_suite(ROOT / "configs" / "ablations.yaml"))
