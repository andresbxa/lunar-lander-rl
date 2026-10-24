"""Reproduce every result, figure and GIF in the README.

Each stage runs exactly the command documented in the README, in a fresh
Python process, and stops at the first failure.

Usage::

    python -m lunar_rl.reproduce                    # everything (~4-5 h on a laptop CPU)
    python -m lunar_rl.reproduce --skip-training    # re-evaluate saved models, redo figures and GIFs
    python -m lunar_rl.reproduce --dry-run          # print the commands only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from lunar_rl.utils import get_logger

log = get_logger("reproduce")

ALGO_CONFIGS = {"sb3_ppo": "configs/sb3_ppo.yaml", "dqn": "configs/dqn.yaml", "ppo": "configs/ppo.yaml"}
MODEL_GLOBS = {"sb3_ppo": "models/sb3_ppo/seed_*/best_model.zip", "dqn": "models/dqn/seed_*/best_model.pt",
               "ppo": "models/ppo/seed_*/best_model.pt"}
ALGO_BASELINES = ["dqn", "ppo"]  # from-scratch algorithms the ablations are compared with
ABLATIONS = ["dqn_double", "dqn_no_target", "ppo_big_net", "ppo_lr_1e-4", "ppo_lr_1e-3", "ppo_gamma_0.99"]
CURVE_FIGURES = [["dqn", "ppo", "sb3_ppo"], ["ppo", "sb3_ppo"], ["dqn", "sb3_ppo"], ["dqn"], ["ppo"], ["sb3_ppo"],
                 ["dqn", "dqn_double", "dqn_no_target"], ["ppo", "ppo_big_net", "ppo_lr_1e-4", "ppo_lr_1e-3"],
                 ["ppo", "ppo_gamma_0.99"]]
GIFS = {"random": (None, "Random agent"), "dqn": ("models/dqn/seed_0/best_model.pt", "DQN (ours)"),
        "ppo": ("models/ppo/seed_0/best_model.pt", "PPO (ours)"),
        "sb3_ppo": ("models/sb3_ppo/seed_0/best_model.zip", "SB3 PPO")}


def build_stages(skip_training: bool, workers: int) -> list[tuple[str, list[str]]]:
    """Return (stage name, argv after ``python -m``) in execution order."""
    stages: list[tuple[str, list[str]]] = [("random agent", ["lunar_rl.evaluate", "--agent", "random"])]
    if not skip_training:
        for algo, config in ALGO_CONFIGS.items():
            stages.append((f"train {algo}", ["lunar_rl.train", "--config", config, "--workers", str(workers)]))
    for algo, pattern in MODEL_GLOBS.items():
        stages.append((f"evaluate {algo}", ["lunar_rl.evaluate", "--agent", algo, "--model", pattern]))
    stages.append(("robustness to wind", ["lunar_rl.robustness", "--workers", str(workers)]))
    ablate = ["lunar_rl.ablate", "--variants", *ABLATIONS, "--workers", str(workers)]
    stages.append(("ablations", ablate + (["--evaluate-only"] if skip_training else [])))
    stages.append(("ablation baselines (best and final checkpoints)",
                   ["lunar_rl.ablate", "--variants", *ALGO_BASELINES, "--evaluate-only", "--workers", str(workers)]))
    for algos in CURVE_FIGURES:
        stages.append((f"curves {' vs '.join(algos)}", ["lunar_rl.plot_curves", "--algo", *algos, "--ymin", "-800"]))
    for agent, (model, label) in GIFS.items():
        argv = ["lunar_rl.record", "--agent", agent, "--label", label, "--out",
                f"assets/{'random_agent' if agent == 'random' else agent}.gif"]
        stages.append((f"GIF {agent}", argv + (["--model", model] if model else [])))
    return stages


def keep_awake() -> None:
    """On Windows, block idle sleep and display-off standby while this process lives."""
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x1 | 0x2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-training", action="store_true", help="Use the models already in models/.")
    p.add_argument("--workers", type=int, default=3, help="Parallel processes for training / evaluation.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    stages = build_stages(args.skip_training, args.workers)
    if args.dry_run:
        for name, argv in stages:
            print(f"# {name}\npython -m {' '.join(argv)}")
        return

    keep_awake()
    t0 = time.perf_counter()
    for i, (name, argv) in enumerate(stages, 1):
        log.info("[%d/%d] %s", i, len(stages), name)
        subprocess.run([sys.executable, "-m", *argv], check=True)
    log.info("All %d stages done in %.1f min", len(stages), (time.perf_counter() - t0) / 60)


if __name__ == "__main__":
    main()
