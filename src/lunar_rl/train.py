"""Generic training entry point: picks the trainer from the config's ``algo`` key.

Usage::

    python -m lunar_rl.train --config configs/dqn.yaml --workers 3      # all seeds in parallel
    python -m lunar_rl.train --config configs/sb3_ppo.yaml --seed 0
    python -m lunar_rl.train --config configs/dqn.yaml --seed 0 --set train.total_timesteps=50000
"""

from __future__ import annotations

import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lunar_rl.utils import get_logger, load_config

log = get_logger("train")

TRAINERS = {
    "sb3_ppo": "lunar_rl.train_sb3",
    "dqn": "lunar_rl.train_dqn",
}


def train_seed(cfg: dict[str, Any], seed: int, progress_bar: bool = True) -> dict[str, Any]:
    algo = cfg.get("algo")
    if algo not in TRAINERS:
        raise ValueError(f"Config 'algo' must be one of {sorted(TRAINERS)}, got {algo!r}")
    return importlib.import_module(TRAINERS[algo]).train_one(cfg, seed, progress_bar)


def run(cfg: dict[str, Any], seeds: list[int], workers: int = 1) -> list[dict[str, Any]]:
    """Train every seed, optionally in parallel processes (one process per seed)."""
    if workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(train_seed, [cfg] * len(seeds), seeds, [False] * len(seeds)))
    else:
        results = [train_seed(cfg, s) for s in seeds]
    for r in results:
        log.info("seed %d: %.1f min, %.0f steps/s, best val %.1f",
                 r["seed"], r["minutes"], r["fps"], r["best_val_return"])
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=None, help="Train a single seed (default: all in config).")
    p.add_argument("--workers", type=int, default=1, help="Seeds trained in parallel processes.")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Config overrides, e.g. train.total_timesteps=50000")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
    run(cfg, seeds, args.workers)


if __name__ == "__main__":
    main()
