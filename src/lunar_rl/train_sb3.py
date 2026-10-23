"""Train the Stable-Baselines3 PPO reference baseline.

For every training seed this produces:

* ``models/<name>/seed_<s>/best_model.zip``  best checkpoint on *validation* seeds
* ``models/<name>/seed_<s>/final_model.zip`` weights at the end of training
* ``runs/<name>/seed_<s>_1/``                TensorBoard logs
* ``results/curves/<name>/seed_<s>.csv``     validation return vs. timesteps

Usage::

    python -m lunar_rl.train_sb3 --config configs/sb3_ppo.yaml                 # all seeds, sequential
    python -m lunar_rl.train_sb3 --config configs/sb3_ppo.yaml --workers 3     # all seeds in parallel
    python -m lunar_rl.train_sb3 --config configs/sb3_ppo.yaml --seed 0 --set train.total_timesteps=50000
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from lunar_rl.agents.sb3_agent import SB3Agent
from lunar_rl.envs import make_env
from lunar_rl.evaluate import evaluate, summarize
from lunar_rl.utils import get_logger, load_config, save_config, set_global_seed

log = get_logger("train_sb3")


class PeriodicEvalCallback(BaseCallback):
    """Evaluate on fixed validation seeds every ``eval_freq`` steps and keep the best model.

    Unlike SB3's ``EvalCallback`` this resets every evaluation episode with an
    explicit seed, so successive evaluations are directly comparable (same
    terrains every time) and reuse exactly the same code path as final testing.
    """

    def __init__(self, eval_freq: int, n_episodes: int, seed_start: int, env_kwargs: dict[str, Any],
                 best_model_path: Path, curve_path: Path, tag: str = ""):
        super().__init__()
        self.tag = tag
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.seed_start = seed_start
        self.env_kwargs = env_kwargs
        self.best_model_path = best_model_path
        self.curve_path = curve_path
        self.best_mean = -float("inf")
        self.next_eval = 0
        self.rows: list[dict[str, float]] = []
        self.t0 = time.perf_counter()

    def _run_eval(self) -> None:
        df = evaluate(SB3Agent(self.model), self.n_episodes, self.seed_start, True, self.env_kwargs)
        stats = summarize(df)
        row = {"timesteps": self.num_timesteps, **stats, "wall_time_s": time.perf_counter() - self.t0}
        self.rows.append(row)
        # Rewrite the whole CSV each time so a crashed run still leaves a usable curve.
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.curve_path, index=False, float_format="%.3f")

        for k in ("mean_return", "std_return", "success_rate", "mean_length", "truncated_rate"):
            self.logger.record(f"eval/{k}", stats[k])
        is_best = stats["mean_return"] > self.best_mean
        if is_best:
            self.best_mean = stats["mean_return"]
            self.model.save(self.best_model_path)
        self.logger.record("eval/best_mean_return", self.best_mean)
        self.logger.dump(self.num_timesteps)
        log.info("%s step %8d | val return %7.1f +/- %5.1f | success %3.0f%% | best %7.1f%s",
                 self.tag, self.num_timesteps, stats["mean_return"], stats["std_return"],
                 100 * stats["success_rate"], self.best_mean, "  *saved*" if is_best else "")

    def _on_training_start(self) -> None:
        self.t0 = time.perf_counter()
        self._run_eval()  # step-0 point: the untrained policy
        self.next_eval = self.eval_freq

    def _on_rollout_start(self) -> None:
        # PPO only changes its weights between rollouts, so this is the one
        # place where a new evaluation can measure a new policy.
        if self.num_timesteps >= self.next_eval:
            self._run_eval()
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.eval_freq

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["timesteps"] != self.num_timesteps:
            self._run_eval()


def train_one(cfg: dict[str, Any], seed: int, progress_bar: bool = True) -> dict[str, Any]:
    name, train_cfg, eval_cfg = cfg["name"], cfg["train"], cfg["eval"]
    torch.set_num_threads(train_cfg.get("torch_threads", 1))
    set_global_seed(seed)

    model_dir = Path("models") / name / f"seed_{seed}"
    curve_path = Path("results/curves") / name / f"seed_{seed}.csv"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "seed": seed}, model_dir / "config.yaml")
    save_config(cfg, curve_path.parent / "config.yaml")

    # make_vec_env wraps each env in Monitor (episode stats -> TensorBoard rollout/ep_rew_mean)
    # and seeds sub-env i with seed + i.
    vec_env = make_vec_env(make_env, n_envs=train_cfg["n_envs"], seed=seed,
                           env_kwargs=cfg["env"], vec_env_cls=DummyVecEnv)
    model = PPO("MlpPolicy", vec_env, seed=seed, device=train_cfg.get("device", "cpu"),
                tensorboard_log=str(Path("runs") / name), verbose=0, **cfg["ppo"])
    callback = PeriodicEvalCallback(eval_cfg["freq"], eval_cfg["n_episodes"], eval_cfg["seed_start"],
                                    cfg["env"], model_dir / "best_model", curve_path,
                                    tag=f"[seed {seed}]")

    log.info("Training %s seed %d for %d steps", name, seed, train_cfg["total_timesteps"])
    t0 = time.perf_counter()
    model.learn(train_cfg["total_timesteps"], callback=callback, tb_log_name=f"seed_{seed}",
                progress_bar=progress_bar)
    elapsed = time.perf_counter() - t0
    model.save(model_dir / "final_model")
    vec_env.close()

    fps = train_cfg["total_timesteps"] / elapsed
    log.info("Done %s seed %d in %.1f min (%.0f steps/s incl. eval) | best val return %.1f",
             name, seed, elapsed / 60, fps, callback.best_mean)
    return {"seed": seed, "minutes": elapsed / 60, "fps": fps, "best_val_return": callback.best_mean}


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

    if args.workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(train_one, [cfg] * len(seeds), seeds, [False] * len(seeds)))
    else:
        results = [train_one(cfg, s) for s in seeds]

    for r in results:
        log.info("seed %d: %.1f min, %.0f steps/s, best val %.1f",
                 r["seed"], r["minutes"], r["fps"], r["best_val_return"])


if __name__ == "__main__":
    main()
