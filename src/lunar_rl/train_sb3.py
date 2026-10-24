"""Train the Stable-Baselines3 PPO reference baseline.

For every training seed this produces:

* ``models/<name>/seed_<s>/best_model.zip``  best checkpoint on *validation* seeds
* ``models/<name>/seed_<s>/final_model.zip`` weights at the end of training
* ``runs/<name>/seed_<s>_1/``                TensorBoard logs
* ``results/curves/<name>/seed_<s>.csv``     validation return vs. timesteps

Usage (``lunar_rl.train_sb3`` is an alias of the generic ``lunar_rl.train``)::

    python -m lunar_rl.train --config configs/sb3_ppo.yaml --workers 3     # all seeds in parallel
    python -m lunar_rl.train --config configs/sb3_ppo.yaml --seed 0 --set train.total_timesteps=50000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from lunar_rl.agents.sb3_agent import SB3Agent
from lunar_rl.envs import make_env
from lunar_rl.utils import get_logger, save_config, set_global_seed
from lunar_rl.validation import LOGGED_METRICS, PeriodicValidator

log = get_logger("train_sb3")


class PeriodicEvalCallback(BaseCallback):
    """Thin SB3 adapter around :class:`lunar_rl.validation.PeriodicValidator`.

    Unlike SB3's ``EvalCallback`` the validator resets every evaluation episode
    with an explicit seed, so successive evaluations are directly comparable
    (same terrains every time) and reuse the same code path as final testing.
    """

    def __init__(self, validator: PeriodicValidator):
        super().__init__()
        self.validator = validator

    def _run_eval(self) -> None:
        stats = self.validator.validate(SB3Agent(self.model), self.num_timesteps)
        for k in (*LOGGED_METRICS, "best_mean_return"):
            self.logger.record(f"eval/{k}", stats[k])
        self.logger.dump(self.num_timesteps)

    def _on_training_start(self) -> None:
        self.validator.start()
        self._run_eval()  # step-0 point: the untrained policy

    def _on_rollout_start(self) -> None:
        # PPO only changes its weights between rollouts, so this is the one
        # place where a new evaluation can measure a new policy.
        if self.validator.due(self.num_timesteps):
            self._run_eval()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self.validator.last_step != self.num_timesteps:
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
    validator = PeriodicValidator(eval_cfg["freq"], eval_cfg["n_episodes"], eval_cfg["seed_start"], cfg["env"],
                                  curve_path, save_best=lambda: model.save(model_dir / "best_model"),
                                  tag=f"[seed {seed}]")
    callback = PeriodicEvalCallback(validator)

    log.info("Training %s seed %d for %d steps", name, seed, train_cfg["total_timesteps"])
    t0 = time.perf_counter()
    model.learn(train_cfg["total_timesteps"], callback=callback, tb_log_name=f"seed_{seed}",
                progress_bar=progress_bar)
    elapsed = time.perf_counter() - t0
    model.save(model_dir / "final_model")
    vec_env.close()

    fps = train_cfg["total_timesteps"] / elapsed
    log.info("Done %s seed %d in %.1f min (%.0f steps/s incl. eval) | best val return %.1f",
             name, seed, elapsed / 60, fps, validator.best_mean)
    return {"seed": seed, "minutes": elapsed / 60, "fps": fps, "best_val_return": validator.best_mean}


def main() -> None:
    # Kept for backwards compatibility; `python -m lunar_rl.train` is the generic entry point.
    from lunar_rl.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
