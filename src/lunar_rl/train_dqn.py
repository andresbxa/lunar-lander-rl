"""Training loop for the from-scratch DQN agent.

For every training seed this produces (same layout as the SB3 baseline):

* ``models/<name>/seed_<s>/best_model.pt``   best checkpoint on *validation* seeds
* ``models/<name>/seed_<s>/final_model.pt``  weights at the end of training
* ``runs/<name>/seed_<s>/``                  TensorBoard logs
* ``results/curves/<name>/seed_<s>.csv``     validation return vs. timesteps

Run it through the generic entry point::

    python -m lunar_rl.train --config configs/dqn.yaml --workers 3
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lunar_rl.agents.dqn import DQNAgent, DQNConfig, linear_epsilon
from lunar_rl.agents.replay_buffer import ReplayBuffer
from lunar_rl.envs import make_env
from lunar_rl.utils import get_logger, save_config, set_global_seed
from lunar_rl.validation import LOGGED_METRICS, PeriodicValidator

log = get_logger("train_dqn")


def train_one(cfg: dict[str, Any], seed: int, progress_bar: bool = True) -> dict[str, Any]:
    name, train_cfg, eval_cfg = cfg["name"], cfg["train"], cfg["eval"]
    dqn_cfg = DQNConfig(**cfg["dqn"])
    total = int(train_cfg["total_timesteps"])
    log_interval = int(train_cfg.get("log_interval", 1_000))
    device = train_cfg.get("device", "cpu")
    torch.set_num_threads(train_cfg.get("torch_threads", 1))
    set_global_seed(seed)

    model_dir = Path("models") / name / f"seed_{seed}"
    curve_path = Path("results/curves") / name / f"seed_{seed}.csv"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "seed": seed}, model_dir / "config.yaml")
    save_config(cfg, curve_path.parent / "config.yaml")

    env = make_env(seed=seed, **cfg["env"])
    obs_dim, n_actions = int(env.observation_space.shape[0]), int(env.action_space.n)
    agent = DQNAgent(obs_dim, n_actions, dqn_cfg, seed=seed, device=device)
    buffer = ReplayBuffer(dqn_cfg.buffer_size, obs_dim, seed=seed, device=device)
    writer = SummaryWriter(str(Path("runs") / name / f"seed_{seed}"))
    validator = PeriodicValidator(eval_cfg["freq"], eval_cfg["n_episodes"], eval_cfg["seed_start"], cfg["env"],
                                  curve_path, save_best=lambda: agent.save(model_dir / "best_model.pt"),
                                  tag=f"[seed {seed}]")

    def run_validation(step: int) -> None:
        stats = validator.validate(agent, step)
        for k in (*LOGGED_METRICS, "best_mean_return"):
            writer.add_scalar(f"eval/{k}", stats[k], step)

    log.info("Training %s seed %d for %d steps", name, seed, total)
    t0 = time.perf_counter()
    validator.start()
    run_validation(0)  # step-0 point: the untrained policy

    obs, _ = env.reset()  # make_env already seeded the env; this continues its seeded stream
    ep_return, ep_len, episodes = 0.0, 0, 0
    recent_updates: list[dict[str, float]] = []
    bar = tqdm(total=total, disable=not progress_bar, mininterval=1.0, unit="step")

    for step in range(1, total + 1):
        agent.epsilon = linear_epsilon(step - 1, dqn_cfg.eps_start, dqn_cfg.eps_end, dqn_cfg.eps_decay_steps)
        action = agent.act(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Store `terminated` only: a time-limit truncation must still bootstrap.
        buffer.add(obs, action, float(reward), next_obs, terminated)
        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        if terminated or truncated:
            episodes += 1
            writer.add_scalar("train/episode_return", ep_return, step)
            writer.add_scalar("train/episode_length", ep_len, step)
            obs, _ = env.reset()  # no seed: continues the env's seeded RNG stream
            ep_return, ep_len = 0.0, 0

        if step > dqn_cfg.learning_starts and step % dqn_cfg.train_freq == 0:
            for _ in range(dqn_cfg.gradient_steps):
                recent_updates.append(agent.update(buffer.sample(dqn_cfg.batch_size)))

        if step % dqn_cfg.target_update_interval == 0:
            agent.sync_target()

        if step % log_interval == 0:
            writer.add_scalar("train/epsilon", agent.epsilon, step)
            writer.add_scalar("train/episodes", episodes, step)
            writer.add_scalar("time/steps_per_s", step / (time.perf_counter() - t0), step)
            if recent_updates:
                for k in recent_updates[0]:
                    writer.add_scalar(f"train/{k}", float(np.mean([u[k] for u in recent_updates])), step)
                recent_updates.clear()
            bar.update(log_interval)

        if validator.due(step):
            run_validation(step)

    if validator.last_step != total:
        run_validation(total)
    bar.close()
    agent.save(model_dir / "final_model.pt")
    writer.close()
    env.close()

    elapsed = time.perf_counter() - t0
    fps = total / elapsed
    log.info("Done %s seed %d in %.1f min (%.0f steps/s incl. eval) | best val return %.1f",
             name, seed, elapsed / 60, fps, validator.best_mean)
    return {"seed": seed, "minutes": elapsed / 60, "fps": fps, "best_val_return": validator.best_mean}

