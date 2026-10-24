"""Training loop for the from-scratch PPO agent.

For every training seed this produces (same layout as the other algorithms):

* ``models/<name>/seed_<s>/best_model.pt``   best checkpoint on *validation* seeds
* ``models/<name>/seed_<s>/final_model.pt``  weights at the end of training
* ``runs/<name>/seed_<s>/``                  TensorBoard logs
* ``results/curves/<name>/seed_<s>.csv``     validation return vs. timesteps

Run it through the generic entry point::

    python -m lunar_rl.train --config configs/ppo.yaml --workers 3
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lunar_rl.agents.ppo import PPOAgent, PPOConfig, Rollout, compute_gae
from lunar_rl.envs import make_env
from lunar_rl.utils import get_logger, save_config, set_global_seed
from lunar_rl.validation import LOGGED_METRICS, PeriodicValidator

log = get_logger("train_ppo")


def make_vec_env(n_envs: int, env_kwargs: dict[str, Any]) -> SyncVectorEnv:
    # SAME_STEP autoreset: when an episode ends, step() returns the first observation of
    # the next episode and puts the true last observation in info["final_obs"], which we
    # need to bootstrap time-limit truncations.
    return SyncVectorEnv([lambda: make_env(**env_kwargs) for _ in range(n_envs)],
                         autoreset_mode=AutoresetMode.SAME_STEP)


def bootstrap_truncations(rewards: np.ndarray, terminated: np.ndarray, truncated: np.ndarray,
                          final_obs: np.ndarray | None, value_fn: Callable[[np.ndarray], np.ndarray],
                          gamma: float) -> np.ndarray:
    """Add ``gamma * V(final_obs)`` to the reward of time-limit truncations.

    A truncation is not a terminal state, but GAE stops bootstrapping at every
    ``done``; folding the value of the true last observation into the reward
    restores the missing tail of the return. Terminated episodes are untouched.
    """
    rewards = rewards.astype(np.float32, copy=True)
    trunc_only = truncated & ~terminated
    if trunc_only.any():
        rewards[trunc_only] += gamma * value_fn(np.stack(final_obs[trunc_only]))
    return rewards


def train_one(cfg: dict[str, Any], seed: int, progress_bar: bool = True) -> dict[str, Any]:
    name, train_cfg, eval_cfg = cfg["name"], cfg["train"], cfg["eval"]
    ppo_cfg = PPOConfig(**cfg["ppo"])
    n_envs, n_steps = int(train_cfg["n_envs"]), ppo_cfg.n_steps
    batch = n_envs * n_steps
    n_updates = math.ceil(int(train_cfg["total_timesteps"]) / batch)
    total = n_updates * batch
    device = train_cfg.get("device", "cpu")
    torch.set_num_threads(train_cfg.get("torch_threads", 1))
    set_global_seed(seed)  # also seeds torch, which drives action sampling

    model_dir = Path("models") / name / f"seed_{seed}"
    curve_path = Path("results/curves") / name / f"seed_{seed}.csv"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "seed": seed}, model_dir / "config.yaml")
    save_config(cfg, curve_path.parent / "config.yaml")

    envs = make_vec_env(n_envs, cfg["env"])
    obs_dim, n_actions = int(envs.single_observation_space.shape[0]), int(envs.single_action_space.n)
    agent = PPOAgent(obs_dim, n_actions, ppo_cfg, seed=seed, device=device)
    writer = SummaryWriter(str(Path("runs") / name / f"seed_{seed}"))
    validator = PeriodicValidator(eval_cfg["freq"], eval_cfg["n_episodes"], eval_cfg["seed_start"], cfg["env"],
                                  curve_path, save_best=lambda: agent.save(model_dir / "best_model.pt"),
                                  tag=f"[seed {seed}]")

    def run_validation(step: int) -> None:
        stats = validator.validate(agent, step)
        for k in (*LOGGED_METRICS, "best_mean_return"):
            writer.add_scalar(f"eval/{k}", stats[k], step)

    def values_of(o: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return agent.policy.value(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().numpy()

    # Rollout storage, shape (n_steps, n_envs, ...).
    buf_obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
    buf_actions = np.zeros((n_steps, n_envs), dtype=np.int64)
    buf_logp = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_dones = np.zeros((n_steps, n_envs), dtype=np.float32)
    buf_values = np.zeros((n_steps, n_envs), dtype=np.float32)

    log.info("Training %s seed %d for %d steps (%d updates)", name, seed, total, n_updates)
    t0 = time.perf_counter()
    validator.start()
    run_validation(0)

    obs, _ = envs.reset(seed=[seed + i for i in range(n_envs)])  # env i seeded with seed + i, as in SB3
    ep_returns = np.zeros(n_envs)
    ep_lengths = np.zeros(n_envs, dtype=np.int64)
    global_step = 0
    bar = tqdm(total=total, disable=not progress_bar, mininterval=1.0, unit="step")

    for update in range(1, n_updates + 1):
        for t in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = agent.policy.dist(obs_t)
                actions = dist.sample()
                buf_logp[t] = dist.log_prob(actions).cpu().numpy()
                buf_values[t] = agent.policy.value(obs_t).cpu().numpy()
            buf_obs[t] = obs
            buf_actions[t] = actions.cpu().numpy()

            obs, rewards, terminated, truncated, info = envs.step(buf_actions[t])
            ep_returns += rewards
            ep_lengths += 1

            rewards = bootstrap_truncations(rewards, terminated, truncated, info.get("final_obs"),
                                            values_of, ppo_cfg.gamma)

            done = terminated | truncated
            buf_rewards[t] = rewards
            buf_dones[t] = done
            for i in np.flatnonzero(done):
                writer.add_scalar("train/episode_return", ep_returns[i], global_step + n_envs)
                writer.add_scalar("train/episode_length", ep_lengths[i], global_step + n_envs)
                ep_returns[i], ep_lengths[i] = 0.0, 0
            global_step += n_envs

        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, values_of(obs),
                                          ppo_cfg.gamma, ppo_cfg.gae_lambda)

        def flat(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
            return torch.as_tensor(x.reshape(batch, *x.shape[2:]), dtype=dtype, device=device)

        rollout = Rollout(flat(buf_obs), flat(buf_actions, torch.int64), flat(buf_logp),
                          flat(advantages), flat(returns), flat(buf_values))
        stats = agent.update(rollout)
        for k, v in stats.items():
            writer.add_scalar(f"train/{k}", v, global_step)
        writer.add_scalar("time/steps_per_s", global_step / (time.perf_counter() - t0), global_step)
        bar.update(batch)

        # Validate right after a policy update, on the same schedule as the SB3 baseline.
        if validator.due(global_step):
            run_validation(global_step)

    if validator.last_step != global_step:
        run_validation(global_step)
    bar.close()
    agent.save(model_dir / "final_model.pt")
    writer.close()
    envs.close()

    elapsed = time.perf_counter() - t0
    fps = total / elapsed
    log.info("Done %s seed %d in %.1f min (%.0f steps/s incl. eval) | best val return %.1f",
             name, seed, elapsed / 60, fps, validator.best_mean)
    return {"seed": seed, "minutes": elapsed / 60, "fps": fps, "best_val_return": validator.best_mean}
