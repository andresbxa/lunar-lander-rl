"""PPO (Schulman et al., 2017) implemented from scratch in PyTorch.

Implementation details deliberately mirror Stable-Baselines3's PPO so the
comparison against the SB3 baseline is apples to apples (see "The 37
Implementation Details of PPO", Huang et al., 2022):

* separate actor and critic MLPs (64-64, tanh), orthogonal init with gain
  sqrt(2) for hidden layers, 0.01 for the policy head (near-uniform initial
  policy) and 1 for the value head;
* GAE(lambda) advantages; time-limit truncations bootstrap from V(final_obs);
* clipped surrogate objective, unclipped MSE value loss, entropy bonus;
* per-minibatch advantage normalization, Adam (eps=1e-5), grad-norm clipping.

The training loop lives in :mod:`lunar_rl.train_ppo`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    learning_rate: float = 3e-4
    n_steps: int = 1024              # steps per env per rollout
    batch_size: int = 64             # minibatch size
    n_epochs: int = 4                # passes over each rollout
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    adam_eps: float = 1e-5


def mlp(in_dim: int, hidden_sizes: list[int], out_dim: int, out_gain: float) -> nn.Sequential:
    """Tanh MLP with orthogonal init (gain sqrt(2) hidden, ``out_gain`` on the output layer)."""
    layers: list[nn.Module] = []
    for h in hidden_sizes:
        layer = nn.Linear(in_dim, h)
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        nn.init.zeros_(layer.bias)
        layers += [layer, nn.Tanh()]
        in_dim = h
    out = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]):
        super().__init__()
        self.actor = mlp(obs_dim, hidden_sizes, n_actions, out_gain=0.01)
        self.critic = mlp(obs_dim, hidden_sizes, 1, out_gain=1.0)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (log_prob, entropy, value) for a batch of (obs, action) pairs."""
        d = self.dist(obs)
        return d.log_prob(actions), d.entropy(), self.value(obs)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_values: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over a rollout of shape (T, n_envs).

    ``dones[t]`` is True if the episode ended at step t (terminated or
    truncated), so step t+1 belongs to a new episode and must not be
    bootstrapped from. Truncations are handled before this call by adding
    ``gamma * V(final_obs)`` to their reward.

    Returns (advantages, returns) with ``returns = advantages + values``,
    which are the value-function regression targets.
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros_like(last_values, dtype=np.float32)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def clipped_policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
                        clip_range: float) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO's clipped surrogate objective (returned as a loss to minimize).

    ``ratio = pi_new(a|s) / pi_old(a|s)``. Taking the max of the clipped and
    unclipped *losses* (= min of the objectives) removes any incentive to move
    the ratio outside ``[1 - clip, 1 + clip]`` in the direction that would
    increase the objective, which keeps each update close to the data-collecting
    policy without needing TRPO's second-order constraint.
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.max(unclipped, clipped).mean()
    with torch.no_grad():
        stats = {
            # Low-variance KL(old || new) estimator (Schulman, "Approximating KL divergence").
            "approx_kl": ((ratio - 1.0) - log_ratio).mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > clip_range).float().mean().item(),
        }
    return loss, stats


class Rollout(NamedTuple):
    """A flattened rollout (n_steps * n_envs samples) ready for minibatch updates."""

    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 = critic predicts returns perfectly, 0 = no better than a constant, < 0 = worse."""
    var = np.var(returns)
    return float("nan") if var == 0 else float(1.0 - np.var(returns - values) / var)


class PPOAgent:
    def __init__(self, obs_dim: int = 8, n_actions: int = 4, cfg: PPOConfig | None = None,
                 seed: int | None = 0, device: str = "cpu"):
        self.cfg = cfg or PPOConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        self.policy = ActorCritic(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate,
                                          eps=self.cfg.adam_eps)
        self.rng = np.random.default_rng(seed)  # minibatch shuffling

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Most likely action if ``deterministic``, a sample from the policy otherwise."""
        with torch.no_grad():
            d = self.policy.dist(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            action = d.probs.argmax(dim=1) if deterministic else d.sample()
        return int(action.item())

    def update(self, rollout: Rollout) -> dict[str, float]:
        """Run ``n_epochs`` of minibatch SGD on one rollout; returns mean diagnostics."""
        cfg = self.cfg
        n = rollout.obs.shape[0]
        history: dict[str, list[float]] = {}
        for _ in range(cfg.n_epochs):
            for idx in np.array_split(self.rng.permutation(n), max(n // cfg.batch_size, 1)):
                idx_t = torch.as_tensor(idx, device=self.device)
                adv = rollout.advantages[idx_t]
                if cfg.normalize_advantage and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                logp, entropy, values = self.policy.evaluate_actions(rollout.obs[idx_t], rollout.actions[idx_t])
                pg_loss, stats = clipped_policy_loss(logp, rollout.logprobs[idx_t], adv, cfg.clip_range)
                value_loss = F.mse_loss(values, rollout.returns[idx_t])
                entropy_loss = -entropy.mean()
                loss = pg_loss + cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in {"policy_loss": pg_loss.item(), "value_loss": value_loss.item(),
                             "entropy": -entropy_loss.item(), **stats}.items():
                    history.setdefault(k, []).append(v)

        out = {k: float(np.mean(v)) for k, v in history.items()}
        out["explained_variance"] = explained_variance(rollout.values.cpu().numpy(), rollout.returns.cpu().numpy())
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "config": asdict(self.cfg),
                    "obs_dim": int(self.obs_dim), "n_actions": int(self.n_actions)}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PPOAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], PPOConfig(**ckpt["config"]), device=device)
        agent.policy.load_state_dict(ckpt["policy"])
        return agent
