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

from dataclasses import dataclass, field

import numpy as np
import torch
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
