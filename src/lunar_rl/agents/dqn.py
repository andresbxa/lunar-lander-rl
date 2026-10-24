"""DQN (Mnih et al., 2015) implemented from scratch in PyTorch.

Ingredients:

* **Q-network**: MLP mapping a state to one Q-value per action, so the greedy
  action is a single forward pass + argmax.
* **Target network**: a frozen copy of the Q-network used to compute TD
  targets, synced every ``target_update_interval`` env steps. Without it the
  regression target moves with every gradient step and training tends to
  oscillate or diverge.
* **TD target**: ``r + gamma * (1 - terminated) * max_a' Q_target(s', a')``.
  Only true terminal states stop bootstrapping; time-limit truncations do not.
* **Huber loss** and gradient-norm clipping to keep early, large TD errors
  from producing huge updates.
* **Double DQN** (optional, van Hasselt et al., 2016): the online network picks
  ``a'`` and the target network evaluates it, reducing the max-operator's
  overestimation bias.

The replay buffer lives in :mod:`lunar_rl.agents.replay_buffer` and the
training loop in :mod:`lunar_rl.train_dqn`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lunar_rl.agents.replay_buffer import Batch


@dataclass
class DQNConfig:
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 6.3e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    learning_starts: int = 10_000        # env steps collected before the first update
    train_freq: int = 4                  # env steps between update phases
    gradient_steps: int = 4              # updates per phase (4/4 -> one update per env step)
    target_update_interval: int = 250    # env steps between target-network syncs
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 12_000        # linear decay length, in env steps
    max_grad_norm: float = 10.0
    double_dqn: bool = False
    use_target_network: bool = True      # False: TD targets from the online net (ablation)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linearly anneal epsilon from ``start`` to ``end`` over ``decay_steps``, then hold."""
    if decay_steps <= 0:
        return end
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


@torch.no_grad()
def td_targets(rewards: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor,
               q_target: nn.Module, gamma: float, q_online: nn.Module | None = None) -> torch.Tensor:
    """Bootstrapped one-step targets. Pass ``q_online`` to get Double DQN targets."""
    next_q = q_target(next_obs)
    if q_online is None:
        next_v = next_q.max(dim=1).values
    else:
        best_actions = q_online(next_obs).argmax(dim=1, keepdim=True)
        next_v = next_q.gather(1, best_actions).squeeze(1)
    return rewards + gamma * (1.0 - terminated) * next_v


class DQNAgent:
    def __init__(self, obs_dim: int = 8, n_actions: int = 4, cfg: DQNConfig | None = None,
                 seed: int | None = 0, device: str = "cpu"):
        self.cfg = cfg or DQNConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        self.q = QNetwork(obs_dim, n_actions, self.cfg.hidden_sizes).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.cfg.learning_rate)
        self.rng = np.random.default_rng(seed)  # exploration RNG, independent of global state
        self.epsilon = self.cfg.eps_start

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Greedy action if ``deterministic``, epsilon-greedy otherwise."""
        if not deterministic and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step on a replay batch; returns diagnostics."""
        q_sa = self.q(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        target_net = self.q_target if self.cfg.use_target_network else self.q
        target = td_targets(batch.rewards, batch.next_obs, batch.terminated, target_net,
                            self.cfg.gamma, self.q if self.cfg.double_dqn else None)
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        return {"loss": loss.item(), "q_mean": q_sa.mean().item(), "grad_norm": float(grad_norm)}

    def sync_target(self) -> None:
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "config": asdict(self.cfg),
                    "obs_dim": int(self.obs_dim), "n_actions": int(self.n_actions)}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], DQNConfig(**ckpt["config"]), device=device)
        agent.q.load_state_dict(ckpt["q"])
        agent.sync_target()
        return agent
