"""Uniform experience replay for off-policy agents.

Transitions are stored in pre-allocated NumPy arrays used as a ring buffer:
once full, the oldest transition is overwritten. Sampling uniformly from a
large buffer breaks the strong correlation between consecutive steps, which
would otherwise make SGD updates highly biased, and lets every transition be
reused for many updates (sample efficiency of off-policy methods).

Only ``terminated`` is stored, not ``truncated``: a time-limit cut is not a
real terminal state, so the TD target must still bootstrap from ``next_obs``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    obs: torch.Tensor          # (B, obs_dim) float32
    actions: torch.Tensor      # (B,) int64
    rewards: torch.Tensor      # (B,) float32
    next_obs: torch.Tensor     # (B, obs_dim) float32
    terminated: torch.Tensor   # (B,) float32, 1.0 if next_obs is a true terminal state


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x[idx], device=self.device)

        return Batch(t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminated))
