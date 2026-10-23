"""Uniform random policy: the lower bound every learned agent must beat."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int = 4, seed: int | None = 0):
        self.n_actions = n_actions
        # Own generator so results do not depend on global RNG state.
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        # A random policy has no "greedy" mode; `deterministic` is ignored.
        return int(self.rng.integers(self.n_actions))
