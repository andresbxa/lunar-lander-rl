"""Adapter that exposes a stable-baselines3 model through the `Agent` interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SB3Agent:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str | Path, algo: str = "ppo") -> "SB3Agent":
        from stable_baselines3 import DQN, PPO

        algos = {"ppo": PPO, "dqn": DQN}
        return cls(algos[algo].load(str(path), device="cpu"))

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)
