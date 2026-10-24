from __future__ import annotations

from pathlib import Path

from lunar_rl.agents.base import Agent
from lunar_rl.agents.random_agent import RandomAgent

AGENT_NAMES = ("random", "sb3_ppo", "dqn")


def load_agent(name: str, model_path: str | Path | None = None, seed: int = 0) -> Agent:
    """Build an agent by name (used by the evaluate / record CLIs)."""
    if name == "random":
        return RandomAgent(n_actions=4, seed=seed)
    if name == "sb3_ppo":
        if model_path is None:
            raise ValueError("--model is required for sb3_ppo")
        from lunar_rl.agents.sb3_agent import SB3Agent

        return SB3Agent.load(model_path, algo="ppo")
    if name == "dqn":
        if model_path is None:
            raise ValueError("--model is required for dqn")
        from lunar_rl.agents.dqn import DQNAgent

        return DQNAgent.load(model_path)
    raise ValueError(f"Unknown agent {name!r}; choose from {AGENT_NAMES}")


__all__ = ["AGENT_NAMES", "Agent", "RandomAgent", "load_agent"]
