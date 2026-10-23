"""Environment factory for LunarLander-v3.

Seeding in Gymnasium works through ``env.reset(seed=...)``: passing a seed
re-initialises the environment's internal RNG (terrain shape, initial push,
wind phase), and subsequent ``reset()`` calls without a seed continue that
same RNG stream. ``make_env`` therefore performs the first seeded reset for
you, and evaluation code resets every episode with an explicit seed so each
episode is individually reproducible.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

ENV_ID = "LunarLander-v3"


def make_env(
    seed: int | None = None,
    *,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a discrete-action LunarLander-v3 environment.

    Args:
        seed: If given, seeds the environment RNG (via a first ``reset``) and
            the action / observation spaces, so ``env.action_space.sample()``
            is reproducible too.
        enable_wind: Turn on the wind + turbulence model.
        wind_power: Max magnitude of the linear wind (Gymnasium accepts 0-20).
        turbulence_power: Max magnitude of rotational turbulence (0-2).
        render_mode: ``None`` for training, ``"rgb_array"`` for recording GIFs,
            ``"human"`` to watch live.

    Returns:
        The wrapped environment (with Gymnasium's default 1000-step TimeLimit).
    """
    env = gym.make(
        ENV_ID,
        continuous=False,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        render_mode=render_mode,
    )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def env_condition_name(env_kwargs: dict[str, Any] | None) -> str:
    """Short, stable label for an environment configuration (used in result files)."""
    env_kwargs = env_kwargs or {}
    if not env_kwargs.get("enable_wind", False):
        return "no_wind"
    wind = env_kwargs.get("wind_power", 15.0)
    turb = env_kwargs.get("turbulence_power", 1.5)
    return f"wind{wind:g}_turb{turb:g}"
