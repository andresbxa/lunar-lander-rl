"""Framework-agnostic periodic validation used by every training loop.

SB3 (through a thin callback) and the from-scratch agents call the same
``PeriodicValidator``, so all learning curves are produced by identical code:
same validation seeds, same metrics, same CSV format, same "best checkpoint"
rule. That is what makes the curves directly comparable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from lunar_rl.agents.base import Agent
from lunar_rl.evaluate import evaluate, summarize
from lunar_rl.utils import get_logger

log = get_logger("validation")

LOGGED_METRICS = ("mean_return", "std_return", "success_rate", "mean_length", "truncated_rate")


class PeriodicValidator:
    """Evaluate on fixed validation seeds every ``eval_freq`` env steps and keep the best model.

    Args:
        save_best: Called (with no arguments) whenever validation return improves.
        curve_path: CSV rewritten after every evaluation, one row per evaluation.
        tag: Prefix for log lines (e.g. ``"[seed 0]"``), useful when seeds run in parallel.
    """

    def __init__(self, eval_freq: int, n_episodes: int, seed_start: int, env_kwargs: dict[str, Any],
                 curve_path: Path, save_best: Callable[[], None], tag: str = ""):
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.seed_start = seed_start
        self.env_kwargs = env_kwargs
        self.curve_path = Path(curve_path)
        self.save_best = save_best
        self.tag = tag
        self.best_mean = -float("inf")
        self.next_eval = 0
        self.rows: list[dict[str, float]] = []
        self.t0 = time.perf_counter()

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def due(self, step: int) -> bool:
        return step >= self.next_eval

    def validate(self, agent: Agent, step: int) -> dict[str, float]:
        """Run validation now; returns the stats plus ``best_mean_return``."""
        stats = summarize(evaluate(agent, self.n_episodes, self.seed_start, True, self.env_kwargs))
        self.rows.append({"timesteps": step, **stats, "wall_time_s": time.perf_counter() - self.t0})
        # Rewrite the whole CSV each time so a crashed run still leaves a usable curve.
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.curve_path, index=False, float_format="%.3f")

        is_best = stats["mean_return"] > self.best_mean
        if is_best:
            self.best_mean = stats["mean_return"]
            self.save_best()
        while self.next_eval <= step:
            self.next_eval += self.eval_freq

        log.info("%s step %8d | val return %7.1f +/- %5.1f | success %3.0f%% | best %7.1f%s",
                 self.tag, step, stats["mean_return"], stats["std_return"],
                 100 * stats["success_rate"], self.best_mean, "  *saved*" if is_best else "")
        return {**stats, "best_mean_return": self.best_mean}

    def maybe_validate(self, agent: Agent, step: int) -> dict[str, float] | None:
        return self.validate(agent, step) if self.due(step) else None

    @property
    def last_step(self) -> int | None:
        return int(self.rows[-1]["timesteps"]) if self.rows else None
