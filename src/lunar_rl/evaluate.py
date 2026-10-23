"""Evaluate any `Agent` on a fixed, held-out set of episode seeds.

Every episode ``i`` is reset with seed ``seed_start + i``, so two agents
evaluated with the same arguments face exactly the same 100 terrains and
initial pushes: differences in score come from the policy, not from luck.

Usage::

    python -m lunar_rl.evaluate --agent random
    python -m lunar_rl.evaluate --agent sb3_ppo --model models/sb3_ppo/seed_*/best_model.zip
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lunar_rl.agents import AGENT_NAMES, Agent, load_agent
from lunar_rl.envs import env_condition_name, make_env
from lunar_rl.utils import TEST_SEED_START, get_logger

SOLVED_THRESHOLD = 200.0
EPISODE_COLUMNS = ["episode", "env_seed", "return", "length", "terminated", "truncated", "success"]
SUMMARY_COLUMNS = [
    "agent", "train_seed", "condition", "n_episodes", "n_seeds",
    "mean_return", "std_return", "std_over", "success_rate", "mean_length", "truncated_rate",
]

log = get_logger("evaluate")


def run_episode(agent: Agent, env, seed: int, deterministic: bool = True) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    total, length = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=deterministic))
        total += float(reward)
        length += 1
    return {"env_seed": seed, "return": total, "length": length,
            "terminated": terminated, "truncated": truncated}


def evaluate(
    agent: Agent,
    n_episodes: int = 100,
    seed_start: int = TEST_SEED_START,
    deterministic: bool = True,
    env_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run ``n_episodes`` and return one row per episode (see ``EPISODE_COLUMNS``).

    ``truncated`` episodes hit the 1000-step limit without landing or crashing:
    a high truncation rate is the signature of a "hovering" agent.
    """
    env = make_env(**(env_kwargs or {}))
    rows = []
    for i in range(n_episodes):
        row = run_episode(agent, env, seed_start + i, deterministic)
        rows.append({"episode": i, **row, "success": row["return"] >= SOLVED_THRESHOLD})
    env.close()
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Episode-level statistics (sample std, ddof=1)."""
    return {
        "n_episodes": len(df),
        "mean_return": float(df["return"].mean()),
        "std_return": float(df["return"].std(ddof=1)) if len(df) > 1 else 0.0,
        "success_rate": float(df["success"].mean()),
        "mean_length": float(df["length"].mean()),
        "truncated_rate": float(df["truncated"].mean()),
    }


def aggregate_over_seeds(per_seed: list[dict[str, float]]) -> dict[str, float]:
    """Combine per-training-seed summaries.

    The headline number is the mean of per-seed means, and its spread is the
    std *across training seeds*: that measures how reliable the algorithm is,
    which is what we care about, not how noisy one agent's episodes are.
    """
    means = np.array([s["mean_return"] for s in per_seed])
    return {
        "n_episodes": int(sum(s["n_episodes"] for s in per_seed)),
        "n_seeds": len(per_seed),
        "mean_return": float(means.mean()),
        "std_return": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
        "std_over": "train_seeds",
        "success_rate": float(np.mean([s["success_rate"] for s in per_seed])),
        "mean_length": float(np.mean([s["mean_length"] for s in per_seed])),
        "truncated_rate": float(np.mean([s["truncated_rate"] for s in per_seed])),
    }


def upsert_summary(path: Path, row: dict[str, Any]) -> None:
    """Insert or replace the row keyed by (agent, train_seed, condition)."""
    key = ["agent", "train_seed", "condition"]
    new = pd.DataFrame([row], columns=SUMMARY_COLUMNS).astype({"train_seed": str})
    if path.exists():
        old = pd.read_csv(path, dtype={"train_seed": str})
        mask = (old[key] == new[key].iloc[0]).all(axis=1)
        new = pd.concat([old[~mask], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(key).to_csv(path, index=False, float_format="%.3f")


def _train_seed_from_path(path: str) -> str:
    match = re.search(r"seed_(\d+)", Path(path).as_posix())
    return match.group(1) if match else "na"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, choices=AGENT_NAMES)
    p.add_argument("--model", nargs="*", default=[], help="Model path(s); glob patterns allowed.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=TEST_SEED_START, help="First evaluation episode seed.")
    p.add_argument("--agent-seed", type=int, default=0, help="RNG seed for the random agent.")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of acting greedily.")
    p.add_argument("--wind", action="store_true")
    p.add_argument("--wind-power", type=float, default=15.0)
    p.add_argument("--turbulence-power", type=float, default=1.5)
    p.add_argument("--out-dir", type=Path, default=Path("results/eval"))
    p.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    args = p.parse_args()

    env_kwargs = {"enable_wind": args.wind, "wind_power": args.wind_power,
                  "turbulence_power": args.turbulence_power}
    condition = env_condition_name(env_kwargs)

    model_paths = sorted({m for pattern in args.model for m in (glob.glob(pattern) or [pattern])})
    targets = model_paths or [None]

    per_seed = []
    for model_path in targets:
        train_seed = _train_seed_from_path(model_path) if model_path else "na"
        agent = load_agent(args.agent, model_path, seed=args.agent_seed)
        df = evaluate(agent, args.episodes, args.seed_start, not args.stochastic, env_kwargs)

        stem = args.agent if train_seed == "na" else f"{args.agent}_seed{train_seed}"
        out = args.out_dir / f"{stem}_{condition}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, float_format="%.3f")

        stats = summarize(df)
        per_seed.append(stats)
        upsert_summary(args.summary, {"agent": args.agent, "train_seed": train_seed, "condition": condition,
                                      "n_seeds": 1, "std_over": "episodes", **stats})
        log.info("%s [train seed %s, %s]: %.1f +/- %.1f over %d episodes | success %.0f%% | truncated %.0f%% -> %s",
                 args.agent, train_seed, condition, stats["mean_return"], stats["std_return"],
                 stats["n_episodes"], 100 * stats["success_rate"], 100 * stats["truncated_rate"], out)

    if len(per_seed) > 1:
        agg = aggregate_over_seeds(per_seed)
        upsert_summary(args.summary, {"agent": args.agent, "train_seed": "all", "condition": condition, **agg})
        log.info("%s [%d train seeds, %s]: %.1f +/- %.1f (std across seeds) | success %.0f%%",
                 args.agent, agg["n_seeds"], condition, agg["mean_return"], agg["std_return"],
                 100 * agg["success_rate"])


if __name__ == "__main__":
    main()
