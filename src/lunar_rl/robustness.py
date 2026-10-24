"""Robustness to wind: evaluate agents trained *without* wind under increasing wind.

Every model is evaluated on the same 100 test episodes (seeds 10000+) under
each condition. The wind phase is drawn from the environment's RNG, so with the
same episode seed every agent faces exactly the same gusts: the comparison
between algorithms is paired.

Usage::

    python -m lunar_rl.robustness --workers 9          # evaluate + plot
    python -m lunar_rl.robustness --plot-only          # re-plot from results/robustness.csv
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lunar_rl.agents import Agent, load_agent  # noqa: E402
from lunar_rl.envs import env_condition_name  # noqa: E402
from lunar_rl.evaluate import SOLVED_THRESHOLD, evaluate, summarize  # noqa: E402
from lunar_rl.plot_curves import GRID, SURFACE, TEXT, TEXT_MUTED, algo_colors  # noqa: E402
from lunar_rl.utils import TEST_SEED_START, get_logger  # noqa: E402

log = get_logger("robustness")

# (wind_power, turbulence_power); wind_power=0 means wind disabled. Gymnasium accepts 0-20 / 0-2.
WIND_LEVELS = [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0), (15.0, 1.5), (20.0, 2.0)]

MODEL_PATHS = {
    "dqn": "models/dqn/seed_{seed}/best_model.pt",
    "ppo": "models/ppo/seed_{seed}/best_model.pt",
    "sb3_ppo": "models/sb3_ppo/seed_{seed}/best_model.zip",
}


def wind_kwargs(wind_power: float, turbulence_power: float) -> dict[str, Any]:
    return {"enable_wind": wind_power > 0, "wind_power": wind_power, "turbulence_power": turbulence_power}


def evaluate_conditions(agent: Agent, levels: list[tuple[float, float]], n_episodes: int,
                        seed_start: int = TEST_SEED_START,
                        episodes_dir: Path | None = None, stem: str = "") -> pd.DataFrame:
    """Evaluate one agent under every wind level; one summary row per level."""
    rows = []
    for wind, turb in levels:
        env_kwargs = wind_kwargs(wind, turb)
        df = evaluate(agent, n_episodes, seed_start, True, env_kwargs)
        condition = env_condition_name(env_kwargs)
        if episodes_dir is not None:
            episodes_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(episodes_dir / f"{stem}_{condition}.csv", index=False, float_format="%.3f")
        rows.append({"condition": condition, "wind_power": wind, "turbulence_power": turb, **summarize(df)})
    return pd.DataFrame(rows)


def _job(agent_name: str, seed: int, n_episodes: int, episodes_dir: str) -> pd.DataFrame:
    agent = load_agent(agent_name, MODEL_PATHS[agent_name].format(seed=seed))
    df = evaluate_conditions(agent, WIND_LEVELS, n_episodes, episodes_dir=Path(episodes_dir),
                             stem=f"{agent_name}_seed{seed}")
    df.insert(0, "train_seed", seed)
    df.insert(0, "agent", agent_name)
    log.info("%s seed %d: %s", agent_name, seed,
             " | ".join(f"w{r.wind_power:g}: {r.mean_return:.0f}" for r in df.itertuples()))
    return df


def plot_robustness(df: pd.DataFrame, out: Path) -> None:
    """Two panels (mean return, success rate) vs wind level; mean +/- std across training seeds."""
    agents = list(dict.fromkeys(df["agent"]))
    per_seed = df.groupby(["agent", "wind_power"])
    stats = per_seed[["mean_return", "success_rate"]].agg(["mean", "std"]).reset_index()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    levels = sorted(df["wind_power"].unique())
    labels = ["none" if w == 0 else f"{w:g} / {t:g}" for w, t in
              sorted(set(zip(df["wind_power"], df["turbulence_power"])))]

    for ax, metric, ylabel, scale in [
        (axes[0], "mean_return", "mean test return", 1.0),
        (axes[1], "success_rate", "success rate (return ≥ 200)", 100.0),
    ]:
        ax.set_facecolor(SURFACE)
        for agent, color in zip(agents, algo_colors(agents)):
            s = stats[stats["agent"] == agent].sort_values("wind_power")
            mean = s[(metric, "mean")].to_numpy() * scale
            std = np.nan_to_num(s[(metric, "std")].to_numpy()) * scale
            ax.fill_between(s["wind_power"], mean - std, mean + std, color=color, alpha=0.18, lw=0)
            ax.plot(s["wind_power"], mean, color=color, lw=2, marker="o", ms=5, label=agent)
        if metric == "mean_return":
            ax.axhline(SOLVED_THRESHOLD, color=TEXT_MUTED, lw=1, ls="--", zorder=1)
            ax.annotate(f"solved ({SOLVED_THRESHOLD:.0f})", (0.99, SOLVED_THRESHOLD),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 4), textcoords="offset points",
                        ha="right", va="bottom", color=TEXT_MUTED, fontsize=8.5)
        else:
            ax.set_ylim(0, 105)
        ax.set_xticks(levels, labels)
        ax.set_xlabel("wind power / turbulence power")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    axes[0].legend(frameon=False, loc="lower left", labelcolor=TEXT)
    fig.suptitle("Robustness: agents trained without wind, tested with wind", x=0.01, ha="left",
                 color=TEXT, fontsize=11)
    fig.text(0.01, 0.01, f"100 test episodes per point and seed. Line: mean across {df['train_seed'].nunique()} "
             "training seeds. Band: ±1 std across seeds.", color=TEXT_MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    log.info("Saved %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", nargs="+", default=list(MODEL_PATHS), choices=list(MODEL_PATHS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("results/robustness.csv"))
    p.add_argument("--episodes-dir", type=Path, default=Path("results/eval/robustness"))
    p.add_argument("--figure", type=Path, default=Path("results/figures/robustness_wind.png"))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    if not args.plot_only:
        jobs = [(a, s) for a in args.agents for s in args.seeds]
        n = len(jobs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            frames = list(pool.map(_job, [a for a, _ in jobs], [s for _, s in jobs],
                                   [args.episodes] * n, [str(args.episodes_dir)] * n))
        df = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, float_format="%.3f")
        log.info("Saved %s", args.out)

    df = pd.read_csv(args.out)
    plot_robustness(df, args.figure)
    summary = df.groupby(["agent", "wind_power"])["mean_return"].agg(["mean", "std"]).round(1)
    log.info("Mean test return across seeds:\n%s", summary.unstack("wind_power").to_string())


if __name__ == "__main__":
    main()
