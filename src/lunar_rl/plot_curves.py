"""Plot learning curves (validation return vs. env steps), mean +/- std across training seeds.

Reads ``results/curves/<algo>/seed_*.csv`` written during training.

Usage::

    python -m lunar_rl.plot_curves --algo sb3_ppo
    python -m lunar_rl.plot_curves --algo sb3_ppo dqn --out results/figures/ppo_vs_dqn.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from lunar_rl.evaluate import SOLVED_THRESHOLD  # noqa: E402
from lunar_rl.utils import get_logger  # noqa: E402

log = get_logger("plot_curves")

# Categorical slots in fixed order (one per algorithm), plus neutral inks.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#6b6a66", "#e6e5e1"


def load_curves(curves_dir: Path, algo: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timesteps grid, returns of shape [n_seeds, n_points]).

    Seeds are interpolated onto the first seed's grid, truncated to the
    shortest run, so runs with slightly different eval points still align.
    """
    files = sorted((curves_dir / algo).glob("seed_*.csv"))
    if not files:
        raise FileNotFoundError(f"No curves found in {curves_dir / algo}")
    runs = [pd.read_csv(f) for f in files]
    t_max = min(r["timesteps"].max() for r in runs)
    grid = runs[0]["timesteps"].to_numpy()
    grid = grid[grid <= t_max]
    returns = np.stack([np.interp(grid, r["timesteps"], r["mean_return"]) for r in runs])
    return grid, returns


def _steps_formatter(x: float, _pos) -> str:
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def plot(algos: list[str], curves_dir: Path, out: Path, random_csv: Path | None) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED,
                         "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED})
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    seed_counts = []
    for i, algo in enumerate(algos):
        color = SERIES_COLORS[i]
        grid, returns = load_curves(curves_dir, algo)
        n = len(returns)
        seed_counts.append(n)
        mean = returns.mean(axis=0)
        std = returns.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        for seed_curve in returns:
            ax.plot(grid, seed_curve, color=color, lw=0.8, alpha=0.25, zorder=2)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(grid, mean, color=color, lw=2, zorder=3, label=f"{algo} (n={n} seeds)")
        ax.annotate(f"{mean[-1]:.0f}", (grid[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TEXT, fontsize=9)

    ax.axhline(SOLVED_THRESHOLD, color=TEXT_MUTED, lw=1, ls="--", zorder=1)
    ax.annotate(f"solved ({SOLVED_THRESHOLD:.0f})", (0, SOLVED_THRESHOLD), xytext=(4, 4),
                textcoords="offset points", color=TEXT_MUTED, fontsize=8.5)
    if random_csv is not None and random_csv.exists():
        rand = pd.read_csv(random_csv)["return"].mean()
        ax.axhline(rand, color=TEXT_MUTED, lw=1, ls=":", zorder=1)
        ax.annotate(f"random agent ({rand:.0f})", (0, rand), xytext=(4, 4), textcoords="offset points",
                    color=TEXT_MUTED, fontsize=8.5)

    title = "LunarLander-v3: validation return during training"
    if len(algos) == 1:
        title += f" ({algos[0]}, {seed_counts[0]} seeds)"
    ax.set_title(title, loc="left", color=TEXT, fontsize=11)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean validation return")
    ax.xaxis.set_major_formatter(FuncFormatter(_steps_formatter))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.margins(x=0.02)
    if len(algos) > 1:
        ax.legend(frameon=False, loc="lower right", labelcolor=TEXT)
    fig.text(0.01, 0.01, "Line: mean across training seeds. Band: ±1 std across seeds. Thin lines: individual seeds.",
             color=TEXT_MUTED, fontsize=7.5)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    log.info("Saved %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", nargs="+", required=True, help="Experiment name(s) under results/curves/.")
    p.add_argument("--curves-dir", type=Path, default=Path("results/curves"))
    p.add_argument("--random-csv", type=Path, default=Path("results/eval/random_no_wind.csv"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if len(args.algo) > len(SERIES_COLORS):
        raise SystemExit(f"At most {len(SERIES_COLORS)} algorithms per figure; split into several plots.")
    out = args.out or Path("results/figures") / f"learning_curve_{'_vs_'.join(args.algo)}.png"
    plot(args.algo, args.curves_dir, out, args.random_csv)


if __name__ == "__main__":
    main()
