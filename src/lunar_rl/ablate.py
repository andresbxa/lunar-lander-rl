"""Run the ablation suite defined in ``configs/ablations.yaml``.

Each variant is ``base config + overrides`` and is trained on the base
config's seeds (all variant seeds share one process pool), selected on
validation, then evaluated on the 100 held-out test episodes.

Usage::

    python -m lunar_rl.ablate --list
    python -m lunar_rl.ablate --variants ppo_big_net ppo_gamma_0.99 --workers 6
    python -m lunar_rl.ablate --variants dqn ppo --evaluate-only   # baselines, both checkpoints
    python -m lunar_rl.ablate --report            # markdown table vs. the base algorithms

Every run is evaluated twice on the test episodes:

* **best** checkpoint (selected on validation): the peak performance that
  validation-based selection can recover, which is what the main results report;
* **final** weights: what training actually ends with. This is the stability
  measure: a run that collapses late can still have a good *best* checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lunar_rl.agents import load_agent
from lunar_rl.evaluate import aggregate_over_seeds, evaluate, summarize
from lunar_rl.train import train_seed
from lunar_rl.utils import TEST_SEED_START, get_logger, load_config

log = get_logger("ablate")

MODEL_EXT = {"dqn": ".pt", "ppo": ".pt", "sb3_ppo": ".zip"}
CHECKPOINTS = ("best", "final")


def load_suite(path: Path) -> dict[str, dict[str, Any]]:
    """Variants and baselines in one mapping; baselines are a base config with no override."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    baselines = {name: {"base": base, "change": "baseline", "set": {}, "baseline": True}
                 for name, base in raw.get("baselines", {}).items()}
    return {**baselines, **raw["variants"]}


def variant_config(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    overrides = [f"{k}={yaml.safe_dump(v, default_flow_style=True).strip()}" for k, v in spec["set"].items()]
    cfg = load_config(spec["base"], overrides)
    cfg["name"] = name
    return cfg


def evaluate_variant(cfg: dict[str, Any], n_episodes: int, episodes_dir: str) -> list[dict[str, Any]]:
    """Evaluate the best and final checkpoints of every seed on the test episodes."""
    name, algo = cfg["name"], cfg["algo"]
    episodes_path = Path(episodes_dir)
    episodes_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt in CHECKPOINTS:
        per_seed = []
        for seed in cfg["seeds"]:
            agent = load_agent(algo, Path("models") / name / f"seed_{seed}" / f"{ckpt}_model{MODEL_EXT[algo]}")
            df = evaluate(agent, n_episodes, TEST_SEED_START, True, cfg["env"])
            df.to_csv(episodes_path / f"{name}_{ckpt}_seed{seed}_no_wind.csv", index=False, float_format="%.3f")
            stats = summarize(df)
            per_seed.append(stats)
            rows.append({"variant": name, "checkpoint": ckpt, "train_seed": str(seed), "n_seeds": 1,
                         "std_over": "episodes", **stats})
        agg = aggregate_over_seeds(per_seed)
        rows.append({"variant": name, "checkpoint": ckpt, "train_seed": "all", **agg})
        log.info("%s [%s]: %.1f +/- %.1f (std across seeds) | per seed %s | success %.0f%%", name, ckpt,
                 agg["mean_return"], agg["std_return"], [round(s["mean_return"], 1) for s in per_seed],
                 100 * agg["success_rate"])
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    new = pd.DataFrame(rows).astype({"train_seed": str})
    if path.exists():
        old = pd.read_csv(path, dtype={"train_seed": str})
        new = pd.concat([old[~old["variant"].isin(new["variant"])], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["variant", "checkpoint", "train_seed"]).to_csv(path, index=False, float_format="%.3f")


def verdict(delta: float, std_a: float, std_b: float) -> str:
    """With 3 seeds, call a difference *likely real* only if it exceeds the
    across-seed std of both runs; otherwise report it as *inconclusive*."""
    if abs(delta) > max(std_a, std_b):
        return ("better" if delta > 0 else "worse") + " (likely real)"
    return "inconclusive"


def report(suite: dict[str, dict[str, Any]], ablations_csv: Path) -> str:
    """Markdown table: every run's best and final checkpoints, variants compared to their baseline."""
    abl = pd.read_csv(ablations_csv, dtype={"train_seed": str})
    agg = abl[abl["train_seed"] == "all"].set_index(["variant", "checkpoint"])
    lines = ["| Run | Change | Best checkpoint | Δ best | Final weights | Δ final | Verdict (best / final) |",
             "|---|---|---:|---:|---:|---:|---|"]
    for name, spec in suite.items():
        if (name, "best") not in agg.index:
            continue
        cells, verdicts = [], []
        base_name = None if spec.get("baseline") else load_config(spec["base"])["name"]
        for ckpt in CHECKPOINTS:
            r = agg.loc[(name, ckpt)]
            cells.append(f"{r['mean_return']:.1f} ± {r['std_return']:.1f}")
            if base_name is None or (base_name, ckpt) not in agg.index:
                cells.append("–")
                continue
            b = agg.loc[(base_name, ckpt)]
            delta = r["mean_return"] - b["mean_return"]
            cells.append(f"{delta:+.1f}")
            verdicts.append(verdict(delta, r["std_return"], b["std_return"]))
        label = f"**`{name}`**" if spec.get("baseline") else f"`{name}`"
        lines.append(f"| {label} | {spec['change']} | {' | '.join(cells)} | {' / '.join(verdicts) or '–'} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", type=Path, default=Path("configs/ablations.yaml"))
    p.add_argument("--variants", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=1, help="Processes shared by all (variant, seed) jobs.")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Extra overrides applied to every variant (e.g. for a smoke test).")
    p.add_argument("--name-suffix", default="", help="Appended to run names (keeps smoke tests separate).")
    p.add_argument("--out", type=Path, default=Path("results/ablations.csv"))
    p.add_argument("--episodes-dir", type=Path, default=Path("results/eval/ablations"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    suite = load_suite(args.suite)
    if args.list:
        for name, spec in suite.items():
            print(f"{name:16s} {spec['change']}")
        return
    if args.report:
        sys.stdout.reconfigure(encoding="utf-8")  # the table uses ± and Δ (Windows consoles default to cp1252)
        print(report(suite, args.out))
        return

    unknown = set(args.variants) - set(suite)
    if unknown or not args.variants:
        raise SystemExit(f"Choose --variants from {list(suite)} (unknown: {sorted(unknown)})")
    if not args.evaluate_only and any(suite[v].get("baseline") for v in args.variants):
        raise SystemExit("Baselines are trained with `python -m lunar_rl.train`; use --evaluate-only here.")

    from lunar_rl.utils.config import apply_overrides

    cfgs = {v: apply_overrides(variant_config(v, suite[v]), args.set) for v in args.variants}
    for v, cfg in cfgs.items():
        cfg["name"] = v + args.name_suffix
    if not args.evaluate_only:
        jobs = [(cfgs[v], s) for v in args.variants for s in cfgs[v]["seeds"]]
        log.info("Training %d jobs (%s) with %d workers", len(jobs), ", ".join(args.variants), args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(train_seed, [c for c, _ in jobs], [s for _, s in jobs], [False] * len(jobs)))
        for (cfg, _), r in zip(jobs, results):
            log.info("%s seed %d: %.1f min, best val %.1f", cfg["name"], r["seed"], r["minutes"], r["best_val_return"])

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        per_variant = list(pool.map(evaluate_variant, [cfgs[v] for v in args.variants],
                                    [args.episodes] * len(args.variants),
                                    [str(args.episodes_dir)] * len(args.variants)))
    write_rows(args.out, [row for rows in per_variant for row in rows])
    log.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
