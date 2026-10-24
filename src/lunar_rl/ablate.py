"""Run the ablation suite defined in ``configs/ablations.yaml``.

Each variant is ``base config + overrides`` and is trained on the base
config's seeds (all variant seeds share one process pool), selected on
validation, then evaluated on the 100 held-out test episodes.

Usage::

    python -m lunar_rl.ablate --list
    python -m lunar_rl.ablate --variants ppo_big_net ppo_gamma_0.99 --workers 6
    python -m lunar_rl.ablate --variants dqn_double --evaluate-only
    python -m lunar_rl.ablate --report            # markdown table vs. the base algorithms
"""

from __future__ import annotations

import argparse
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

MODEL_FILES = {"dqn": "best_model.pt", "ppo": "best_model.pt", "sb3_ppo": "best_model.zip"}


def load_suite(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["variants"]


def variant_config(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    overrides = [f"{k}={yaml.safe_dump(v, default_flow_style=True).strip()}" for k, v in spec["set"].items()]
    cfg = load_config(spec["base"], overrides)
    cfg["name"] = name
    return cfg


def evaluate_variant(cfg: dict[str, Any], n_episodes: int, episodes_dir: Path) -> list[dict[str, Any]]:
    name, algo = cfg["name"], cfg["algo"]
    rows, per_seed = [], []
    for seed in cfg["seeds"]:
        agent = load_agent(algo, Path("models") / name / f"seed_{seed}" / MODEL_FILES[algo])
        df = evaluate(agent, n_episodes, TEST_SEED_START, True, cfg["env"])
        episodes_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(episodes_dir / f"{name}_seed{seed}_no_wind.csv", index=False, float_format="%.3f")
        stats = summarize(df)
        per_seed.append(stats)
        rows.append({"variant": name, "train_seed": str(seed), "n_seeds": 1, "std_over": "episodes", **stats})
    agg = aggregate_over_seeds(per_seed)
    rows.append({"variant": name, "train_seed": "all", **agg})
    log.info("%s: %.1f +/- %.1f (std across seeds) | per seed %s | success %.0f%%", name,
             agg["mean_return"], agg["std_return"], [round(s["mean_return"], 1) for s in per_seed],
             100 * agg["success_rate"])
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    new = pd.DataFrame(rows).astype({"train_seed": str})
    if path.exists():
        old = pd.read_csv(path, dtype={"train_seed": str})
        new = pd.concat([old[~old["variant"].isin(new["variant"])], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["variant", "train_seed"]).to_csv(path, index=False, float_format="%.3f")


def report(suite: dict[str, dict[str, Any]], ablations_csv: Path, summary_csv: Path) -> str:
    """Markdown table: each variant vs. its base algorithm, with a conservative verdict.

    With 3 seeds, a difference is called *likely real* only if it is larger than
    the across-seed std of both the base and the variant; otherwise it is
    reported as *inconclusive* rather than dressed up as a finding.
    """
    abl = pd.read_csv(ablations_csv, dtype={"train_seed": str})
    base = pd.read_csv(summary_csv, dtype={"train_seed": str})
    base = base[(base["train_seed"] == "all") & (base["condition"] == "no_wind")].set_index("agent")
    lines = ["| Variant | Change | Test return | Success | Δ vs base | Verdict |",
             "|---|---|---:|---:|---:|---|"]
    for name, spec in suite.items():
        row = abl[(abl["variant"] == name) & (abl["train_seed"] == "all")]
        if row.empty:
            continue
        r = row.iloc[0]
        algo = load_config(spec["base"])["algo"]
        b = base.loc[algo]
        delta = r["mean_return"] - b["mean_return"]
        real = abs(delta) > max(r["std_return"], b["std_return"])
        verdict = ("better" if delta > 0 else "worse") + " (likely real)" if real else "inconclusive"
        lines.append(f"| `{name}` | {spec['change']} | {r['mean_return']:.1f} ± {r['std_return']:.1f} | "
                     f"{100 * r['success_rate']:.1f}% | {delta:+.1f} | {verdict} |")
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
    p.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    suite = load_suite(args.suite)
    if args.list:
        for name, spec in suite.items():
            print(f"{name:16s} {spec['change']}")
        return
    if args.report:
        print(report(suite, args.out, args.summary))
        return

    unknown = set(args.variants) - set(suite)
    if unknown or not args.variants:
        raise SystemExit(f"Choose --variants from {list(suite)} (unknown: {sorted(unknown)})")

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

    rows = [row for v in args.variants for row in evaluate_variant(cfgs[v], args.episodes, args.episodes_dir)]
    write_rows(args.out, rows)
    log.info("Saved %s", args.out)


if __name__ == "__main__":
    main()
