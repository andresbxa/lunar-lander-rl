"""YAML experiment configs with command-line overrides.

Example::

    cfg = load_config("configs/sb3_ppo.yaml", overrides=["train.total_timesteps=50000"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides or [])


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML (so ``50000`` is an int)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {p!r} is not a mapping")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
