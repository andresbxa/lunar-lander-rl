"""Seeding helpers and the project-wide seed protocol.

Three disjoint seed ranges are used so that no number we report was used to
make a decision:

* **train** seeds (0, 1, 2, ...): network init, exploration noise and the
  terrain of training episodes.
* **validation** seeds (1000 ...): periodic evaluation during training, used
  to pick the "best" checkpoint.
* **test** seeds (10000 ...): final evaluation only. Because checkpoint
  selection never sees these episodes, the reported score is not inflated by
  selection bias.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

VAL_SEED_START = 1_000
TEST_SEED_START = 10_000


def seed_range(start: int, n: int) -> list[int]:
    """Return ``n`` consecutive episode seeds starting at ``start``."""
    return list(range(start, start + n))


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
