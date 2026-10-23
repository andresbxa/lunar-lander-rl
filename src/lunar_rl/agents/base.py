"""The single interface every agent in this repo implements.

Keeping it this small means evaluation, GIF recording and (later) robustness
tests are written once and work for random, SB3 and from-scratch agents alike.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Return a discrete action for a single (unbatched) observation.

        ``deterministic=True`` means "exploit": greedy / argmax action, no
        exploration noise. Evaluation always uses it.
        """
        ...
