"""Minimal console logging shared by all entry points."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)
