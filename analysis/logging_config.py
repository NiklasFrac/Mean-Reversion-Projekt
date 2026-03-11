"""Logging setup shared by data analysis modules."""

from __future__ import annotations

import logging

logger = logging.getLogger("data_analysis")
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_ch)
logger.setLevel(logging.INFO)

__all__ = ["logger"]
