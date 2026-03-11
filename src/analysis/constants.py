"""Constant definitions for analysis."""

from __future__ import annotations

PRICE_COL_CANDIDATES: tuple[str, ...] = (
    "adj_close",
    "close",
    "close_adj",
    "price",
    "px",
)

__all__ = ["PRICE_COL_CANDIDATES"]
