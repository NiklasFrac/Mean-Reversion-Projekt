# src/backtest/config/types.py
"""
Small stable type definitions for config and execution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

import pandas as pd

__all__ = ["Fill", "Side", "BorrowCtx", "PricingCfg"]


class Fill(TypedDict, total=False):
    qty: int
    price: float
    ts: pd.Timestamp
    liquidity: Literal["M", "T"]
    order_id: str


Side = Literal["buy", "sell"]


class BorrowCtx(Protocol):
    day_basis: int


@dataclass(frozen=True)
class PricingCfg:
    reference: str = "mid_on_submit"
