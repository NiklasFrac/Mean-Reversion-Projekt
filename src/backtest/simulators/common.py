# src/backtest/execute/common

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from backtest.utils.common.prices import price_at_or_prior as _price_at_or_prior
from ..config.types import PricingCfg, Side
from ..utils.be import _safe_float


def pick_reference_price(
    *,
    pricing_cfg: PricingCfg,
    side: Side,
    submit_mid: float | None,
    target_vwap: float | None = None,
) -> float | None:
    policy = str(getattr(pricing_cfg, "reference", "mid_on_submit")).lower()
    if (
        policy == "vwap_target"
        and target_vwap is not None
        and _safe_float(target_vwap, float("nan")) == target_vwap
    ):
        return float(target_vwap)
    if submit_mid is not None and _safe_float(submit_mid, float("nan")) == submit_mid:
        return float(submit_mid)
    return None


def cap_cross_price(
    *, side: Side, submit_mid: float | None, cap_bps: float | None
) -> float | None:
    m = _safe_float(submit_mid, float("nan"))
    if not (m and m > 0):
        return None
    bps = _safe_float(cap_bps, float("nan"))
    if not (bps and bps > 0):
        return None
    frac = bps / 10_000.0
    if side == "buy":
        return float(m * (1.0 + frac))
    else:
        return float(m * (1.0 - frac))


SideTxt = Literal["buy", "sell"]


def infer_units(row: pd.Series, leg: str) -> int:
    leg = leg.lower()
    # 1) per-leg
    for k in (
        f"units_{leg}",
        f"qty_{leg}",
        f"quantity_{leg}",
        f"contracts_{leg}",
        f"size_{leg}",
    ):
        if k in row and pd.notna(row[k]):
            try:
                return max(1, int(round(abs(float(row[k])))))
            except Exception:
                pass
    # 2) generic
    for k in ("units", "size", "qty", "quantity"):
        if k in row and pd.notna(row[k]):
            try:
                return max(1, int(round(abs(float(row[k])))))
            except Exception:
                pass
    # 3) notional / price
    notional = None
    for k in (
        f"notional_{leg}",
        f"{leg[0]}_notional",
        f"leg{'1' if leg.startswith('y') else '2'}_notional",
    ):
        if k in row and pd.notna(row[k]):
            try:
                notional = abs(float(row[k]))
                break
            except Exception:
                pass
    price = None
    for k in (
        f"exec_entry_vwap_{leg}",
        f"entry_price_{leg}",
        f"{leg}_price",
        f"price_{leg}",
        f"entry_{leg}",
        f"px_{leg}",
    ):
        if k in row and pd.notna(row[k]):
            try:
                price = abs(float(row[k]))
                break
            except Exception:
                pass
    if notional and price and price > 0:
        return max(1, int(round(notional / price)))
    return 1


def infer_side(row: pd.Series, leg: str, *, default: SideTxt = "buy") -> SideTxt:
    leg = leg.lower()
    # explicit fields first
    for k in (f"side_{leg}", f"{leg}_side", f"dir_{leg}", f"{leg}_dir"):
        s = row.get(k)
        if isinstance(s, str) and s.strip():
            x = s.strip().lower()
            if x in {"buy", "long", "+", "b", "l", "1"}:
                return "buy"
            if x in {"sell", "short", "-", "s", "sh", "-1"}:
                return "sell"
    # sign from raw qty if present (before abs())
    for k in (
        f"units_{leg}",
        f"qty_{leg}",
        f"quantity_{leg}",
        f"contracts_{leg}",
        "units",
        "size",
        "qty",
        "quantity",
    ):
        if k in row and pd.notna(row[k]):
            try:
                v = float(row[k])
                if np.isfinite(v) and v != 0:
                    return "sell" if v < 0 else "buy"
            except Exception:
                pass
    return default


def opposite_side(side: SideTxt) -> SideTxt:
    return "buy" if side == "sell" else "sell"


def price_at_or_prior(
    series: pd.Series | None, ts: pd.Timestamp, *, allow_zero: bool = False
) -> float | None:
    return _price_at_or_prior(series, ts, allow_zero=allow_zero)


def limit_from_bps(
    side: SideTxt, ref_px: float | None, bps: float | None
) -> float | None:
    if ref_px is None or not np.isfinite(ref_px) or ref_px <= 0 or not bps:
        return None
    frac = float(bps) / 10_000.0
    return ref_px * (1.0 + frac) if side == "buy" else ref_px * (1.0 - frac)
