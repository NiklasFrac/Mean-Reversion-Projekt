from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.config.types import BorrowConfig
from backtest.utils.tz import to_naive_day

logger = logging.getLogger("backtest.borrow.events")
_EVENT_COLUMNS = [
    "date",
    "symbol",
    "type",
    "rate_annual",
    "locate_fee_bps",
    "lead_days",
    "notes",
]
def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=_EVENT_COLUMNS)


def _symbol_seed(sym: str) -> int:
    s = str(sym).strip().upper()
    # deterministic across processes; avoid Python's hash randomization
    acc = 0
    for ch in s:
        acc = (acc * 131 + ord(ch)) % (2**31 - 1)
    return int(acc)


def generate_borrow_events(
    *,
    universe: list[str] | tuple[str, ...] | set[str],
    day: Any,
    borrow_cfg: BorrowConfig | None = None,
    lead_days: Any | None = None,
    locate_fee_bps: Any | None = None,
    availability_df: pd.DataFrame | None = None,
    borrow_ctx: Any | None = None,
) -> pd.DataFrame:
    """
    Synthetic Borrow events generator.

    Returns a normalized long DataFrame with:
      ["date","symbol","type","rate_annual","locate_fee_bps","lead_days","notes"]

    - If `borrow_ctx` provides resolve_borrow_rate(symbol, day), uses it.
    - Else uses the provided typed `borrow_cfg`.
    - Availability is passed through only as a tag in 'notes' (no enforcement here).
    """
    syms = sorted({str(s).strip().upper() for s in universe if str(s).strip()})
    day_ts = pd.to_datetime(day, errors="coerce")
    if pd.isna(day_ts):
        d = pd.NaT
    else:
        d = to_naive_day(pd.Timestamp(day_ts))
    if not syms or pd.isna(d):
        return _empty_events()

    if borrow_cfg is not None and not isinstance(borrow_cfg, BorrowConfig):
        raise TypeError("borrow_cfg must be a BorrowConfig")

    enabled = bool(borrow_cfg.enabled) if borrow_cfg is not None else False
    if borrow_ctx is not None and hasattr(borrow_ctx, "enabled"):
        try:
            enabled = bool(getattr(borrow_ctx, "enabled"))
        except Exception:
            pass
    if not enabled:
        return _empty_events()

    try:
        default_rate = (
            float(getattr(borrow_ctx, "default_rate_annual"))
            if borrow_ctx is not None
            else float(borrow_cfg.default_rate_annual if borrow_cfg is not None else 0.0)
        )
    except Exception:
        default_rate = float(
            borrow_cfg.default_rate_annual if borrow_cfg is not None else 0.0
        )

    try:
        day_basis = (
            int(getattr(borrow_ctx, "day_basis"))
            if borrow_ctx is not None
            else int(borrow_cfg.day_basis if borrow_cfg is not None else 252)
        )
    except Exception:
        day_basis = int(borrow_cfg.day_basis if borrow_cfg is not None else 252)

    try:
        jitter_sigma = float(
            borrow_cfg.synthetic_jitter_sigma if borrow_cfg is not None else 0.0
        )
    except Exception:
        jitter_sigma = 0.0

    # Optional locate fee (bps) / lead days.
    ld = None if lead_days is None else int(lead_days)
    try:
        lfb = None if locate_fee_bps is None else float(locate_fee_bps)
    except Exception:
        lfb = None

    rows: list[dict[str, Any]] = []
    for sym in syms:
        rate = default_rate
        if borrow_ctx is not None:
            resolve_fn = getattr(borrow_ctx, "resolve_borrow_rate", None)
            if callable(resolve_fn):
                try:
                    r = resolve_fn(sym, d)
                    if r is not None:
                        rate = float(r)
                except Exception:
                    rate = default_rate

        rate_jit = float(rate)
        if jitter_sigma > 0:
            # Deterministic per-symbol wiggle; off by default.
            rng = np.random.RandomState(
                (_symbol_seed(sym) + int(d.value // 10**9)) % (2**31 - 1)
            )
            rate_jit = float(rate) * (
                1.0 + float(rng.normal(loc=0.0, scale=jitter_sigma))
            )
            rate_jit = float(np.clip(rate_jit, 0.0, 10.0))

        notes = f"synthetic(day_basis={day_basis})"
        if (
            availability_df is not None
            and isinstance(availability_df, pd.DataFrame)
            and not availability_df.empty
        ):
            notes += ";availability_hint"

        rows.append(
            {
                "date": d,
                "symbol": sym,
                "type": "borrow_rate",
                "rate_annual": rate_jit,
                "locate_fee_bps": lfb,
                "lead_days": ld,
                "notes": notes,
            }
        )

    out = pd.DataFrame(rows)
    out["date"] = to_naive_day(pd.to_datetime(out["date"], errors="coerce"))
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["type"] = out["type"].astype(str).str.lower()
    out["rate_annual"] = pd.to_numeric(out["rate_annual"], errors="coerce")
    if "locate_fee_bps" in out.columns:
        out["locate_fee_bps"] = pd.to_numeric(out["locate_fee_bps"], errors="coerce")
    if "lead_days" in out.columns:
        out["lead_days"] = pd.to_numeric(out["lead_days"], errors="coerce").astype(
            "Int64"
        )
    return out[_EVENT_COLUMNS]
