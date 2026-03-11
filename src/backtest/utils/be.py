from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from backtest.utils.common.pairs import parse_pair_symbols as _parse_pair_symbols_ssot
from backtest.utils.common.prices import series_price_at as _series_price_at_base
from ..config.types import Fill, Side

# ---------- Generic helpers ----------


def _side_sign(side: Side) -> int:
    return 1 if str(side).lower() == "buy" else -1


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        v = int(round(float(x)))
        return v
    except Exception:
        return default


def _as_timestamp(x: Any) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(x)
    except Exception:
        return None


def _nan_to_zero(v: Any) -> float:
    try:
        f = float(v)
        return 0.0 if (not np.isfinite(f)) else f
    except Exception:
        return 0.0


def _ticks_from_prices(price_a: float, price_b: float, tick: float) -> float:
    t = _safe_float(tick, 0.0)
    if t <= 0.0:
        return 0.0
    return (float(price_a) - float(price_b)) / t


# ---------- Fills / Fees / Slippage ----------


def _compute_vwap_and_totals(fills: Sequence[Fill]) -> tuple[float | None, float, int]:
    notional = 0.0
    qty = 0
    for f in fills or ():
        q = _safe_int(f.get("qty"), 0)
        p = _safe_float(f.get("price"), np.nan)
        if q > 0 and np.isfinite(p):
            notional += q * p
            qty += q
    if qty <= 0:
        return None, 0.0, 0
    vwap = notional / float(qty)
    return float(vwap), float(notional), int(qty)


def _calc_fees_from_fills(
    fills: Sequence[Fill], *, maker_bps: float, taker_bps: float, min_fee: float = 0.0
) -> tuple[float, int, int]:
    total = 0.0
    m_cnt = t_cnt = 0
    for f in fills or ():
        q = _safe_int(f.get("qty"), 0)
        p = _safe_float(f.get("price"), np.nan)
        if q <= 0 or not np.isfinite(p):
            continue
        notion = q * p
        liq = str(f.get("liquidity", "T") or "T").upper()
        bps = maker_bps if liq == "M" else taker_bps
        if liq == "M":
            m_cnt += 1
        else:
            t_cnt += 1
        total += -(notion * (bps / 10_000.0))  # Kosten negativ, Rebate positiv
    if abs(min_fee) > 0.0:
        total = min(total, -abs(min_fee))
    return float(total), int(m_cnt), int(t_cnt)


def _calc_slippage_from_fills(
    fills: Sequence[Fill], *, side: Side, reference_px: float, tick: float | None = None
) -> dict[str, float]:
    ref = _safe_float(reference_px, np.nan)
    vwap, _notional, qty = _compute_vwap_and_totals(fills)
    vwap_f = _safe_float(vwap, np.nan)
    if not np.isfinite(ref) or qty <= 0 or not np.isfinite(vwap_f):
        return {"slippage_ccy": 0.0, "slippage_ticks_avg": 0.0, "vwap": float("nan")}
    adverse_sum = 0.0
    ticks_sum = 0.0
    for f in fills or ():
        q = _safe_int(f.get("qty"), 0)
        p = _safe_float(f.get("price"), np.nan)
        if q <= 0 or not np.isfinite(p):
            continue
        if side == "buy":
            adv = max(0.0, p - ref)
        else:
            adv = max(0.0, ref - p)
        adverse_sum += adv * q
        if tick and tick > 0:
            tks = (adv / tick) if tick > 0 else 0.0
            ticks_sum += tks * q
    slippage_ccy = adverse_sum  # ≥ 0
    slippage_ticks_avg = (ticks_sum / qty) if (tick and tick > 0 and qty > 0) else 0.0
    return {
        "slippage_ccy": float(slippage_ccy),
        "slippage_ticks_avg": float(slippage_ticks_avg),
        "vwap": float(vwap_f),
    }


def _realized_spread_bps(*, vwap: float, reference_mid: float, side: Side) -> float:
    m = _safe_float(reference_mid, np.nan)
    v = _safe_float(vwap, np.nan)
    if not (np.isfinite(m) and m > 0 and np.isfinite(v)):
        return float("nan")
    s = _side_sign(side)
    return float(s * (m - v) / m * 10_000.0)


def _time_to_fill_ms(order_ts: pd.Timestamp | None, fills: Sequence[Fill]) -> float:
    t0 = _as_timestamp(order_ts) if order_ts is not None else None
    if t0 is None:
        return float("nan")
    t_min: pd.Timestamp | None = None
    for f in fills or ():
        ts = _as_timestamp(f.get("ts"))
        if ts is None:
            continue
        t_min = ts if (t_min is None or ts < t_min) else t_min
    if t_min is None:
        return float("nan")
    return float((t_min - t0).total_seconds() * 1000.0)


# ---------- DataFrame helpers ----------


def _ensure_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = (
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "total_costs",
        "net_pnl",
    )
    out = df.copy()
    for c in need:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out


def _ensure_lob_diag_columns(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    cols = [
        "exec_entry_vwap_y",
        "exec_exit_vwap_y",
        "exec_entry_ticks_y",
        "exec_exit_ticks_y",
        "exec_entry_vwap_x",
        "exec_exit_vwap_x",
        "exec_entry_ticks_x",
        "exec_exit_ticks_x",
        "lob_fill_ratio",
        "realized_spread_bps",
        "time_to_fill_ms",
        "maker_fills",
        "taker_fills",
        "lob_net_pnl",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


# ---------- Pair parsing / Prices ----------


def _parse_pair_symbols_generic(pair_val: Any) -> tuple[str | None, str | None]:
    return _parse_pair_symbols_ssot(pair_val, upper=True)


def _parse_pair_symbols(pair: Any) -> tuple[str | None, str | None]:
    return _parse_pair_symbols_generic(pair)


def _series_price_at(
    price_data: Mapping[str, Union[pd.Series, pd.DataFrame]],
    sym: str | None,
    ts: pd.Timestamp,
) -> Optional[float]:
    if not sym:
        return None
    return _series_price_at_base(
        price_data, sym, ts, prefer_col="close", allow_zero=False
    )


def _infer_annualization_factor(dates: pd.DatetimeIndex) -> int:
    try:
        inferred = pd.infer_freq(dates)
        if inferred is None:
            return 252
        inferred = inferred.upper()
        if inferred.startswith(("B", "D")):
            return 252
        if inferred.startswith("W"):
            return 52
        if inferred.startswith("M"):
            return 12
        if inferred.startswith("Q"):
            return 4
        if inferred.startswith(("A", "Y")):
            return 1
        return 252
    except Exception:
        return 252


def _infer_exit_column(df: pd.DataFrame) -> str | None:
    if df is None or df.empty:
        return None
    lc = {c.lower(): c for c in df.columns}
    candidates = (
        "exit_date",
        "exit",
        "exit_time",
        "exitdt",
        "exit_datetime",
        "close_date",
        "close",
        "close_time",
        "closedt",
        "close_datetime",
        "exit_dt",
        "close_dt",
        "timestamp_exit",
        "timestamp_close",
    )
    for name in candidates:
        if name in lc:
            return lc[name]
    entry_orig = lc.get("entry_date")
    datetime_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    datetime_cols = [
        c for c in datetime_cols if (entry_orig is None or c != entry_orig)
    ]
    if datetime_cols:
        return datetime_cols[0]
    for c in df.columns:
        cl = c.lower()
        if "exit" in cl or "close" in cl or "sell" in cl or "cover" in cl:
            try:
                try:
                    pd.to_datetime(df[c], errors="raise", format="mixed")
                except TypeError:
                    pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    return None
