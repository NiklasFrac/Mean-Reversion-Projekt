from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.utils.tz import (
    NY_TZ,
    align_ts_to_series,
    ensure_dtindex_tz,
    ensure_index_tz,
    to_naive_local,
)

__all__ = [
    "as_price_map",
    "price_at_or_prior",
    "series_price_at",
]


def _coerce_timezone_series(s: pd.Series, coerce_timezone: str | None) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    pol = str(coerce_timezone or "keep").strip().lower()
    if pol in {"keep", "none", "off"}:
        return s
    if pol in {"exchange", "ny", "new_york"}:
        return ensure_index_tz(s, NY_TZ)
    if pol in {"utc"}:
        s2 = ensure_index_tz(s, NY_TZ)
        s2 = s2.copy()
        s2.index = ensure_dtindex_tz(cast(pd.DatetimeIndex, s2.index), "UTC")
        return s2
    if pol in {"naive", "local_naive"}:
        s2 = ensure_index_tz(s, NY_TZ)
        s2 = s2.copy()
        s2.index = to_naive_local(cast(pd.DatetimeIndex, s2.index))
        return s2
    # unknown policy -> keep
    return s


def as_price_map(
    price_data: pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame],
    *,
    prefer_col: str = "close",
    coerce_numeric: bool = True,
    coerce_timezone: str | None = None,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    def _norm_series(s: pd.Series) -> pd.Series:
        s2 = s.copy()
        s2 = _coerce_timezone_series(s2, coerce_timezone)
        if not s2.index.is_monotonic_increasing:
            s2 = s2.sort_index(kind="mergesort")
        s2 = s2.loc[~s2.index.duplicated()]
        if coerce_numeric:
            s2 = pd.to_numeric(s2, errors="coerce").astype(float)
        return s2

    if isinstance(price_data, pd.DataFrame):
        for c in price_data.columns:
            out[str(c)] = _norm_series(price_data[c])
        return out

    if isinstance(price_data, Mapping):
        for sym, v in price_data.items():
            if isinstance(v, pd.Series):
                out[str(sym)] = _norm_series(v)
            elif isinstance(v, pd.DataFrame):
                if prefer_col in v.columns:
                    out[str(sym)] = _norm_series(v[prefer_col])
                else:
                    num_cols = [
                        col
                        for col in v.columns
                        if pd.api.types.is_numeric_dtype(v[col])
                    ]
                    col = num_cols[0] if num_cols else v.columns[0]
                    out[str(sym)] = _norm_series(v[col])
    return out


def price_at_or_prior(
    series: pd.Series | None,
    ts: pd.Timestamp | str,
    *,
    allow_zero: bool = False,
    coerce_index: bool = False,
    dropna: bool = False,
    align_ts: bool = True,
) -> float | None:
    """
    Return the last price at-or-prior to `ts`.

    Parameters
    ----------
    allow_zero : bool
        If True, allow zero prices (>=0). If False, require > 0.
    coerce_index : bool
        If True, coerce index to DatetimeIndex (drops NaT rows).
    dropna : bool
        If True, drop NaNs before asof lookup.
    align_ts : bool
        If True, align `ts` to the series' timezone semantics.
    """
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return None
    s = pd.to_numeric(series, errors="coerce")

    if coerce_index:
        try:
            idx_dt = pd.to_datetime(s.index, errors="coerce", format="mixed")
        except TypeError:
            idx_dt = pd.to_datetime(s.index, errors="coerce")
        mask = ~pd.isna(idx_dt)
        if not bool(mask.any()):
            return None
        s = s.loc[mask].copy()
        s.index = pd.DatetimeIndex(idx_dt[mask])
    else:
        if not isinstance(s.index, pd.DatetimeIndex):
            return None

    if dropna:
        s = s.dropna()
    if s.empty:
        return None

    if not s.index.is_monotonic_increasing:
        s = s.sort_index(kind="mergesort")

    t = align_ts_to_series(ts, s) if align_ts else pd.Timestamp(ts)
    pos = int(s.index.searchsorted(t, side="right") - 1)
    if pos < 0:
        return None
    try:
        v = float(s.iloc[pos])
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if allow_zero:
        return v if v >= 0 else None
    return v if v > 0 else None


def _extract_series(obj: Any, *, prefer_col: str = "close") -> pd.Series | None:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if prefer_col in obj.columns:
            s = obj[prefer_col]
            return s if isinstance(s, pd.Series) else None
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if not num_cols:
            return None
        s = obj[num_cols[0]]
        return s if isinstance(s, pd.Series) else None
    return None


def series_price_at(
    price_data: pd.Series | Mapping[str, pd.Series | pd.DataFrame],
    symbol: str | None,
    ts: pd.Timestamp | str,
    *,
    prefer_col: str = "close",
    allow_zero: bool = False,
    coerce_index: bool = False,
    dropna: bool = False,
    align_ts: bool = True,
) -> float | None:
    """
    Price at-or-prior for a symbol in a mapping (Series/DataFrame).
    """
    s: pd.Series | None
    if isinstance(price_data, pd.Series):
        s = price_data
    elif isinstance(price_data, Mapping):
        if not symbol:
            return None
        obj = price_data.get(symbol)
        s = _extract_series(obj, prefer_col=prefer_col)
    else:
        s = None

    if s is None:
        return None
    return price_at_or_prior(
        s,
        ts,
        allow_zero=allow_zero,
        coerce_index=coerce_index,
        dropna=dropna,
        align_ts=align_ts,
    )
