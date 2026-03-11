from __future__ import annotations

import pandas as pd

from ..utils.tz import (
    coerce_series_like_index,
    coerce_series_to_tz,
    coerce_ts_to_tz,
    ensure_dtindex_tz,
)


def _to_ex_tz_series(s: pd.Series, ex_tz: str, naive_is_utc: bool) -> pd.Series:
    """Convert or localize a datetime Series to exchange tz."""
    return coerce_series_to_tz(s, ex_tz, naive_is_utc=naive_is_utc)


def _to_ex_tz_timestamp(
    x: pd.Timestamp, ex_tz: str, naive_is_utc: bool
) -> pd.Timestamp:
    """Convert or localize a single timestamp to exchange tz."""
    return coerce_ts_to_tz(x, ex_tz, naive_is_utc=naive_is_utc)


def _ensure_calendar_tz(cal: pd.DatetimeIndex, ex_tz: str) -> pd.DatetimeIndex:
    """Ensure calendar index is in exchange tz."""
    return ensure_dtindex_tz(cal, ex_tz)


def _coerce_like_index(dt_series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Coerce a datetime Series to match index tz; normalize if daily."""
    return coerce_series_like_index(dt_series, idx)
