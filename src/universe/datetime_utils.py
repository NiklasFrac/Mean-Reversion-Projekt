from __future__ import annotations

from typing import Any, Literal

import pandas as pd


def coerce_utc_naive_timestamp(
    value: Any,
    *,
    normalize: bool = False,
    errors: Literal["coerce", "raise"] = "coerce",
) -> pd.Timestamp | None:
    try:
        ts_any = pd.to_datetime(value, errors=errors)
    except Exception:
        if errors == "raise":
            raise
        return None
    ts: Any
    if isinstance(ts_any, pd.DatetimeIndex):
        ts = ts_any[0] if len(ts_any) else pd.NaT
    else:
        ts = ts_any
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert("UTC")
        except Exception:
            pass
        ts = ts.tz_localize(None)
    if normalize:
        ts = ts.normalize()
    return ts


def coerce_date_like_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    ts = coerce_utc_naive_timestamp(value, errors="coerce")
    if ts is None:
        return value
    return ts.date().isoformat()
