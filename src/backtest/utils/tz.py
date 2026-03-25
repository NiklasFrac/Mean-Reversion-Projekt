"""Timezone SSOT for backtest runtime contracts and boundary alignment."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

__all__ = [
    "NY_TZ",
    "utc_now",
    "assert_ny_index",
    "assert_ny_series",
    "assert_ny_frame",
    "to_ny_timestamp",
    "to_ny_index",
    "to_ny_series",
    "align_ts_to_series",
    "align_ts_to_index",
    "align_index_to_index",
    "to_naive_local",
    "to_naive_day",
    "same_tz_or_raise",
]

NY_TZ = "America/New_York"


def _tz_to_str(tz: Any) -> str | None:
    if tz is None:
        return None
    for attr in ("key", "zone"):
        val = getattr(tz, attr, None)
        if isinstance(val, str) and val:
            return val
    s = str(tz)
    return s if s and s.lower() != "none" else None


def _normalize_tz_name(name: str | None) -> str | None:
    if not name:
        return None
    aliases = {
        "US/Eastern": NY_TZ,
        "EST": NY_TZ,
        "EDT": NY_TZ,
        "America/NewYork": NY_TZ,
    }
    return aliases.get(name, name)


def _extract_tz_from_index_like(obj: Any) -> str | None:
    try:
        if isinstance(obj, pd.DatetimeIndex):
            return _normalize_tz_name(_tz_to_str(obj.tz))
        if isinstance(obj, pd.Series) and hasattr(obj, "dt"):
            return _normalize_tz_name(_tz_to_str(getattr(obj.dt, "tz", None)))
        ts = pd.Timestamp(obj)
        return _normalize_tz_name(_tz_to_str(ts.tz))
    except Exception:
        return None


def _context_suffix(context: str) -> str:
    return f" ({context})" if context else ""


def _localize_or_convert_timestamp(
    ts: pd.Timestamp | str | dt.datetime,
    tz_name: str,
) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if pd.isna(out):
        return pd.NaT  # type: ignore[return-value]
    if out.tzinfo is None:
        return out.tz_localize(tz_name)
    return out.tz_convert(tz_name)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_ny_timestamp(ts: pd.Timestamp | str | dt.datetime) -> pd.Timestamp:
    return _localize_or_convert_timestamp(ts, NY_TZ)


def to_ny_index(idx: pd.DatetimeIndex | Any) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
    if out.tz is None:
        return out.tz_localize(NY_TZ)
    return out.tz_convert(NY_TZ)


def to_ny_series(values: pd.Series | Any) -> pd.Series:
    out = pd.to_datetime(values, errors="coerce")
    if isinstance(out, pd.Series):
        if isinstance(out.dtype, pd.DatetimeTZDtype):
            return out.dt.tz_convert(NY_TZ)
        return out.dt.tz_localize(NY_TZ)
    series = pd.Series(out)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(NY_TZ)
    return series.dt.tz_localize(NY_TZ)


def assert_ny_index(
    idx: pd.DatetimeIndex,
    *,
    context: str = "",
) -> pd.DatetimeIndex:
    suffix = _context_suffix(context)
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex{suffix}.")
    if bool(idx.isna().any()):
        raise ValueError(f"DatetimeIndex contains NaT values{suffix}.")
    tz_name = _normalize_tz_name(_tz_to_str(idx.tz))
    if tz_name is None:
        raise ValueError(
            f"DatetimeIndex must be tz-aware in {NY_TZ}, got tz-naive index{suffix}."
        )
    if tz_name != NY_TZ:
        raise ValueError(
            f"DatetimeIndex must use {NY_TZ}, got {tz_name!r}{suffix}."
        )
    if not idx.is_monotonic_increasing:
        raise ValueError(f"DatetimeIndex must be sorted ascending{suffix}.")
    if bool(idx.has_duplicates):
        raise ValueError(f"DatetimeIndex must not contain duplicates{suffix}.")
    return idx


def assert_ny_series(
    series: pd.Series,
    *,
    context: str = "",
) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"Expected Series{_context_suffix(context)}.")
    assert_ny_index(series.index, context=context)
    return series


def assert_ny_frame(
    frame: pd.DataFrame,
    *,
    context: str = "",
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Expected DataFrame{_context_suffix(context)}.")
    assert_ny_index(frame.index, context=context)
    return frame


def align_ts_to_series(
    ts: pd.Timestamp | str | dt.datetime,
    ser: pd.Series,
) -> pd.Timestamp:
    s_tz: str | None = None
    if hasattr(ser, "dt"):
        try:
            s_tz = _normalize_tz_name(_tz_to_str(getattr(ser.dt, "tz", None)))
        except Exception:
            s_tz = None
    if s_tz is None and isinstance(ser.index, pd.DatetimeIndex):
        s_tz = _normalize_tz_name(_tz_to_str(ser.index.tz))

    out = pd.Timestamp(ts)
    if pd.isna(out):
        return pd.NaT  # type: ignore[return-value]
    if s_tz is not None:
        return _localize_or_convert_timestamp(out, s_tz)
    return out.tz_localize(None) if out.tzinfo is not None else out


def align_ts_to_index(
    ts: pd.Timestamp | str | dt.datetime,
    idx: pd.DatetimeIndex,
) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if pd.isna(out) or not isinstance(idx, pd.DatetimeIndex):
        return out
    idx_tz = _normalize_tz_name(_tz_to_str(idx.tz))
    if idx_tz is None:
        return out.tz_localize(None) if out.tzinfo is not None else out
    return _localize_or_convert_timestamp(out, idx_tz)


def align_index_to_index(
    idx: pd.DatetimeIndex | Any,
    ref_idx: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
    if not isinstance(ref_idx, pd.DatetimeIndex):
        return out
    ref_tz = _normalize_tz_name(_tz_to_str(ref_idx.tz))
    if ref_tz is None:
        return out.tz_localize(None) if out.tz is not None else out
    if out.tz is None:
        return out.tz_localize(ref_tz)
    return out.tz_convert(ref_tz)


def to_naive_local(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.tz_localize(None) if obj.tz is not None else obj
    if isinstance(obj, pd.DatetimeIndex):
        return obj.tz_localize(None) if obj.tz is not None else obj
    if isinstance(obj, pd.Series):
        try:
            tz_name = _extract_tz_from_index_like(obj)
            if tz_name is not None and hasattr(obj, "dt"):
                return obj.dt.tz_localize(None)
        except Exception:
            pass
        return obj
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        return out
    return obj


def to_naive_day(obj: Any) -> Any:
    out = to_naive_local(obj)
    if isinstance(out, pd.Timestamp):
        return out.normalize()
    if isinstance(out, pd.DatetimeIndex):
        return out.normalize()
    if isinstance(out, pd.Series):
        try:
            if hasattr(out, "dt"):
                return out.dt.normalize()
        except Exception:
            pass
        return out
    if isinstance(out, pd.DataFrame):
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.copy()
            out.index = out.index.normalize()
        return out
    return out


def same_tz_or_raise(
    idx1: pd.DatetimeIndex | pd.Series | pd.Timestamp | Any,
    idx2: pd.DatetimeIndex | pd.Series | pd.Timestamp | Any,
    *,
    allow_naive_pair: bool = False,
    context: str = "",
) -> None:
    tz1 = _extract_tz_from_index_like(idx1)
    tz2 = _extract_tz_from_index_like(idx2)

    if tz1 == tz2 and tz1 is not None:
        return
    if tz1 is None and tz2 is None:
        if allow_naive_pair:
            return
        raise ValueError(
            f"Both operands are tz-naive{_context_suffix(context)}."
        )
    raise ValueError(
        "Incompatible timezones: "
        f"left={tz1!r}, right={tz2!r}{_context_suffix(context)}."
    )
