"""Timezone utilities for Pandas objects (Series/DataFrame/Index/Timestamp).

Project policy:
- Primary exchange timezone: "America/New_York" (NY_TZ).
- naive_is_utc = False.
- Never use tz_convert(None); to drop tz, use tz_localize(None).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "NY_TZ",
    "NAIVE_IS_UTC",
    "get_naive_is_utc",
    "get_ex_tz",
    "utc_now",
    "ensure_dtindex_tz",
    "ensure_index_tz",
    "align_ts_to_series",
    "align_ts_to_index",
    "coerce_ts_to_tz",
    "coerce_series_to_tz",
    "coerce_series_like_index",
    "to_naive_local",
    "to_naive_utc",
    "to_naive_day",
    "same_tz_or_raise",
]

NY_TZ = "America/New_York"
logger = logging.getLogger(__name__)
NAIVE_IS_UTC: bool = str(os.getenv("WF_NAIVE_IS_UTC", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# ---- typing helpers for tz_localize signatures --------------------------------

# DatetimeIndex.tz_localize:
#   nonexistent: Literal[...] | Timedelta
#   ambiguous:   Literal['infer','NaT','raise'] | ndarray[bool]
NonexistentParam: TypeAlias = (
    Literal["shift_forward", "shift_backward", "NaT", "raise"]
    | pd.Timedelta
    | dt.timedelta
)
AmbiguousIdxParam: TypeAlias = Literal["infer", "NaT", "raise"] | NDArray[np.bool_]

# Timestamp.tz_localize:
#   nonexistent: Literal[...] | Timedelta
#   ambiguous:   Literal['raise','NaT'] | bool
AmbiguousTsParam: TypeAlias = Literal["raise", "NaT"] | bool


# ---- internals -----------------------------------------------------------------


def _tz_to_str(tz: Any) -> str | None:
    """Return IANA tz name if available, else None."""
    if tz is None:
        return None
    for attr in ("key", "zone"):
        val = getattr(tz, attr, None)
        if isinstance(val, str) and val:
            return val
    s = str(tz)
    return s if s and s.lower() != "none" else None


def _extract_tz_from_index_like(obj: Any) -> str | None:
    """Try to extract tz name from DatetimeIndex/Series/Timestamp; None if naive."""
    try:
        if isinstance(obj, pd.DatetimeIndex):
            return _tz_to_str(obj.tz)
        if isinstance(obj, pd.Series):
            if hasattr(obj, "dt"):
                return _tz_to_str(getattr(obj.dt, "tz", None))
        ts = pd.Timestamp(obj)
        return _tz_to_str(ts.tz)
    except Exception:
        return None


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Safe nested dict access; returns None when missing."""
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _normalize_tz_name(name: str | None) -> str | None:
    """Normalize common timezone aliases to IANA names."""
    if not name:
        return None
    aliases = {
        "US/Eastern": NY_TZ,
        "EST": NY_TZ,  # offset aliases normalized to IANA
        "EDT": NY_TZ,
        "America/NewYork": NY_TZ,
    }
    return aliases.get(name, name)


def _infer_utc_hint(
    values: pd.Series | pd.Index | Sequence[Any], *, sample_size: int = 200
) -> bool:
    """
    Best-effort detector for timezone-aware string payloads.
    Returns True when sampled values look like they carry explicit UTC/offset suffixes.
    """
    try:
        s = values if isinstance(values, pd.Series) else pd.Series(values)
        sample = pd.Series(s).dropna().astype(str).head(max(1, int(sample_size)))
        if sample.empty:
            return False
        return bool(sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any())
    except Exception:
        return False


# ---- public API ----------------------------------------------------------------


def utc_now() -> dt.datetime:
    """UTC-aware current timestamp."""
    return dt.datetime.now(dt.timezone.utc)


def get_naive_is_utc() -> bool:
    """Return the env-configured naive_is_utc policy."""
    return bool(NAIVE_IS_UTC)


def get_ex_tz(
    cfg: Mapping[str, Any],
    prices: pd.Series | pd.DataFrame | None = None,
    default: str = NY_TZ,
) -> str:
    """Derive exchange timezone from env/cfg/prices, fallback to default."""
    env_tz = os.getenv("WF_EXCHANGE_TZ")
    if env_tz:
        norm = _normalize_tz_name(env_tz)
        if norm:
            return norm

    for path in (
        ["backtest", "timezone"],
        ["time", "timezone"],
        ["determinism", "timezone"],
        ["data", "timezone"],
        ["backtest", "calendar", "tz"],
        ["backtest", "calendars", "tz"],
        ["venues", "XNYS", "tz"],
    ):
        val = _get_nested(cfg, path)
        if isinstance(val, str) and val.strip():
            norm = _normalize_tz_name(val.strip())
            if norm:
                return norm

    if prices is not None:
        idx = (
            prices.index
            if isinstance(prices, pd.DataFrame)
            else getattr(prices, "index", None)
        )
        if isinstance(idx, pd.DatetimeIndex):
            tzs = _tz_to_str(idx.tz)
            if tzs:
                return tzs

    return default


def ensure_index_tz(
    obj: pd.Series | pd.DataFrame,
    tz: str,
    *,
    inplace: bool = False,
    nonexistent: NonexistentParam = "shift_forward",
    ambiguous: AmbiguousIdxParam = "NaT",
) -> pd.Series | pd.DataFrame:
    """Ensure Series/DataFrame index is in target tz (localize or convert)."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj if inplace else obj.copy()

    idx: pd.DatetimeIndex = obj.index
    target = _normalize_tz_name(tz) or tz

    if idx.tz is None:
        new_idx = idx.tz_localize(target, nonexistent=nonexistent, ambiguous=ambiguous)
    else:
        cur = _tz_to_str(idx.tz)
        if cur == target:
            return obj
        new_idx = idx.tz_convert(target)

    if inplace:
        obj.index = new_idx
        return obj
    out = obj.copy()
    out.index = new_idx
    return out


def ensure_dtindex_tz(
    idx: pd.DatetimeIndex,
    tz: str,
    *,
    naive_is_utc: bool | None = None,
    nonexistent: NonexistentParam = "shift_forward",
    ambiguous: AmbiguousIdxParam = "NaT",
) -> pd.DatetimeIndex:
    """Ensure a DatetimeIndex is in target tz (localize or convert)."""
    target = _normalize_tz_name(tz) or tz
    if idx.tz is None:
        if bool(naive_is_utc):
            return idx.tz_localize("UTC").tz_convert(target)
        return idx.tz_localize(target, nonexistent=nonexistent, ambiguous=ambiguous)
    cur = _tz_to_str(idx.tz)
    if cur == target:
        return idx
    return idx.tz_convert(target)


def coerce_ts_to_tz(
    ts: pd.Timestamp | str | dt.datetime,
    tz: str,
    *,
    naive_is_utc: bool | None = None,
    nonexistent: NonexistentParam = "shift_forward",
    ambiguous: AmbiguousTsParam = "NaT",
) -> pd.Timestamp:
    """Convert or localize a single timestamp to target tz."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return pd.NaT  # type: ignore[return-value]
    target = _normalize_tz_name(tz) or tz
    t_tz = _tz_to_str(getattr(t, "tz", None))
    if t_tz is None:
        if bool(naive_is_utc):
            return t.tz_localize("UTC").tz_convert(target)
        return t.tz_localize(target, nonexistent=nonexistent, ambiguous=ambiguous)
    if t_tz != target:
        return t.tz_convert(target)
    return t


def coerce_series_to_tz(
    s: pd.Series,
    tz: str,
    *,
    naive_is_utc: bool | None = None,
    utc_hint: bool | Literal["auto"] = False,
    errors: Literal["raise", "coerce"] = "coerce",
) -> pd.Series:
    """
    Convert/localize a datetime Series to target tz.

    `utc_hint="auto"` infers whether parsing should force UTC (useful for
    mixed timezone-aware string payloads).
    """
    utc_flag = _infer_utc_hint(s) if utc_hint == "auto" else bool(utc_hint)
    s2 = pd.to_datetime(s, errors=errors, utc=utc_flag)
    if s2.empty:
        return s2
    target = _normalize_tz_name(tz) or tz
    cur_tz = _tz_to_str(getattr(getattr(s2, "dt", None), "tz", None))
    if cur_tz is None:
        if bool(naive_is_utc):
            return s2.dt.tz_localize("UTC").dt.tz_convert(target)
        return s2.dt.tz_localize(target)
    if cur_tz != target:
        return s2.dt.tz_convert(target)
    return s2


def align_ts_to_series(
    ts: pd.Timestamp | str | dt.datetime,
    ser: pd.Series,
    *,
    naive_is_utc: bool | None = None,
    nonexistent: NonexistentParam = "shift_forward",
    ambiguous: AmbiguousTsParam = "NaT",
) -> pd.Timestamp:
    """Align single timestamp to the timezone semantics of a Series."""
    s_tz: str | None = None
    # prefer tz of datetime-like values
    if hasattr(ser, "dt"):
        try:
            s_tz = _tz_to_str(getattr(ser.dt, "tz", None))
        except Exception:
            s_tz = None
    # fallback: tz from DatetimeIndex (common for numeric values)
    if s_tz is None and isinstance(ser.index, pd.DatetimeIndex):
        s_tz = _tz_to_str(ser.index.tz)

    t = pd.Timestamp(ts)
    t_tz = _tz_to_str(t.tz)

    if s_tz is not None:
        if t_tz is None:
            if bool(naive_is_utc):
                return t.tz_localize("UTC").tz_convert(s_tz)
            return t.tz_localize(s_tz, nonexistent=nonexistent, ambiguous=ambiguous)
        if t_tz != s_tz:
            return t.tz_convert(s_tz)
        return t

    # series is naive → drop tz on aware timestamp
    if t_tz is not None:
        return t.tz_localize(None)
    return t


def align_ts_to_index(
    ts: pd.Timestamp | str | dt.datetime,
    idx: pd.DatetimeIndex,
    *,
    naive_is_utc: bool | None = None,
    nonexistent: NonexistentParam = "shift_forward",
    ambiguous: AmbiguousTsParam = "NaT",
) -> pd.Timestamp:
    """Align single timestamp to the timezone semantics of a DatetimeIndex."""
    t = pd.Timestamp(ts)
    if not isinstance(idx, pd.DatetimeIndex):
        return t
    idx_tz = _tz_to_str(idx.tz)
    t_tz = _tz_to_str(t.tz)
    if idx_tz is None:
        return t.tz_localize(None) if t_tz is not None else t
    if t_tz is None:
        if bool(naive_is_utc):
            return t.tz_localize("UTC").tz_convert(idx_tz)
        return t.tz_localize(idx_tz, nonexistent=nonexistent, ambiguous=ambiguous)
    if t_tz != idx_tz:
        return t.tz_convert(idx_tz)
    return t


def coerce_series_like_index(
    dt_series: pd.Series,
    idx: pd.DatetimeIndex,
    *,
    naive_is_utc: bool | None = None,
) -> pd.Series:
    """
    Coerce a datetime-like Series to match an index tz; normalize if index is daily.
    """
    s = pd.to_datetime(dt_series, errors="coerce")
    if s.empty:
        return s
    tz_idx = _tz_to_str(getattr(idx, "tz", None))
    tz_s = _tz_to_str(getattr(getattr(s, "dt", None), "tz", None))

    if tz_idx is None:
        if tz_s is not None:
            s = s.dt.tz_localize(None)
    else:
        if tz_s is None:
            if bool(naive_is_utc):
                s = s.dt.tz_localize("UTC").dt.tz_convert(tz_idx)
            else:
                s = s.dt.tz_localize(tz_idx)
        else:
            s = s.dt.tz_convert(tz_idx)

    try:
        has_time = any(getattr(ts, "hour", 0) != 0 for ts in idx[:64])
    except Exception:
        has_time = False
    if not has_time:
        s = s.dt.normalize()
    return s


def to_naive_local(obj: Any) -> Any:
    """Drop tz locally (wall-clock) for Timestamp/Index/Series/DataFrame."""
    if isinstance(obj, pd.Timestamp):
        return obj.tz_localize(None) if obj.tz is not None else obj
    if isinstance(obj, pd.DatetimeIndex):
        return obj.tz_localize(None) if obj.tz is not None else obj
    if isinstance(obj, pd.Series):
        try:
            if hasattr(obj, "dt"):
                tz = _tz_to_str(getattr(obj.dt, "tz", None))
                if tz is not None:
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


def to_naive_utc(obj: Any) -> Any:
    """Convert tz-aware values to UTC, then drop tz (naive)."""
    if isinstance(obj, pd.Timestamp):
        if obj.tz is not None:
            return obj.tz_convert("UTC").tz_localize(None)
        return obj
    if isinstance(obj, pd.DatetimeIndex):
        if obj.tz is not None:
            return obj.tz_convert("UTC").tz_localize(None)
        return obj
    if isinstance(obj, pd.Series):
        try:
            if hasattr(obj, "dt"):
                tz = _tz_to_str(getattr(obj.dt, "tz", None))
                if tz is not None:
                    return obj.dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            pass
        return obj
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
            out.index = out.index.tz_convert("UTC").tz_localize(None)
        return out
    return obj


def to_naive_day(obj: Any) -> Any:
    """Drop tz locally and normalize to day boundary when applicable."""
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
    """Validate tz compatibility; raise on mismatch."""
    tz1 = _extract_tz_from_index_like(idx1)
    tz2 = _extract_tz_from_index_like(idx2)

    if tz1 == tz2 and tz1 is not None:
        return
    if tz1 is None and tz2 is None:
        if allow_naive_pair:
            return
        raise ValueError(
            f"Both operands are tz-naive (context={context!r}). "
            "Localize one side or allow_naive_pair=True."
        )
    raise ValueError(
        "Incompatible timezones: "
        f"left={tz1!r}, right={tz2!r}, context={context!r}. "
        "Localize/convert before comparing."
    )


if __name__ == "__main__":  # pragma: no cover
    # minimal self-checks
    logging.basicConfig(level=logging.INFO)
    s = pd.Series([1, 2, 3], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    s_ny = ensure_index_tz(s, NY_TZ)
    assert _extract_tz_from_index_like(s_ny.index) == NY_TZ
    t = align_ts_to_series(pd.Timestamp("2024-01-02"), s_ny)
    assert _tz_to_str(t.tz) == NY_TZ
    same_tz_or_raise(s_ny.index, t, allow_naive_pair=False, context="debug")
    s_naive = to_naive_local(s_ny)
    assert isinstance(s_naive.index, pd.DatetimeIndex) and s_naive.index.tz is None
    logger.info("tz_utils self-checks passed.")
