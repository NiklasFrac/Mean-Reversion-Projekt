from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from backtest.utils.tz import (
    align_index_to_index,
    align_ts_to_index,
    same_tz_or_raise,
    to_naive_local,
)

logger = logging.getLogger("backtest.calendars")


def _is_intraday(idx: pd.DatetimeIndex) -> bool:
    if idx.empty:
        return False
    # treat as intraday if there is any non-midnight timestamp
    try:
        return not bool(idx.is_normalized)
    except Exception:
        try:
            offs = idx - idx.normalize()
            return bool((offs != pd.Timedelta(0)).any())
        except Exception:
            return False


def _clean_dt_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.empty:
        return out
    if bool(out.isna().any()):
        out = out[~out.isna()]
    if out.empty:
        return out
    if not out.is_monotonic_increasing:
        out = out.sort_values()
    if bool(out.has_duplicates):
        out = out[~out.duplicated()]
    return out


def build_trading_calendar(
    price_data: Mapping[str, pd.Series],
    *,
    calendar_name: str | None = "XNYS",
) -> pd.DatetimeIndex:
    """
    Build a canonical time grid for evaluation.

    - Daily data: uses exchange calendar when available (pandas_market_calendars), else BDay range.
    - Intraday data: returns the union of timestamps present in `price_data`.
    """
    idxs: list[pd.DatetimeIndex] = []
    for s in (price_data or {}).values():
        if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 0:
            idx = _clean_dt_index(s.index)
            if not idx.empty:
                idxs.append(idx)
    if not idxs:
        return pd.DatetimeIndex([])

    ref_idx = idxs[0]
    for idx in idxs[1:]:
        same_tz_or_raise(
            ref_idx,
            idx,
            allow_naive_pair=True,
            context="build_trading_calendar",
        )
    is_intraday = any(_is_intraday(idx) for idx in idxs)

    if is_intraday:
        base = ref_idx
        for idx in idxs[1:]:
            base = base.union(idx)
        return _clean_dt_index(base)

    start = min(pd.Timestamp(idx.min()) for idx in idxs)
    end = max(pd.Timestamp(idx.max()) for idx in idxs)

    # Daily: attempt to align to an exchange calendar by session date.
    try:
        import pandas_market_calendars as mcal  # type: ignore[import-not-found, import-untyped]

        cal = mcal.get_calendar(str(calendar_name or "XNYS"))
        start_d = pd.Timestamp(to_naive_local(start)).date()
        end_d = pd.Timestamp(to_naive_local(end)).date()
        sched = cal.schedule(start_date=start_d, end_date=end_d)
        return align_index_to_index(pd.DatetimeIndex(sched.index), ref_idx)
    except Exception as e:
        logger.debug(
            "Calendar fallback (no pandas_market_calendars or invalid calendar): %s", e
        )

    start_naive = to_naive_local(start)
    end_naive = to_naive_local(end)
    days = pd.bdate_range(start=start_naive, end=end_naive, freq="B")
    return align_index_to_index(days, ref_idx)


def map_to_calendar(
    ts: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    policy: str = "prior",
) -> pd.Timestamp | None:
    """
    Map `ts` to a timestamp that exists in `calendar`.

    policy:
      - "prior":   last <= ts
      - "next":    first >= ts
      - "nearest": closest by absolute delta
    """
    if calendar is None or len(calendar) == 0:
        return None
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None

    try:
        t = align_ts_to_index(t, calendar)
    except Exception:
        t = pd.Timestamp(to_naive_local(t))

    if t in calendar:
        return t

    idx = calendar
    side = "right"
    pos = int(idx.searchsorted(t, side=side))
    pol = str(policy or "prior").strip().lower()

    if pol in {"prior", "previous", "left"}:
        j = pos - 1
        return idx[j] if j >= 0 else None
    if pol in {"next", "after", "right"}:
        return idx[pos] if pos < len(idx) else None
    if pol in {"nearest", "closest"}:
        left = idx[pos - 1] if pos - 1 >= 0 else None
        right = idx[pos] if pos < len(idx) else None
        if left is None:
            return right
        if right is None:
            return left
        return left if abs(t - left) <= abs(right - t) else right

    # unknown policy -> be conservative
    j = pos - 1
    return idx[j] if j >= 0 else None


def apply_settlement_lag(
    ts: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    lag_bars: int = 0,
) -> pd.Timestamp:
    """Shift forward by `lag_bars` positions in `calendar` (clamped)."""
    t = pd.Timestamp(ts)
    if calendar is None or len(calendar) == 0:
        return t
    if lag_bars <= 0:
        return t
    try:
        pos = int(calendar.get_indexer([t])[0])
    except Exception:
        pos = -1
    if pos < 0:
        mapped = map_to_calendar(t, calendar, policy="nearest")
        if mapped is None:
            return t
        pos = int(calendar.get_indexer([mapped])[0])
    pos2 = min(len(calendar) - 1, pos + int(lag_bars))
    return pd.Timestamp(calendar[pos2])


__all__ = ["build_trading_calendar", "map_to_calendar", "apply_settlement_lag"]
