from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.runner.calendars import map_to_calendar
from backtest.utils.tz import align_ts_to_index, to_naive_local


def prev_session(calendar: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    if calendar.empty:
        return None
    t = pd.Timestamp(ts)
    try:
        t = align_ts_to_index(t, calendar)
    except Exception:
        t = pd.Timestamp(to_naive_local(t))

    if t not in calendar:
        t_m = map_to_calendar(t, calendar, policy="next")
        if t_m is None:
            return None
        t = t_m

    pos = int(calendar.get_indexer([t])[0])
    if pos <= 0:
        return None
    return pd.Timestamp(calendar[pos - 1])


def synthesize_analysis_split(
    calendar: pd.DatetimeIndex, train_start: Any
) -> dict[str, str] | None:
    if calendar.empty:
        return None
    prev = prev_session(calendar, pd.Timestamp(train_start))
    if prev is None:
        return None
    return {
        "start": str(pd.Timestamp(calendar[0]).date()),
        "end": str(pd.Timestamp(prev).date()),
    }
