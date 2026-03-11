from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from backtest.calendars import map_to_calendar
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


def walkforward_window_splits(
    calendar: pd.DatetimeIndex, window: Any
) -> dict[str, dict[str, str]]:
    analysis = synthesize_analysis_split(calendar, getattr(window, "train_start"))
    if analysis is None:
        raise ValueError(
            "Cannot synthesize analysis window (no session before range.start)."
        )
    return {"analysis": analysis, **window.as_splits()}


def slice_frame(df: pd.DataFrame, *, start: Any, end: Any) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a non-empty DatetimeIndex")

    idx = df.index
    s = align_ts_to_index(start, idx)
    e = align_ts_to_index(end, idx)
    out = df[(idx >= s) & (idx <= e)]
    if out.empty:
        raise ValueError(f"Empty slice for window {start}..{end}")
    return out


def _require_split_window(splits: Mapping[str, Any], name: str) -> dict[str, Any]:
    win = dict(
        (splits.get(name) or {}) if isinstance(splits.get(name), Mapping) else {}
    )
    if not (win.get("start") and win.get("end")):
        raise KeyError(f"Missing split window: {name}")
    return win


def slice_frame_from_named_splits(
    df: pd.DataFrame,
    *,
    splits: Mapping[str, Any],
    start_key: str,
    end_key: str,
) -> pd.DataFrame:
    start_win = _require_split_window(splits, start_key)
    end_win = _require_split_window(splits, end_key)
    return slice_frame(df, start=start_win.get("start"), end=end_win.get("end"))


def remap_cfg_for_named_eval(
    cfg: Mapping[str, Any],
    *,
    splits: Mapping[str, Any],
    eval_split: str,
    order: Mapping[str, tuple[str, str]],
    disable_walkforward: bool = False,
) -> dict[str, Any]:
    if eval_split not in order:
        raise ValueError(f"Unsupported eval_split: {eval_split!r}")

    train_key, test_key = order[eval_split]
    train_win = _require_split_window(splits, train_key)
    test_win = _require_split_window(splits, test_key)

    out = dict(cfg)
    bt = dict(out.get("backtest") or {})
    bt["splits"] = {"train": train_win, "test": test_win}
    if disable_walkforward and isinstance(bt.get("walkforward"), Mapping):
        wf = dict(bt.get("walkforward") or {})
        wf["enabled"] = False
        bt["walkforward"] = wf
    out["backtest"] = bt
    return out
