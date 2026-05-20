from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.config import BacktestConfig


@dataclass(frozen=True)
class Window:
    i: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(index: pd.DatetimeIndex, cfg: BacktestConfig) -> list[Window]:
    idx = index.drop_duplicates().sort_values()
    start = _next(idx, pd.Timestamp(cfg.start)) if cfg.start else idx[0]
    end = _prior(idx, pd.Timestamp(cfg.end)) if cfg.end else idx[-1]
    idx = idx[(idx >= start) & (idx <= end)]
    wf = cfg.walkforward
    if wf.step_months < wf.test_months:
        raise ValueError("step_months must be >= test_months")
    if not wf.enabled:
        split = max(1, int(len(idx) * 0.7))
        return [Window(0, idx[0], idx[split - 1], idx[split], idx[-1])]

    out: list[Window] = []
    i = 0
    while True:
        train_start_target = (
            start
            if wf.train_mode == "expanding"
            else start + pd.DateOffset(months=i * wf.step_months)
        )
        train_end_target = start + pd.DateOffset(
            months=wf.train_months + i * wf.step_months
        )
        train_start = _next(idx, train_start_target)
        train_end = _prior(idx, train_end_target)
        if train_start is None or train_end is None or train_end <= train_start:
            break
        test_start = idx[idx > train_end].min()
        if pd.isna(test_start):
            break
        test_end = _prior(idx, train_end_target + pd.DateOffset(months=wf.test_months))
        if test_end is None or test_end < test_start:
            break
        out.append(Window(i, train_start, train_end, test_start, test_end))
        if test_end >= end:
            break
        i += 1
    return out


def _next(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    found = idx[idx >= ts]
    return None if found.empty else pd.Timestamp(found[0])


def _prior(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    found = idx[idx <= ts]
    return None if found.empty else pd.Timestamp(found[-1])
