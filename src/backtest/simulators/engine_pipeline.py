from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.config.types import AppConfig
from backtest.runner.calendars import build_trading_calendar
from backtest.simulators.performance import compute_drawdowns
from backtest.utils.tz import NY_TZ, align_ts_to_index, to_naive_local

from .engine_trades import _clip_trades_to_eval_window, _normalize_trades

logger = logging.getLogger("backtest")


def _align_datetime_series_to_index(
    values: pd.Series,
    idx: pd.DatetimeIndex,
) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    aligned = [
        align_ts_to_index(ts, idx) if pd.notna(ts) else pd.NaT for ts in parsed.tolist()
    ]
    return pd.Series(aligned, index=values.index)


def _calendar_name_from_cfg(cfg: AppConfig) -> str:
    return str(cfg.data.calendar_name or "XNYS")


def _build_calendar_and_window(
    cfg: Any,
    price_data: Mapping[str, pd.Series],
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    calendar = build_trading_calendar(
        price_data, calendar_name=_calendar_name_from_cfg(cfg)
    )

    e0, e1 = _resolve_eval_window(calendar, cfg)
    e0 = align_ts_to_index(e0, calendar)
    e1 = align_ts_to_index(e1, calendar)

    calendar = calendar[(calendar >= e0) & (calendar <= e1)]
    if calendar.empty:
        raise ValueError("Calendar is empty in evaluation window")
    return calendar, e0, e1


def _collect_and_normalize_trades(
    portfolio: Mapping[str, Mapping[str, Any]],
    *,
    calendar: pd.DatetimeIndex,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
    price_data: Mapping[str, pd.Series],
) -> tuple[list[pd.DataFrame], int, int, int]:
    frames: list[pd.DataFrame] = []
    dropped_outside_eval = 0
    hard_exit_count = 0
    total_trades_seen = 0
    for pair, meta in portfolio.items():
        trades_obj = meta.get("trades") if isinstance(meta, Mapping) else None
        df = (
            _normalize_trades(str(pair), trades_obj) if trades_obj is not None else None
        )
        if df is None or df.empty:
            continue
        for col in ("entry_date", "exit_date"):
            df[col] = _align_datetime_series_to_index(df[col], calendar)

        total_trades_seen += int(len(df))
        df, rep = _clip_trades_to_eval_window(df, e0=e0, e1=e1, price_data=price_data)
        dropped_outside_eval += int(rep.get("dropped", 0) or 0)
        hard_exit_count += int(rep.get("hard_exits", 0) or 0)
        if df.empty:
            continue
        frames.append(df)

    return frames, dropped_outside_eval, hard_exit_count, total_trades_seen


def _ensure_exit_after_entry(
    trades_df: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    same_or_before = trades_df["exit_date"] <= trades_df["entry_date"]
    if same_or_before.any():
        loc = calendar.get_indexer(
            trades_df.loc[same_or_before, "entry_date"], method="bfill"
        )
        loc = np.clip(loc + 1, 0, len(calendar) - 1)
        trades_df.loc[same_or_before, "exit_date"] = calendar[loc]
    return trades_df


def _flat_equity_stats(
    calendar: pd.DatetimeIndex,
    *,
    cfg: AppConfig,
    mode: str,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
) -> pd.DataFrame:
    eq = pd.Series(cfg.backtest.initial_capital, index=calendar, name="equity")
    returns = eq.pct_change().fillna(0.0)
    dd, max_dd, _, _ = compute_drawdowns(eq)
    dd_pct = dd.astype(float)
    stats = pd.DataFrame(
        {"equity": eq, "returns": returns, "drawdown": dd, "drawdown_pct": dd_pct}
    )
    stats["Sharpe"] = 0.0
    stats["CAGR"] = 0.0
    stats["max_drawdown"] = float(max_dd or 0.0)
    stats["WinRate"] = 0.0
    stats["NumTrades"] = 0
    stats.attrs.update(
        {
            "EquityFinal": float(eq.iloc[-1]),
            "EquityRawEnd": float(eq.iloc[-1]),
            "Sharpe": 0.0,
            "CAGR": 0.0,
            "MaxDrawdown": float(max_dd or 0.0),
            "WinRate": 0.0,
            "NumTrades": 0,
            "mode": mode,
            "eval_window_start": e0.isoformat(),
            "eval_window_end": e1.isoformat(),
            "mapped_trades": 0,
            "calendar_name": _calendar_name_from_cfg(cfg),
            "calendar_source": "exchange_calendars",
            "exec_mode": cfg.execution.mode,
            "exec_rejected_count": 0,
        }
    )
    return stats


def _select_eval_trades(
    trades_df: pd.DataFrame, *, e0: pd.Timestamp, e1: pd.Timestamp
) -> pd.DataFrame:
    out = trades_df.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")
    if isinstance(out["exit_date"].dtype, pd.DatetimeTZDtype):
        ref_idx = pd.DatetimeIndex([pd.Timestamp(e0), pd.Timestamp(e1)], tz=NY_TZ)
        out["entry_date"] = _align_datetime_series_to_index(out["entry_date"], ref_idx)
        out["exit_date"] = _align_datetime_series_to_index(out["exit_date"], ref_idx)
        e0 = align_ts_to_index(e0, ref_idx)
        e1 = align_ts_to_index(e1, ref_idx)
    return out[(out["exit_date"] >= e0) & (out["exit_date"] <= e1)].copy()


def _resolve_eval_window(
    calendar: pd.DatetimeIndex, cfg: AppConfig
) -> tuple[pd.Timestamp, pd.Timestamp]:
    splits = cfg.backtest.splits
    if not splits or "test" not in splits:
        raise KeyError(
            "backtest.splits['test'] missing (conservative mode required)"
        )

    e0 = pd.to_datetime(splits["test"].start)
    e1 = pd.to_datetime(splits["test"].end)

    def _chk(k: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if splits and k in splits:
            return pd.to_datetime(splits[k].start), pd.to_datetime(splits[k].end)
        return None

    a = _chk("analysis")
    t = _chk("train")
    if a and t:
        a0, a1 = a
        t0, t1 = t
        if not (a1 < t0 and t1 < e0):
            raise ValueError(
                "splits must be disjoint & ordered: analysis < train < test"
            )

    try:
        t0 = align_ts_to_index(e0, calendar)
        t1 = align_ts_to_index(e1, calendar)
    except Exception:
        t0 = pd.Timestamp(to_naive_local(pd.Timestamp(e0)))
        t1 = pd.Timestamp(to_naive_local(pd.Timestamp(e1)))

    e0 = max(t0, calendar[0])
    e1 = min(t1, calendar[-1])
    if e0 > e1:
        raise ValueError("Eval window outside calendar")
    return e0, e1
