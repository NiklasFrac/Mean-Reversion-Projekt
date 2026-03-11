from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from backtest.calendars import build_trading_calendar
from backtest.simulators.performance import compute_drawdowns
from backtest.utils.tz import align_ts_to_index, to_naive_local

from . import engine_state as _state
from .engine_trades import _clip_trades_to_eval_window, _normalize_trades
from .engine_tz import (
    _coerce_like_index,
    _ensure_calendar_tz,
    _to_ex_tz_series,
    _to_ex_tz_timestamp,
)

logger = logging.getLogger("backtest")


def _calendar_name_from_cfg(cfg: Any) -> str:
    raw = getattr(cfg, "raw_yaml", {}) or {}
    if isinstance(raw, Mapping):
        data = raw.get("data")
        if isinstance(data, Mapping):
            name = data.get("calendar_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return "XNYS"


def _build_calendar_and_window(
    cfg: Any,
    price_data: Mapping[str, pd.Series],
    *,
    ex_tz: str,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    calendar = build_trading_calendar(
        price_data, calendar_name=_calendar_name_from_cfg(cfg)
    )
    calendar = _ensure_calendar_tz(calendar, ex_tz)

    e0, e1 = _resolve_eval_window(calendar, cfg)
    e0 = _to_ex_tz_timestamp(e0, ex_tz, _state._NAIVE_IS_UTC)
    e1 = _to_ex_tz_timestamp(e1, ex_tz, _state._NAIVE_IS_UTC)

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
            df[col] = _coerce_like_index(df[col], calendar)

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
    cfg: Any,
    mode: str,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
) -> pd.DataFrame:
    eq = pd.Series(cfg.initial_capital, index=calendar, name="equity")
    returns = eq.pct_change().fillna(0.0)
    dd, max_dd, _, _ = compute_drawdowns(eq)
    dd_pct = (dd / eq.cummax().replace(0, np.nan)).fillna(0.0).astype(float)
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
            "exec_mode": cfg.exec_mode,
            "exec_rejected_count": 0,
        }
    )
    return stats


def _select_eval_trades(
    trades_df: pd.DataFrame, *, e0: pd.Timestamp, e1: pd.Timestamp
) -> pd.DataFrame:
    try:
        trades_df["entry_date"] = _to_ex_tz_series(
            trades_df["entry_date"], _state._EX_TZ, _state._NAIVE_IS_UTC
        )
        trades_df["exit_date"] = _to_ex_tz_series(
            trades_df["exit_date"], _state._EX_TZ, _state._NAIVE_IS_UTC
        )
    except Exception as _tz_e:
        logger.warning("TZ coercion failed (entry/exit): %s", _tz_e)
    _e0 = _to_ex_tz_timestamp(e0, _state._EX_TZ, _state._NAIVE_IS_UTC)
    _e1 = _to_ex_tz_timestamp(e1, _state._EX_TZ, _state._NAIVE_IS_UTC)

    return trades_df[
        (trades_df["exit_date"] >= _e0) & (trades_df["exit_date"] <= _e1)
    ].copy()


def _resolve_eval_window(
    calendar: pd.DatetimeIndex, cfg: Any
) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Conservative-only: windows are defined by `cfg.splits`.
    if not cfg.splits or "test" not in cfg.splits:
        raise KeyError(
            "BacktestConfig.splits['test'] missing (conservative mode required)"
        )

    e0 = pd.to_datetime(cfg.splits["test"]["start"])
    e1 = pd.to_datetime(cfg.splits["test"]["end"])

    def _chk(k: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if cfg.splits and k in cfg.splits:
            return pd.to_datetime(cfg.splits[k]["start"]), pd.to_datetime(
                cfg.splits[k]["end"]
            )
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
