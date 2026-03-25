from __future__ import annotations

import heapq
import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.config.cfg import config_to_dict
from backtest.config.types import AppConfig
from backtest.risk.policy import (
    build_risk_policy,
    cap_units_by_participation,
    cap_units_by_trade_notional,
    size_units_from_risk_budget,
)
from backtest.risk.state import PortfolioRiskState
from backtest.utils.tz import align_ts_to_index

from .contracts import ExecutionContext, IntentSimulationResult, PlannedTrade
from .engine_trades import _asof_price_for_ts
from .execution_backends import ExecutionBackend
from .trade_accounting import build_canonical_trade, finalize_trade_accounting

logger = logging.getLogger("backtest.simulators.intent_scheduler")

__all__ = [
    "portfolio_has_intents",
    "simulate_intent_portfolio",
]


def portfolio_has_intents(portfolio: Mapping[str, Mapping[str, Any]]) -> bool:
    for meta in (portfolio or {}).values():
        if isinstance(meta, Mapping) and isinstance(meta.get("intents"), pd.DataFrame):
            return True
    return False


def _state_prices(state: Mapping[str, Any]) -> pd.DataFrame:
    prices = state.get("prices")
    if isinstance(prices, pd.DataFrame) and {"y", "x"} <= set(prices.columns):
        out = prices.loc[:, ["y", "x"]].copy()
    else:
        out = pd.DataFrame(columns=["y", "x"])
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    for col in ("y", "x"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _state_series(state: Mapping[str, Any], key: str) -> pd.Series:
    prices = _state_prices(state)
    idx = pd.DatetimeIndex(prices.index)
    value = state.get(key)
    if isinstance(value, pd.Series):
        out = pd.to_numeric(value, errors="coerce").reindex(idx)
    else:
        out = pd.Series(np.nan, index=idx, dtype=float)
    out.name = key
    return out.astype(float)


def _normalize_ts(ts: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(ts, errors="coerce"))


def _calendar_pos(calendar: pd.DatetimeIndex, ts: pd.Timestamp) -> int | None:
    if calendar.empty or pd.isna(ts):
        return None
    ts_n = align_ts_to_index(pd.Timestamp(ts), calendar).normalize()
    cal_n = calendar.normalize()
    pos = int(cal_n.searchsorted(ts_n, side="left"))
    if pos >= len(calendar) or cal_n[pos] != ts_n:
        return None
    return pos


def _lagged_session(
    calendar: pd.DatetimeIndex, ts: pd.Timestamp, lag_bars: int
) -> pd.Timestamp | None:
    pos = _calendar_pos(calendar, ts)
    if pos is None:
        return None
    tgt = pos + int(max(0, lag_bars))
    if tgt < 0 or tgt >= len(calendar):
        return None
    return pd.Timestamp(calendar[tgt])


def _cooldown_signal_until(
    calendar: pd.DatetimeIndex, exit_ts: pd.Timestamp, cooldown_days: int
) -> pd.Timestamp | None:
    cool = int(max(0, cooldown_days))
    if cool <= 0:
        return None
    pos = _calendar_pos(calendar, exit_ts)
    if pos is None:
        return None
    tgt = min(len(calendar) - 1, pos + cool)
    return pd.Timestamp(calendar[tgt])


def _pair_symbols(state: Mapping[str, Any]) -> tuple[str, str]:
    y = str(state.get("y_symbol") or "Y").strip().upper() or "Y"
    x = str(state.get("x_symbol") or "X").strip().upper() or "X"
    return y, x


def _sum_adv(*values: float | None) -> float | None:
    total = 0.0
    seen = False
    for raw in values:
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if np.isfinite(val) and val > 0.0:
            total += val
            seen = True
    return float(total) if seen and total > 0.0 else None


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return float(out) if np.isfinite(out) else None


def _normalize_intents(
    portfolio: Mapping[str, Mapping[str, Any]],
    *,
    calendar: pd.DatetimeIndex,
    lag_bars: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for portfolio_key, meta in (portfolio or {}).items():
        if not isinstance(meta, Mapping):
            continue
        intents = meta.get("intents")
        state = meta.get("state")
        if not isinstance(intents, pd.DataFrame) or intents.empty:
            continue
        if not isinstance(state, Mapping):
            continue
        df = intents.copy()
        df["signal_date"] = pd.to_datetime(df.get("signal_date"), errors="coerce")
        df["signal"] = pd.to_numeric(df.get("signal", 0), errors="coerce").fillna(0)
        df = df.dropna(subset=["signal_date"])
        df = df.loc[df["signal"].astype(int).ne(0)].copy()
        if df.empty:
            continue
        df["first_exec_date"] = [
            _lagged_session(calendar, pd.Timestamp(ts), lag_bars)
            for ts in df["signal_date"].tolist()
        ]
        df = df.dropna(subset=["first_exec_date"])
        if df.empty:
            continue
        df["first_exec_date"] = pd.to_datetime(df["first_exec_date"], errors="coerce")
        df["portfolio_key"] = str(portfolio_key)
        df["pair_key"] = str(state.get("pair_key") or portfolio_key)
        df["_state"] = [state] * len(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(
        ["first_exec_date", "signal_date", "pair_key", "portfolio_key"]
    ).reset_index(drop=True)
    out.insert(0, "intent_id", [f"intent-{i:06d}" for i in range(len(out))])
    out["signal"] = out["signal"].astype(int)
    return out


def _projected_prices(
    state: Mapping[str, Any], ts: pd.Timestamp
) -> tuple[float | None, float | None]:
    prices = _state_prices(state)
    py = _asof_price_for_ts(prices.get("y"), ts)
    px = _asof_price_for_ts(prices.get("x"), ts)
    y = float(py) if py is not None and np.isfinite(float(py)) else None
    x = float(px) if px is not None and np.isfinite(float(px)) else None
    return y, x


def _beta_at(state: Mapping[str, Any], ts: pd.Timestamp) -> float | None:
    beta = _state_series(state, "beta")
    raw = beta.get(align_ts_to_index(pd.Timestamp(ts), pd.DatetimeIndex(beta.index)))
    try:
        val = float(raw)
    except Exception:
        return None
    if not np.isfinite(val) or val <= 0.0:
        return None
    return val


def _risk_geometry(
    state: Mapping[str, Any], signal_date: pd.Timestamp
) -> tuple[float | None, float | None]:
    z_ser = _state_series(state, "z")
    sigma_ser = _state_series(state, "sigma")
    aligned_signal_date = align_ts_to_index(
        pd.Timestamp(signal_date), pd.DatetimeIndex(z_ser.index)
    )
    raw_z = z_ser.get(aligned_signal_date)
    raw_sigma = sigma_ser.get(aligned_signal_date)
    try:
        z_abs = abs(float(raw_z))
        sigma = float(raw_sigma)
    except Exception:
        return None, None
    entry_z = abs(float(state.get("entry_z", 2.0) or 2.0))
    stop_z = abs(float(state.get("stop_z", 3.0) or 3.0))
    if (
        not np.isfinite(z_abs)
        or not np.isfinite(sigma)
        or sigma <= 0.0
        or z_abs < entry_z
        or z_abs >= stop_z
        or stop_z <= entry_z
    ):
        return None, None
    gap = float(stop_z - z_abs)
    if gap <= 0.0:
        return None, None
    return gap, float(gap * sigma)


def _current_capital(
    *,
    initial_capital: float,
    realized_pnl: float,
    open_rows: Mapping[str, Mapping[str, Any]],
    ts: pd.Timestamp,
    price_data: Mapping[str, pd.Series],
) -> float:
    total = float(initial_capital) + float(realized_pnl)
    for row in open_rows.values():
        y_sym = str(row.get("y_symbol") or "").upper()
        x_sym = str(row.get("x_symbol") or "").upper()
        py = _asof_price_for_ts(price_data.get(y_sym), ts) if y_sym else None
        px = _asof_price_for_ts(price_data.get(x_sym), ts) if x_sym else None
        if py is None:
            py = row.get("entry_price_y")
        if px is None:
            px = row.get("entry_price_x")
        py_val = _coerce_float_or_none(py)
        px_val = _coerce_float_or_none(px)
        entry_y = _coerce_float_or_none(row.get("entry_price_y", 0.0))
        entry_x = _coerce_float_or_none(row.get("entry_price_x", 0.0))
        y_units = _coerce_float_or_none(row.get("y_units", 0.0))
        x_units = _coerce_float_or_none(row.get("x_units", 0.0))
        if None in (py_val, px_val, entry_y, entry_x, y_units, x_units):
            continue
        total += float(y_units) * (float(py_val) - float(entry_y))
        total += float(x_units) * (float(px_val) - float(entry_x))
    return float(total)


def _scan_exit_trigger(
    state: Mapping[str, Any],
    *,
    signal: int,
    actual_entry: pd.Timestamp,
) -> pd.Timestamp:
    prices = _state_prices(state)
    idx = pd.DatetimeIndex(prices.index)
    if idx.empty:
        return pd.Timestamp(actual_entry)
    z_ser = _state_series(state, "z")
    entry_z = abs(float(state.get("entry_z", 2.0) or 2.0))
    exit_z = abs(float(state.get("exit_z", 0.5) or 0.5))
    stop_z = abs(float(state.get("stop_z", 3.0) or 3.0))
    max_hold_days = int(max(1, int(state.get("max_hold_days", 10) or 10)))
    exit_end = _normalize_ts(state.get("exit_end", idx[-1]))
    start_pos = _calendar_pos(idx, actual_entry)
    end_pos = _calendar_pos(idx, exit_end)
    if start_pos is None:
        return pd.Timestamp(actual_entry)
    if end_pos is None or end_pos < start_pos:
        end_pos = len(idx) - 1
    held = 0
    for pos in range(start_pos + 1, end_pos + 1):
        ts = pd.Timestamp(idx[pos])
        try:
            zval = float(z_ser.get(ts))
        except Exception:
            zval = float("nan")
        held += 1
        if not np.isfinite(zval):
            return ts
        if abs(zval) <= exit_z or abs(zval) >= stop_z or held >= max_hold_days:
            return ts
        if (signal > 0 and zval >= entry_z) or (signal < 0 and zval <= -entry_z):
            return ts
    return pd.Timestamp(idx[end_pos])


def simulate_intent_portfolio(
    *,
    portfolio: Mapping[str, Mapping[str, Any]],
    ctx: ExecutionContext,
    backend: ExecutionBackend,
    initial_capital: float,
) -> IntentSimulationResult:
    if not isinstance(ctx.cfg_obj, AppConfig):
        raise TypeError("ctx.cfg_obj must be an AppConfig")
    cfg_dict = config_to_dict(ctx.cfg_obj)
    bt_cfg = cast(dict[str, Any], cfg_dict["backtest"])
    risk_cfg = cast(dict[str, Any], cfg_dict["risk"])
    exec_cfg = cast(dict[str, Any], cfg_dict["execution"])
    lag_bars = int(max(0, ctx.cfg_obj.backtest.execution_lag_bars))
    intents_df = _normalize_intents(portfolio, calendar=ctx.calendar, lag_bars=lag_bars)
    if intents_df.empty:
        return IntentSimulationResult(
            trades=pd.DataFrame(),
            entry_intents=pd.DataFrame(),
            state_transitions=pd.DataFrame(),
            debug_artifacts={
                "exec_entry_blocked_count": 0,
                "exec_delayed_entry_count": 0,
                "exec_delayed_exit_count": 0,
                "exec_forced_exit_count": 0,
                "exec_regime_histogram": {"entry": {}, "exit": {}},
            },
        )

    rm: PortfolioRiskState | None = None
    if bool(ctx.cfg_obj.risk.enabled):
        rm = PortfolioRiskState(initial_capital, cfg=cast(dict[str, Any], risk_cfg))
    sizing_policy = build_risk_policy(
        risk_cfg=cast(Mapping[str, Any], risk_cfg),
        backtest_cfg=cast(Mapping[str, Any], bt_cfg),
        execution_cfg=cast(Mapping[str, Any], exec_cfg),
    ).sizing

    transitions_heap: list[tuple[pd.Timestamp, int, str, str]] = []
    pending_rows: dict[str, dict[str, Any]] = {}
    open_rows: dict[str, dict[str, Any]] = {}
    pair_busy: dict[str, str] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    realized_pnl = 0.0
    trade_rows: list[dict[str, Any]] = []
    state_transitions: list[dict[str, Any]] = []
    blocked_exec = 0
    delayed_entry = 0
    delayed_exit = 0
    forced_exit = 0
    reject_reasons: Counter[str] = Counter()
    regime_entry_counts: Counter[str] = Counter()
    regime_exit_counts: Counter[str] = Counter()

    def _flush_until(ts: pd.Timestamp) -> None:
        nonlocal realized_pnl
        while transitions_heap and transitions_heap[0][0] <= ts:
            event_ts, _prio, event_kind, intent_id = heapq.heappop(transitions_heap)
            if event_kind == "fill":
                pending = pending_rows.pop(intent_id, None)
                if pending is None:
                    continue
                if rm is not None:
                    rm.register_close_pair(
                        pending["reservation_key"],
                        pending["symbols"],
                        pending["projected_notionals"],
                    )
                    rm.register_open_pair(
                        pending["trade_key"],
                        pending["symbols"],
                        pending["actual_notionals"],
                    )
                open_rows[intent_id] = pending["trade_row"]
                pair_busy[pending["pair_key"]] = "open"
                state_transitions.append(
                    {
                        "intent_id": intent_id,
                        "pair": pending["pair_key"],
                        "event_date": pd.Timestamp(event_ts),
                        "state_from": "pending_entry",
                        "state_to": "open",
                        "reason": "entry_fill",
                    }
                )
                continue

            if event_kind == "pending_exit":
                row = open_rows.get(intent_id)
                if row is None:
                    continue
                pair_key = str(row["pair"])
                pair_busy[pair_key] = "pending_exit"
                state_transitions.append(
                    {
                        "intent_id": intent_id,
                        "pair": pair_key,
                        "event_date": pd.Timestamp(event_ts),
                        "state_from": "open",
                        "state_to": "pending_exit",
                        "reason": "exit_triggered",
                    }
                )
                continue

            row = open_rows.pop(intent_id, None)
            if row is None:
                continue
            if rm is not None:
                rm.register_close_pair(
                    row["trade_key"],
                    row["symbols"],
                    row["actual_notionals"],
                )
            realized_pnl += float(
                row.get("net_pnl_for_risk", row.get("gross_pnl", 0.0))
            )
            pair_key = str(row["pair"])
            cool_until = _cooldown_signal_until(
                ctx.calendar,
                pd.Timestamp(row["exit_date"]),
                int(row.get("cooldown_days", 0) or 0),
            )
            if cool_until is not None:
                cooldown_until[pair_key] = cool_until
            pair_busy.pop(pair_key, None)
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": pd.Timestamp(event_ts),
                    "state_from": "open",
                    "state_to": "cooldown" if cool_until is not None else "flat",
                    "reason": "trade_close",
                }
            )

    for _, raw_intent in intents_df.iterrows():
        state = cast(Mapping[str, Any], raw_intent["_state"])
        pair_key = str(raw_intent["pair_key"])
        intent_id = str(raw_intent["intent_id"])
        signal_date = pd.Timestamp(raw_intent["signal_date"])
        first_exec_date = pd.Timestamp(raw_intent["first_exec_date"])
        signal = int(raw_intent["signal"])
        z_signal = float(raw_intent.get("z_signal", np.nan))

        _flush_until(first_exec_date)

        cool_until = cooldown_until.get(pair_key)
        if cool_until is not None and signal_date <= cool_until:
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": signal_date,
                    "state_from": "cooldown",
                    "state_to": "cooldown",
                    "reason": "signal_during_cooldown",
                }
            )
            continue
        if pair_key in pair_busy:
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": signal_date,
                    "state_from": pair_busy[pair_key],
                    "state_to": pair_busy[pair_key],
                    "reason": "pair_busy",
                }
            )
            continue

        cap_now = _current_capital(
            initial_capital=float(initial_capital),
            realized_pnl=float(realized_pnl),
            open_rows=open_rows,
            ts=first_exec_date,
            price_data=ctx.price_data,
        )
        if rm is not None:
            rm.update_capital(cap_now)

        py0, px0 = _projected_prices(state, first_exec_date)
        beta_entry = _beta_at(state, first_exec_date)
        stop_gap_z, per_unit_risk = _risk_geometry(state, signal_date)
        if (
            py0 is None
            or px0 is None
            or beta_entry is None
            or stop_gap_z is None
            or per_unit_risk is None
        ):
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": first_exec_date,
                    "state_from": "flat",
                    "state_to": "flat",
                    "reason": "invalid_entry_geometry",
                }
            )
            continue

        gross_entry = float(abs(py0) + abs(beta_entry) * abs(px0))
        size = size_units_from_risk_budget(
            capital=float(cap_now),
            risk_per_trade=float(sizing_policy.risk_per_trade),
            per_unit_risk=float(per_unit_risk),
            min_units_if_positive=False,
        )
        size = cap_units_by_trade_notional(
            units=int(size),
            capital=float(cap_now),
            max_trade_pct=float(sizing_policy.max_trade_pct),
            per_unit_notional=float(gross_entry),
            min_units_if_positive=False,
        )
        adv_sum_entry = _sum_adv(
            cast(float | None, state.get("adv_t1")),
            cast(float | None, state.get("adv_t2")),
        )
        if adv_sum_entry is not None and float(sizing_policy.max_participation) > 0.0:
            size = cap_units_by_participation(
                units=int(size),
                max_participation=float(sizing_policy.max_participation),
                adv_sum_usd=float(adv_sum_entry),
                per_unit_notional=float(gross_entry),
                require_gt_one_capacity=False,
                min_units_if_positive=False,
            )
        if size <= 0:
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": first_exec_date,
                    "state_from": "flat",
                    "state_to": "flat",
                    "reason": "size_zero",
                }
            )
            continue

        symbols = _pair_symbols(state)
        y_units = int(size) * (1 if signal > 0 else -1)
        x_units = int(round(float(-signal) * float(size) * float(beta_entry)))
        projected_notionals = (
            float(y_units) * float(py0),
            float(x_units) * float(px0),
        )
        reservation_key = f"{pair_key}#reservation#{intent_id}"
        trade_key = f"{pair_key}#trade#{intent_id}"
        if rm is not None:
            short_reason = rm.short_availability_pair_reason(
                leg_symbols=symbols,
                leg_notionals=projected_notionals,
                leg_units=(float(y_units), float(x_units)),
                leg_entry_prices=(float(py0), float(px0)),
                leg_adv_usd=(
                    cast(float | None, state.get("adv_t1")),
                    cast(float | None, state.get("adv_t2")),
                ),
                block_on_missing=True,
            )
            if short_reason:
                state_transitions.append(
                    {
                        "intent_id": intent_id,
                        "pair": pair_key,
                        "event_date": first_exec_date,
                        "state_from": "flat",
                        "state_to": "flat",
                        "reason": short_reason,
                    }
                )
                continue
            if not rm.can_open_pair(reservation_key, symbols, projected_notionals):
                state_transitions.append(
                    {
                        "intent_id": intent_id,
                        "pair": pair_key,
                        "event_date": first_exec_date,
                        "state_from": "flat",
                        "state_to": "flat",
                        "reason": "portfolio_blocked",
                    }
                )
                continue
            rm.register_open_pair(reservation_key, symbols, projected_notionals)

        planned = PlannedTrade(
            intent_id=intent_id,
            pair_key=pair_key,
            signal_date=signal_date,
            planned_entry_date=first_exec_date,
            planned_exit_date=_normalize_ts(state.get("exit_end", ctx.calendar[-1])),
            signal=signal,
            size=int(size),
            beta_entry=float(beta_entry),
            y_symbol=symbols[0],
            x_symbol=symbols[1],
            y_units=int(y_units),
            x_units=int(x_units),
            metadata={"z_signal": float(z_signal)},
        )

        pair_busy[pair_key] = "pending_entry"
        state_transitions.append(
            {
                "intent_id": intent_id,
                "pair": pair_key,
                "event_date": first_exec_date,
                "state_from": "flat",
                "state_to": "pending_entry",
                "reason": "intent_admitted",
            }
        )

        entry_result = backend.resolve_entry(planned, ctx)
        if entry_result.executed_at is None or entry_result.status in {"blocked", "rejected"}:
            blocked_exec += 1
            reason = str(entry_result.reject_reason or "entry_execution_blocked")
            if reason:
                reject_reasons[reason] += 1
            if rm is not None:
                rm.register_close_pair(reservation_key, symbols, projected_notionals)
            pair_busy.pop(pair_key, None)
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": first_exec_date,
                    "state_from": "pending_entry",
                    "state_to": "flat",
                    "reason": reason,
                }
            )
            continue

        planned_exit = _scan_exit_trigger(
            state,
            signal=signal,
            actual_entry=pd.Timestamp(entry_result.executed_at),
        )
        planned_exit = align_ts_to_index(pd.Timestamp(planned_exit), ctx.calendar)
        exit_result = backend.resolve_exit(
            planned,
            entry_result,
            planned_exit,
            ctx,
        )
        if exit_result.executed_at is None or exit_result.status in {"blocked", "rejected"}:
            blocked_exec += 1
            reason = str(exit_result.reject_reason or "exit_execution_blocked")
            if reason:
                reject_reasons[reason] += 1
            if rm is not None:
                rm.register_close_pair(reservation_key, symbols, projected_notionals)
            pair_busy.pop(pair_key, None)
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": first_exec_date,
                    "state_from": "pending_entry",
                    "state_to": "flat",
                    "reason": reason,
                }
            )
            continue

        canonical = build_canonical_trade(
            plan=planned,
            entry_result=entry_result,
            exit_result=exit_result,
            actual_signal_date=signal_date,
            planned_entry_date=first_exec_date,
            planned_exit_date=planned_exit,
        )
        trade_row = canonical.as_dict()
        trade_frame = finalize_trade_accounting(
            pd.DataFrame([trade_row]),
            cfg_obj=ctx.cfg_obj,
        )
        trade_row = trade_frame.iloc[0].to_dict()
        trade_row["decision_date"] = pd.Timestamp(signal_date)
        trade_row["z_signal"] = float(z_signal)
        trade_row["entry_capital_base"] = float(cap_now)
        trade_row["adv_sum_entry"] = (
            float(adv_sum_entry) if adv_sum_entry is not None else np.nan
        )
        trade_row["cooldown_days"] = int(state.get("cooldown_days", 0) or 0)
        trade_row["net_pnl_for_risk"] = float(
            trade_row.get("gross_pnl", 0.0)
        ) + float(trade_row.get("fees", 0.0)) + float(
            trade_row.get("slippage_cost", 0.0)
        ) + float(
            trade_row.get("impact_cost", 0.0)
        )
        trade_row["symbols"] = symbols
        trade_row["actual_notionals"] = (
            float(trade_row.get("notional_y", 0.0)),
            float(trade_row.get("notional_x", 0.0)),
        )
        trade_row["reservation_key"] = reservation_key
        trade_row["trade_key"] = trade_key

        actual_entry = align_ts_to_index(
            pd.Timestamp(trade_row["entry_date"]), ctx.calendar
        )
        actual_exit = align_ts_to_index(
            pd.Timestamp(trade_row["exit_date"]), ctx.calendar
        )
        trade_row["entry_date"] = actual_entry
        trade_row["exit_date"] = actual_exit

        if str(trade_row.get("exec_entry_status", "")).lower() == "delayed":
            delayed_entry += 1
        if str(trade_row.get("exec_exit_status", "")).lower() == "delayed":
            delayed_exit += 1
        if bool(trade_row.get("exec_forced_exit", False)):
            forced_exit += 1

        exec_hist = exit_result.diagnostics.get("exec_regime_histogram", {})
        if isinstance(exec_hist, Mapping):
            for key, bucket in (("entry", regime_entry_counts), ("exit", regime_exit_counts)):
                side = exec_hist.get(key, {})
                if isinstance(side, Mapping):
                    for name, count in side.items():
                        if str(name):
                            bucket[str(name)] += int(count)

        if actual_entry > first_exec_date:
            pending_rows[intent_id] = {
                "pair_key": pair_key,
                "symbols": symbols,
                "projected_notionals": projected_notionals,
                "actual_notionals": trade_row["actual_notionals"],
                "reservation_key": reservation_key,
                "trade_key": trade_key,
                "trade_row": trade_row,
            }
            heapq.heappush(
                transitions_heap,
                (pd.Timestamp(actual_entry), 0, "fill", intent_id),
            )
        else:
            if rm is not None:
                rm.register_close_pair(reservation_key, symbols, projected_notionals)
                rm.register_open_pair(trade_key, symbols, trade_row["actual_notionals"])
            open_rows[intent_id] = trade_row
            pair_busy[pair_key] = "open"
            state_transitions.append(
                {
                    "intent_id": intent_id,
                    "pair": pair_key,
                    "event_date": actual_entry,
                    "state_from": "pending_entry",
                    "state_to": "open",
                    "reason": "entry_fill",
                }
            )

        if actual_exit > planned_exit:
            heapq.heappush(
                transitions_heap,
                (pd.Timestamp(planned_exit), 1, "pending_exit", intent_id),
            )
        heapq.heappush(
            transitions_heap,
            (
                pd.Timestamp(actual_exit),
                2 if actual_exit > planned_exit else 1,
                "close",
                intent_id,
            ),
        )
        trade_rows.append(trade_row)

    if transitions_heap and len(ctx.calendar):
        _flush_until(pd.Timestamp(ctx.calendar[-1]))

    out = pd.DataFrame(trade_rows)
    for col in ("symbols", "actual_notionals", "reservation_key", "trade_key"):
        if col in out.columns:
            out = out.drop(columns=[col])

    return IntentSimulationResult(
        trades=out,
        entry_intents=intents_df.drop(columns=["_state"]).copy(),
        state_transitions=pd.DataFrame(state_transitions),
        debug_artifacts={
            "exec_entry_blocked_count": int(blocked_exec),
            "exec_delayed_entry_count": int(delayed_entry),
            "exec_delayed_exit_count": int(delayed_exit),
            "exec_forced_exit_count": int(forced_exit),
            "exec_reject_reasons": {
                str(k): int(v) for k, v in reject_reasons.items() if str(k)
            },
            "exec_regime_histogram": {
                "entry": {str(k): int(v) for k, v in regime_entry_counts.items()},
                "exit": {str(k): int(v) for k, v in regime_exit_counts.items()},
            },
        },
    )
