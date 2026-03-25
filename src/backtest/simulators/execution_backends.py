from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.config.types import AppConfig
from .contracts import (
    EntryResult,
    ExecutionContext,
    ExecutionFill,
    ExitResult,
    PlannedTrade,
)
from .engine_trades import _asof_price_for_ts
from .lob import annotate_with_lob

__all__ = [
    "ExecutionBackend",
    "LightExecutionBackend",
    "LobExecutionBackend",
    "make_execution_backend",
]


def _trade_row(
    plan: PlannedTrade,
    *,
    planned_entry_date: pd.Timestamp | None = None,
    planned_exit_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "intent_id": str(plan.intent_id),
                "pair": str(plan.pair_key),
                "y_symbol": str(plan.y_symbol),
                "x_symbol": str(plan.x_symbol),
                "signal": int(plan.signal),
                "size": int(plan.size),
                "beta_entry": float(plan.beta_entry),
                "y_units": int(plan.y_units),
                "x_units": int(plan.x_units),
                "entry_date": pd.Timestamp(
                    planned_entry_date
                    if planned_entry_date is not None
                    else plan.planned_entry_date
                ),
                "exit_date": pd.Timestamp(
                    planned_exit_date
                    if planned_exit_date is not None
                    else plan.planned_exit_date
                ),
            }
        ]
    )


def _side_for_event(units: float, event: str) -> str:
    if event == "entry":
        return "buy" if float(units) > 0 else "sell"
    return "sell" if float(units) > 0 else "buy"


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _int_or_zero(value: Any) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _delay_days(
    planned_ts: pd.Timestamp | None,
    actual_ts: pd.Timestamp | None,
) -> int:
    if planned_ts is None or actual_ts is None:
        return 0
    try:
        return int(max(0, (pd.Timestamp(actual_ts).normalize() - pd.Timestamp(planned_ts).normalize()).days))
    except Exception:
        return 0


def _fill_from_row(row: Mapping[str, Any], *, leg: str, event: str) -> ExecutionFill | None:
    symbol = str(row.get(f"{leg}_symbol") or "").strip().upper()
    if not symbol:
        return None
    units = _float_or_none(row.get(f"{leg}_units"))
    if units is None or units == 0.0:
        return None
    ts_key = "entry_date" if event == "entry" else "exit_date"
    px_key = f"exec_{event}_vwap_{leg}"
    liquidity_key = f"liquidity_{event}_{leg}"
    ts = row.get(ts_key)
    price = _float_or_none(row.get(px_key))
    if price is None:
        price = _float_or_none(
            row.get(f"{'entry' if event == 'entry' else 'exit'}_price_{leg}")
        )
    liquidity = str(row.get(liquidity_key) or row.get(f"liquidity_{leg}") or "taker")
    return ExecutionFill(
        leg=leg,
        event=event,
        symbol=symbol,
        ts=pd.Timestamp(ts) if ts is not None and not pd.isna(ts) else None,
        price=price,
        units=float(units),
        side=_side_for_event(float(units), event),
        liquidity=str(liquidity).strip().lower() or "taker",
    )


def _fills_from_row(row: Mapping[str, Any], event: str) -> tuple[ExecutionFill, ...]:
    out: list[ExecutionFill] = []
    for leg in ("y", "x"):
        fill = _fill_from_row(row, leg=leg, event=event)
        if fill is not None:
            out.append(fill)
    return tuple(out)


class ExecutionBackend:
    def name(self) -> str:
        raise NotImplementedError

    def resolve_entry(
        self,
        plan: PlannedTrade,
        ctx: ExecutionContext,
    ) -> EntryResult:
        raise NotImplementedError

    def resolve_exit(
        self,
        open_trade: PlannedTrade,
        entry_result: EntryResult,
        exit_plan: pd.Timestamp,
        ctx: ExecutionContext,
    ) -> ExitResult:
        raise NotImplementedError


@dataclass(frozen=True)
class LightExecutionBackend(ExecutionBackend):
    def name(self) -> str:
        return "light"

    def _reject_on_missing_price(self, cfg_obj: Any) -> bool:
        if not isinstance(cfg_obj, AppConfig):
            raise TypeError("cfg_obj must be an AppConfig")
        return bool(cfg_obj.execution.light.reject_on_missing_price)

    def resolve_entry(
        self,
        plan: PlannedTrade,
        ctx: ExecutionContext,
    ) -> EntryResult:
        py = _asof_price_for_ts(ctx.price_data.get(plan.y_symbol), plan.planned_entry_date)
        px = _asof_price_for_ts(ctx.price_data.get(plan.x_symbol), plan.planned_entry_date)
        if py is None or px is None:
            reason = "missing_entry_price"
            status = "rejected" if self._reject_on_missing_price(ctx.cfg_obj) else "blocked"
            return EntryResult(status=status, executed_at=None, reject_reason=reason)
        row = {
            "pair": plan.pair_key,
            "y_symbol": plan.y_symbol,
            "x_symbol": plan.x_symbol,
            "y_units": plan.y_units,
            "x_units": plan.x_units,
            "entry_date": pd.Timestamp(plan.planned_entry_date),
            "exec_entry_vwap_y": float(py),
            "exec_entry_vwap_x": float(px),
            "liquidity_entry_y": "taker",
            "liquidity_entry_x": "taker",
        }
        return EntryResult(
            status="filled",
            executed_at=pd.Timestamp(plan.planned_entry_date),
            delay_days=0,
            fills=_fills_from_row(row, "entry"),
            row_data=row,
        )

    def resolve_exit(
        self,
        open_trade: PlannedTrade,
        entry_result: EntryResult,
        exit_plan: pd.Timestamp,
        ctx: ExecutionContext,
    ) -> ExitResult:
        py = _asof_price_for_ts(ctx.price_data.get(open_trade.y_symbol), exit_plan)
        px = _asof_price_for_ts(ctx.price_data.get(open_trade.x_symbol), exit_plan)
        if py is None or px is None:
            reason = "missing_exit_price"
            status = "rejected" if self._reject_on_missing_price(ctx.cfg_obj) else "blocked"
            return ExitResult(status=status, executed_at=None, reject_reason=reason)
        row = {
            "pair": open_trade.pair_key,
            "y_symbol": open_trade.y_symbol,
            "x_symbol": open_trade.x_symbol,
            "y_units": open_trade.y_units,
            "x_units": open_trade.x_units,
            "entry_date": entry_result.executed_at,
            "exit_date": pd.Timestamp(exit_plan),
            "exec_entry_vwap_y": _float_or_none(entry_result.row_data.get("exec_entry_vwap_y")),
            "exec_entry_vwap_x": _float_or_none(entry_result.row_data.get("exec_entry_vwap_x")),
            "exec_exit_vwap_y": float(py),
            "exec_exit_vwap_x": float(px),
            "liquidity_entry_y": str(entry_result.row_data.get("liquidity_entry_y") or "taker"),
            "liquidity_entry_x": str(entry_result.row_data.get("liquidity_entry_x") or "taker"),
            "liquidity_exit_y": "taker",
            "liquidity_exit_x": "taker",
            "exec_entry_status": str(entry_result.status),
            "exec_exit_status": "filled",
            "exec_entry_delay_days": int(entry_result.delay_days),
            "exec_exit_delay_days": 0,
            "exec_forced_exit": False,
            "exec_mode_used": "light",
        }
        return ExitResult(
            status="filled",
            executed_at=pd.Timestamp(exit_plan),
            delay_days=0,
            forced_exit=False,
            fills=_fills_from_row(row, "exit"),
            row_data=row,
        )


@dataclass(frozen=True)
class LobExecutionBackend(ExecutionBackend):
    def name(self) -> str:
        return "lob"

    def resolve_entry(
        self,
        plan: PlannedTrade,
        ctx: ExecutionContext,
    ) -> EntryResult:
        probe = annotate_with_lob(
            _trade_row(plan),
            ctx.price_data,
            ctx.cfg_obj,
            market_data_panel=ctx.market_data_panel,
            adv_map=ctx.adv_map,
            calendar=ctx.calendar,
        )
        if probe.empty:
            return EntryResult(status="blocked", executed_at=None, reject_reason="entry_execution_blocked")
        row = probe.iloc[0].to_dict()
        entry_status = str(row.get("exec_entry_status") or "blocked").strip().lower()
        if bool(row.get("exec_rejected", False)):
            return EntryResult(
                status="rejected",
                executed_at=None,
                reject_reason=str(row.get("exec_reject_reason") or "entry_execution_failed"),
                diagnostics=dict(probe.attrs),
                row_data=row,
            )
        executed_at = row.get("entry_date")
        if entry_status == "blocked" or executed_at is None or pd.isna(executed_at):
            return EntryResult(
                status="blocked",
                executed_at=None,
                reject_reason=str(row.get("exec_reject_reason") or "entry_execution_blocked"),
                diagnostics=dict(probe.attrs),
                row_data=row,
            )
        entry_ts = pd.Timestamp(executed_at)
        delay_days = _int_or_zero(row.get("exec_entry_delay_days"))
        if delay_days <= 0:
            delay_days = _delay_days(plan.planned_entry_date, entry_ts)
        return EntryResult(
            status=entry_status or "filled",
            executed_at=entry_ts,
            delay_days=delay_days,
            fills=_fills_from_row(row, "entry"),
            diagnostics=dict(probe.attrs),
            row_data=row,
        )

    def resolve_exit(
        self,
        open_trade: PlannedTrade,
        entry_result: EntryResult,
        exit_plan: pd.Timestamp,
        ctx: ExecutionContext,
    ) -> ExitResult:
        final = annotate_with_lob(
            _trade_row(open_trade, planned_exit_date=exit_plan),
            ctx.price_data,
            ctx.cfg_obj,
            market_data_panel=ctx.market_data_panel,
            adv_map=ctx.adv_map,
            calendar=ctx.calendar,
        )
        if final.empty:
            return ExitResult(status="blocked", executed_at=None, reject_reason="exit_execution_blocked")
        row = final.iloc[0].to_dict()
        if bool(row.get("exec_rejected", False)):
            return ExitResult(
                status="rejected",
                executed_at=None,
                reject_reason=str(row.get("exec_reject_reason") or "exit_execution_failed"),
                diagnostics=dict(final.attrs),
                row_data=row,
            )
        executed_at = row.get("exit_date")
        if executed_at is None or pd.isna(executed_at):
            return ExitResult(
                status="blocked",
                executed_at=None,
                reject_reason=str(row.get("exec_reject_reason") or "exit_execution_blocked"),
                diagnostics=dict(final.attrs),
                row_data=row,
            )
        exit_ts = pd.Timestamp(executed_at)
        exit_status = str(row.get("exec_exit_status") or "filled").strip().lower()
        delay_days = _int_or_zero(row.get("exec_exit_delay_days"))
        if delay_days <= 0:
            delay_days = _delay_days(exit_plan, exit_ts)
        return ExitResult(
            status=exit_status or "filled",
            executed_at=exit_ts,
            delay_days=delay_days,
            forced_exit=bool(row.get("exec_forced_exit", False)),
            fills=_fills_from_row(row, "exit"),
            diagnostics=dict(final.attrs),
            row_data=row,
        )


def make_execution_backend(cfg_obj: Any) -> ExecutionBackend:
    if not isinstance(cfg_obj, AppConfig):
        raise TypeError("cfg_obj must be an AppConfig")
    mode = str(cfg_obj.execution.mode).strip().lower()
    if mode == "light":
        return LightExecutionBackend()
    return LobExecutionBackend()
