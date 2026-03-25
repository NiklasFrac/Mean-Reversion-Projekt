from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .contracts import CanonicalTrade, EntryResult, ExecutionFill, ExitResult, PlannedTrade
from .costs import compute_post_lob_costs

__all__ = [
    "build_canonical_trade",
    "finalize_trade_accounting",
]


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _pick_float(mapping: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name not in mapping:
            continue
        out = _float_or_none(mapping.get(name))
        if out is not None:
            return out
    return None


def _pick_ts(mapping: Mapping[str, Any], *names: str) -> pd.Timestamp | None:
    for name in names:
        if name not in mapping:
            continue
        value = mapping.get(name)
        if value is None or pd.isna(value):
            continue
        return pd.Timestamp(value)
    return None


def _count_liquidity(fills: tuple[ExecutionFill, ...], role: str) -> int:
    return int(sum(1 for fill in fills if str(fill.liquidity) == role))


def _fill_price(fills: tuple[ExecutionFill, ...], leg: str) -> float | None:
    for fill in fills:
        if fill.leg == leg and fill.price is not None:
            return float(fill.price)
    return None


def _fill_liquidity(fills: tuple[ExecutionFill, ...], leg: str) -> str | None:
    for fill in fills:
        if fill.leg == leg:
            return str(fill.liquidity)
    return None


def _fee_cfg_for_mode(cfg_obj: Any) -> dict[str, Any]:
    mode = str(getattr(cfg_obj, "exec_mode", "lob") or "lob").strip().lower()
    if mode == "light":
        light = getattr(cfg_obj, "exec_light", {}) or {}
        fees = light.get("fees", {}) if isinstance(light, Mapping) else {}
        return {
            "post_costs": {
                "per_trade": float(fees.get("per_trade", 0.0) or 0.0),
                "taker_bps": float(fees.get("bps", 0.0) or 0.0),
                "taker_per_share": float(fees.get("per_share", 0.0) or 0.0),
                "min_fee": float(fees.get("min_fee", 0.0) or 0.0),
                "max_fee": float(fees.get("max_fee", 0.0) or 0.0),
            }
        }
    lob = getattr(cfg_obj, "exec_lob", {}) or {}
    if isinstance(lob, Mapping):
        return {"post_costs": dict(lob.get("post_costs", {}) or {})}
    return {"post_costs": {}}


def build_canonical_trade(
    *,
    plan: PlannedTrade,
    entry_result: EntryResult,
    exit_result: ExitResult,
    actual_signal_date: pd.Timestamp,
    planned_entry_date: pd.Timestamp,
    planned_exit_date: pd.Timestamp,
) -> CanonicalTrade:
    final_row = (
        exit_result.row_data
        if isinstance(exit_result.row_data, Mapping) and exit_result.row_data
        else {}
    )
    entry_row = (
        entry_result.row_data
        if isinstance(entry_result.row_data, Mapping) and entry_result.row_data
        else {}
    )

    entry_ts = _pick_ts(final_row, "entry_date") or entry_result.executed_at
    exit_ts = _pick_ts(final_row, "exit_date") or exit_result.executed_at
    py0 = _pick_float(final_row, "exec_entry_vwap_y", "entry_price_y")
    px0 = _pick_float(final_row, "exec_entry_vwap_x", "entry_price_x")
    py1 = _pick_float(final_row, "exec_exit_vwap_y", "exit_price_y")
    px1 = _pick_float(final_row, "exec_exit_vwap_x", "exit_price_x")
    if py0 is None:
        py0 = _pick_float(entry_row, "exec_entry_vwap_y", "entry_price_y")
    if px0 is None:
        px0 = _pick_float(entry_row, "exec_entry_vwap_x", "entry_price_x")
    if py0 is None:
        py0 = _fill_price(entry_result.fills, "y")
    if px0 is None:
        px0 = _fill_price(entry_result.fills, "x")
    if py1 is None:
        py1 = _fill_price(exit_result.fills, "y")
    if px1 is None:
        px1 = _fill_price(exit_result.fills, "x")

    gross_pnl = _pick_float(final_row, "gross_pnl")
    if gross_pnl is None and None not in (py0, px0, py1, px1):
        gross_pnl = float(plan.y_units) * (float(py1) - float(py0)) + float(
            plan.x_units
        ) * (float(px1) - float(px0))

    slip_total = float(_pick_float(final_row, "slippage_cost") or 0.0)
    slip_entry = _pick_float(final_row, "slippage_cost_entry")
    slip_exit = _pick_float(final_row, "slippage_cost_exit")
    if slip_entry is None and slip_exit is None:
        slip_entry = 0.5 * slip_total
        slip_exit = 0.5 * slip_total
    else:
        slip_entry = float(slip_entry or 0.0)
        slip_exit = float(slip_exit or 0.0)

    impact_total = float(_pick_float(final_row, "impact_cost") or 0.0)
    impact_entry = _pick_float(final_row, "impact_cost_entry")
    impact_exit = _pick_float(final_row, "impact_cost_exit")
    if impact_entry is None and impact_exit is None:
        impact_entry = 0.5 * impact_total
        impact_exit = 0.5 * impact_total
    else:
        impact_entry = float(impact_entry or 0.0)
        impact_exit = float(impact_exit or 0.0)

    maker_fills = _count_liquidity(entry_result.fills + exit_result.fills, "maker")
    taker_fills = _count_liquidity(entry_result.fills + exit_result.fills, "taker")
    if maker_fills == 0 and taker_fills == 0:
        liq_values = [
            final_row.get("liquidity_entry_y"),
            final_row.get("liquidity_entry_x"),
            final_row.get("liquidity_exit_y"),
            final_row.get("liquidity_exit_x"),
        ]
        maker_fills = int(sum(1 for val in liq_values if str(val).strip().lower() == "maker"))
        taker_fills = int(sum(1 for val in liq_values if str(val).strip().lower() == "taker"))

    notional_y = (
        float(plan.y_units) * float(py0)
        if py0 is not None and np.isfinite(float(py0))
        else np.nan
    )
    notional_x = (
        float(plan.x_units) * float(px0)
        if px0 is not None and np.isfinite(float(px0))
        else np.nan
    )
    gross_notional = 0.0
    if pd.notna(notional_y):
        gross_notional += abs(float(notional_y))
    if pd.notna(notional_x):
        gross_notional += abs(float(notional_x))

    payload = {
        "intent_id": str(plan.intent_id),
        "pair": str(plan.pair_key),
        "y_symbol": str(plan.y_symbol),
        "x_symbol": str(plan.x_symbol),
        "signal_date": pd.Timestamp(actual_signal_date),
        "planned_entry_date": pd.Timestamp(planned_entry_date),
        "planned_exit_date": pd.Timestamp(planned_exit_date),
        "entry_date": pd.Timestamp(entry_ts) if entry_ts is not None else pd.NaT,
        "exit_date": pd.Timestamp(exit_ts) if exit_ts is not None else pd.NaT,
        "exec_entry_status": str(
            final_row.get("exec_entry_status") or entry_result.status or "filled"
        ).strip().lower(),
        "exec_exit_status": str(
            final_row.get("exec_exit_status") or exit_result.status or "filled"
        ).strip().lower(),
        "exec_forced_exit": bool(
            final_row.get("exec_forced_exit", exit_result.forced_exit)
        ),
        "exec_reject_reason": str(
            final_row.get("exec_reject_reason")
            or exit_result.reject_reason
            or entry_result.reject_reason
            or ""
        ),
        "signal": int(plan.signal),
        "size": int(plan.size),
        "beta_entry": float(plan.beta_entry),
        "y_units": int(plan.y_units),
        "x_units": int(plan.x_units),
        "exec_entry_vwap_y": py0,
        "exec_entry_vwap_x": px0,
        "exec_exit_vwap_y": py1,
        "exec_exit_vwap_x": px1,
        "entry_price_y": py0,
        "entry_price_x": px0,
        "exit_price_y": py1,
        "exit_price_x": px1,
        "liquidity_entry_y": str(
            final_row.get("liquidity_entry_y")
            or _fill_liquidity(entry_result.fills, "y")
            or ""
        ),
        "liquidity_entry_x": str(
            final_row.get("liquidity_entry_x")
            or _fill_liquidity(entry_result.fills, "x")
            or ""
        ),
        "liquidity_exit_y": str(
            final_row.get("liquidity_exit_y")
            or _fill_liquidity(exit_result.fills, "y")
            or ""
        ),
        "liquidity_exit_x": str(
            final_row.get("liquidity_exit_x")
            or _fill_liquidity(exit_result.fills, "x")
            or ""
        ),
        "fees": 0.0,
        "fees_entry": 0.0,
        "fees_exit": 0.0,
        "slippage_cost": float(slip_total),
        "slippage_cost_entry": float(slip_entry),
        "slippage_cost_exit": float(slip_exit),
        "impact_cost": float(impact_total),
        "impact_cost_entry": float(impact_entry),
        "impact_cost_exit": float(impact_exit),
        "borrow_cost": 0.0,
        "total_costs": 0.0,
        "gross_pnl": float(gross_pnl or 0.0),
        "net_pnl": float(gross_pnl or 0.0),
        "exec_entry_delay_days": int(
            _pick_float(final_row, "exec_entry_delay_days")
            or entry_result.delay_days
            or 0
        ),
        "exec_exit_delay_days": int(
            _pick_float(final_row, "exec_exit_delay_days") or exit_result.delay_days or 0
        ),
        "maker_fills": int(maker_fills),
        "taker_fills": int(taker_fills),
        "notional_y": float(notional_y) if pd.notna(notional_y) else np.nan,
        "notional_x": float(notional_x) if pd.notna(notional_x) else np.nan,
        "gross_notional": float(gross_notional),
        "exec_mode_used": str(final_row.get("exec_mode_used") or ""),
        "lob_regime_entry": final_row.get("lob_regime_entry"),
        "lob_regime_exit": final_row.get("lob_regime_exit"),
    }
    return CanonicalTrade(payload=payload)


def finalize_trade_accounting(
    trades_df: pd.DataFrame,
    *,
    cfg_obj: Any,
) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return trades_df
    out = trades_df.copy()
    fee_frame = compute_post_lob_costs(out, _fee_cfg_for_mode(cfg_obj))
    for col in ("fees", "fees_entry", "fees_exit"):
        if col in fee_frame.columns:
            out[col] = pd.to_numeric(fee_frame[col], errors="coerce").fillna(0.0)
    for col, default in (
        ("slippage_cost", 0.0),
        ("slippage_cost_entry", 0.0),
        ("slippage_cost_exit", 0.0),
        ("impact_cost", 0.0),
        ("impact_cost_entry", 0.0),
        ("impact_cost_exit", 0.0),
        ("borrow_cost", 0.0),
        ("total_costs", 0.0),
    ):
        if col not in out.columns:
            out[col] = default
    return out
