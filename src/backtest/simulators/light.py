from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = ["annotate_with_light"]


def _resolve_exec_light_cfg(cfg_obj: Any) -> dict[str, Any]:
    direct = getattr(cfg_obj, "exec_light", None)
    if isinstance(direct, Mapping):
        return dict(cast(Mapping[str, Any], direct))

    raw_yaml = getattr(cfg_obj, "raw_yaml", {}) or {}
    if isinstance(raw_yaml, Mapping):
        ex = raw_yaml.get("execution")
        if isinstance(ex, Mapping) and isinstance(ex.get("light"), Mapping):
            return dict(cast(Mapping[str, Any], ex.get("light")))
        if isinstance(raw_yaml.get("light"), Mapping):
            return dict(cast(Mapping[str, Any], raw_yaml.get("light")))
    return {}


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _pick_float(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row.index or pd.isna(row[name]):
            continue
        try:
            value = float(row[name])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def _leg_present(row: pd.Series, leg: str) -> bool:
    cols = {
        "y": (
            "y_symbol",
            "y_units",
            "units_y",
            "notional_y",
            "entry_price_y",
            "exit_price_y",
            "price_y",
        ),
        "x": (
            "x_symbol",
            "x_units",
            "units_x",
            "notional_x",
            "entry_price_x",
            "exit_price_x",
            "price_x",
        ),
    }
    for col in cols[leg]:
        if col not in row.index or pd.isna(row[col]):
            continue
        if col.endswith("_symbol"):
            if str(row[col]).strip():
                return True
        else:
            return True
    return False


def _infer_units(row: pd.Series, leg: str, entry_px: float | None) -> float:
    direct_names = ("y_units", "units_y") if leg == "y" else ("x_units", "units_x")
    direct = _pick_float(row, direct_names)
    if direct is not None:
        return float(direct)

    notional_name = "notional_y" if leg == "y" else "notional_x"
    notional = _pick_float(row, (notional_name,))
    if (
        notional is not None
        and entry_px is not None
        and np.isfinite(entry_px)
        and entry_px != 0.0
    ):
        return float(notional) / float(entry_px)

    size = _pick_float(row, ("size", "qty", "quantity", "units"))
    if size is None:
        return 0.0
    signal = _pick_float(row, ("signal",))
    signal_f = 1.0 if signal is None or float(signal) >= 0 else -1.0
    beta = _pick_float(row, ("beta_entry", "beta"))
    beta_f = 1.0 if beta is None else float(beta)
    if leg == "y":
        return abs(float(size)) * (1.0 if signal_f >= 0 else -1.0)
    return abs(float(size)) * (-1.0 if signal_f >= 0 else 1.0) * float(beta_f)


def _event_notional(
    row: pd.Series, *, units: float, px: float | None, fallback_field: str
) -> float:
    if px is not None and np.isfinite(px) and units != 0.0:
        return abs(float(units) * float(px))
    fallback = _pick_float(row, (fallback_field,))
    if fallback is not None:
        return abs(float(fallback))
    return 0.0


def _event_fee(
    *,
    y_notional: float,
    x_notional: float,
    y_units: float,
    x_units: float,
    per_trade: float,
    bps: float,
    per_share: float,
) -> float:
    fee = 0.0
    fee -= abs(float(y_notional)) * float(bps) * 1e-4
    fee -= abs(float(x_notional)) * float(bps) * 1e-4
    fee -= abs(float(y_units)) * float(per_share)
    fee -= abs(float(x_units)) * float(per_share)
    fee -= abs(float(per_trade)) * 2.0
    return float(fee)


def _apply_trade_fee_bounds(total_fee: float, min_fee: float, max_fee: float) -> float:
    out = float(total_fee)
    if min_fee > 0.0 and out < 0.0:
        out = min(out, -abs(float(min_fee)))
    if max_fee > 0.0 and out < 0.0:
        out = max(out, -abs(float(max_fee)))
    return float(out)


def annotate_with_light(
    trades_df: pd.DataFrame,
    _price_data: Mapping[str, pd.Series] | None,
    cfg_obj: Any,
    *,
    market_data_panel: pd.DataFrame | None = None,
    adv_map: Mapping[str, float] | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    del market_data_panel, adv_map, calendar

    if trades_df is None:
        return pd.DataFrame()
    if trades_df.empty:
        out_empty = trades_df.copy()
        for col, default in (
            ("fees", 0.0),
            ("fees_entry", 0.0),
            ("fees_exit", 0.0),
            ("slippage_cost", 0.0),
            ("slippage_cost_entry", 0.0),
            ("slippage_cost_exit", 0.0),
            ("impact_cost", 0.0),
            ("impact_cost_entry", 0.0),
            ("impact_cost_exit", 0.0),
            ("exec_rejected", False),
            ("exec_reject_reason", ""),
            ("exec_mode_used", "light"),
        ):
            out_empty[col] = default
        return out_empty

    light_cfg = _resolve_exec_light_cfg(cfg_obj)
    reject_on_missing_price = bool(light_cfg.get("reject_on_missing_price", True))
    fees_cfg: Mapping[str, Any] = (
        cast(Mapping[str, Any], light_cfg.get("fees"))
        if isinstance(light_cfg.get("fees"), Mapping)
        else {}
    )

    per_trade = float(fees_cfg.get("per_trade", 0.0) or 0.0)
    bps = float(fees_cfg.get("bps", 0.0) or 0.0)
    per_share = float(fees_cfg.get("per_share", 0.0) or 0.0)
    min_fee = float(fees_cfg.get("min_fee", 0.0) or 0.0)
    max_fee = float(fees_cfg.get("max_fee", 0.0) or 0.0)

    out = trades_df.copy()
    out["exec_entry_vwap_y"] = _num_series(out, "entry_price_y")
    out["exec_entry_vwap_x"] = _num_series(out, "entry_price_x")
    out["exec_exit_vwap_y"] = _num_series(out, "exit_price_y")
    out["exec_exit_vwap_x"] = _num_series(out, "exit_price_x")

    rejected: list[bool] = []
    reject_reasons: list[str] = []
    fees_total: list[float] = []
    fees_entry: list[float] = []
    fees_exit: list[float] = []

    for _, row in out.iterrows():
        needs_y = _leg_present(row, "y")
        needs_x = _leg_present(row, "x")

        py0 = _pick_float(row, ("exec_entry_vwap_y", "entry_price_y"))
        px0 = _pick_float(row, ("exec_entry_vwap_x", "entry_price_x"))
        py1 = _pick_float(row, ("exec_exit_vwap_y", "exit_price_y"))
        px1 = _pick_float(row, ("exec_exit_vwap_x", "exit_price_x"))

        missing_entry = (needs_y and py0 is None) or (needs_x and px0 is None)
        missing_exit = (needs_y and py1 is None) or (needs_x and px1 is None)
        if reject_on_missing_price and (missing_entry or missing_exit):
            rejected.append(True)
            if missing_entry and missing_exit:
                reject_reasons.append("missing_entry_exit_price")
            elif missing_entry:
                reject_reasons.append("missing_entry_price")
            else:
                reject_reasons.append("missing_exit_price")
            fees_total.append(0.0)
            fees_entry.append(0.0)
            fees_exit.append(0.0)
            continue

        uy = _infer_units(row, "y", py0)
        ux = _infer_units(row, "x", px0)

        ny0 = _event_notional(row, units=uy, px=py0, fallback_field="notional_y")
        nx0 = _event_notional(row, units=ux, px=px0, fallback_field="notional_x")
        ny1 = _event_notional(row, units=uy, px=py1, fallback_field="notional_y")
        nx1 = _event_notional(row, units=ux, px=px1, fallback_field="notional_x")

        f_entry = _event_fee(
            y_notional=ny0,
            x_notional=nx0,
            y_units=uy,
            x_units=ux,
            per_trade=per_trade,
            bps=bps,
            per_share=per_share,
        )
        f_exit = _event_fee(
            y_notional=ny1,
            x_notional=nx1,
            y_units=uy,
            x_units=ux,
            per_trade=per_trade,
            bps=bps,
            per_share=per_share,
        )
        f_total = _apply_trade_fee_bounds(
            float(f_entry + f_exit), min_fee=min_fee, max_fee=max_fee
        )

        if (f_entry + f_exit) != 0.0:
            scale = float(f_total) / float(f_entry + f_exit)
            f_entry *= scale
            f_exit *= scale
        else:
            f_entry = 0.0
            f_exit = 0.0

        rejected.append(False)
        reject_reasons.append("")
        fees_total.append(float(f_total))
        fees_entry.append(float(f_entry))
        fees_exit.append(float(f_exit))

    out["fees"] = pd.Series(fees_total, index=out.index, dtype=float)
    out["fees_entry"] = pd.Series(fees_entry, index=out.index, dtype=float)
    out["fees_exit"] = pd.Series(fees_exit, index=out.index, dtype=float)
    out["slippage_cost"] = 0.0
    out["slippage_cost_entry"] = 0.0
    out["slippage_cost_exit"] = 0.0
    out["impact_cost"] = 0.0
    out["impact_cost_entry"] = 0.0
    out["impact_cost_exit"] = 0.0
    out["exec_rejected"] = pd.Series(rejected, index=out.index, dtype=bool)
    out["exec_reject_reason"] = pd.Series(reject_reasons, index=out.index, dtype=str)
    out["exec_mode_used"] = "light"
    return out
