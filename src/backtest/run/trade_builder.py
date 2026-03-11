from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.risk_policy import (
    build_risk_policy,
    cap_units_by_participation,
    cap_units_by_trade_notional,
    size_units_from_risk_budget,
)
from backtest.utils.portfolio import _df_trades_to_orders_df


def _coerce_price(series: pd.Series | None, ts: pd.Timestamp) -> float | None:
    if series is None or not isinstance(series, pd.Series):
        return None
    try:
        v = series.loc[ts]
    except Exception:
        try:
            pos = series.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                return None
            v = series.iloc[int(pos)]
        except Exception:
            return None
    try:
        px = float(v)
    except Exception:
        return None
    return px if np.isfinite(px) and px > 0.0 else None


def _infer_pair_symbols(cfg: dict[str, Any], prices: pd.DataFrame) -> tuple[str, str]:
    y_sym = str(cfg.get("_t1_symbol") or "Y").strip().upper() or "Y"
    x_sym = str(cfg.get("_t2_symbol") or "X").strip().upper() or "X"
    if y_sym != "Y" or x_sym != "X":
        return y_sym, x_sym
    if {"y_symbol", "x_symbol"} <= set(prices.columns):
        try:
            y_val = str(prices["y_symbol"].dropna().iloc[0]).strip().upper()
            x_val = str(prices["x_symbol"].dropna().iloc[0]).strip().upper()
            if y_val and x_val:
                return y_val, x_val
        except Exception:
            pass
    return y_sym, x_sym


def _coerce_numeric_series(
    value: Any, index: pd.DatetimeIndex, *, name: str
) -> pd.Series:
    if isinstance(value, pd.Series):
        s = pd.to_numeric(value, errors="coerce").reindex(index)
    else:
        s = pd.Series(np.nan, index=index, dtype=float)
    s.name = name
    return s.astype(float)


def _lag_positions_for_execution(signals: pd.Series, *, periods: int) -> pd.Series:
    s = pd.to_numeric(signals, errors="coerce").fillna(0.0)
    lag = int(max(0, periods))
    if lag <= 0:
        return s.astype(int).rename(getattr(signals, "name", None))
    out = s.shift(lag).fillna(0.0).astype(int)
    out.name = getattr(signals, "name", None)
    return out


def _safe_index_pos(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> int | None:
    try:
        loc = idx.get_loc(ts)
    except Exception:
        pos = int(idx.searchsorted(ts, side="left"))
        if pos >= len(idx) or idx[pos] != ts:
            return None
        return pos
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, np.ndarray):
        if loc.dtype == bool:
            hits = np.flatnonzero(loc)
            return int(hits[0]) if hits.size else None
        return int(loc[0]) if loc.size else None
    return int(loc)


def _sum_adv(*values: float | None) -> float | None:
    total = 0.0
    seen = False
    for raw in values:
        if raw is None:
            continue
        try:
            v = float(raw)
        except Exception:
            continue
        if np.isfinite(v) and v > 0.0:
            total += v
            seen = True
    return float(total) if seen and total > 0.0 else None


def _per_unit_stop_risk(
    *,
    decision_ts: pd.Timestamp,
    sigma_cache: pd.Series,
    stop_gap_z: float,
) -> float | None:
    stop_gap_z = float(stop_gap_z)
    if not np.isfinite(stop_gap_z) or stop_gap_z <= 0.0:
        return None

    spread_sigma_raw = sigma_cache.get(decision_ts)
    try:
        spread_sigma = float(spread_sigma_raw)
    except Exception:
        spread_sigma = float("nan")
    if not np.isfinite(spread_sigma) or spread_sigma <= 0.0:
        return None
    risk = float(stop_gap_z) * spread_sigma
    return risk if np.isfinite(risk) and risk > 0.0 else None


def _remaining_stop_distance_z(
    *,
    decision_ts: pd.Timestamp,
    z_cache: pd.Series,
    entry_z_abs: float,
    stop_z_abs: float,
) -> float | None:
    raw_z = z_cache.get(decision_ts)
    try:
        z_abs = abs(float(raw_z))
    except Exception:
        return None
    if not np.isfinite(z_abs):
        return None

    entry_z_abs = float(entry_z_abs)
    stop_z_abs = float(stop_z_abs)
    if (
        not np.isfinite(entry_z_abs)
        or not np.isfinite(stop_z_abs)
        or entry_z_abs < 0.0
        or stop_z_abs <= entry_z_abs
    ):
        return None
    if z_abs < entry_z_abs or z_abs >= stop_z_abs:
        return None

    stop_gap_z = stop_z_abs - z_abs
    return float(stop_gap_z) if np.isfinite(stop_gap_z) and stop_gap_z > 0.0 else None


def _build_trade_rows(
    *,
    signals: pd.Series,
    prices: pd.DataFrame,
    beta: pd.Series,
    capital: float,
    cfg: dict[str, Any],
    adv_t1: float | None = None,
    adv_t2: float | None = None,
) -> list[dict[str, Any]]:
    if signals.empty or prices.empty:
        return []

    bt_cfg = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    execution_lag_bars = max(0, int(bt_cfg.get("execution_lag_bars", 1)))
    exec_signals = _lag_positions_for_execution(signals, periods=execution_lag_bars)
    idx = pd.DatetimeIndex(exec_signals.index)
    entry_dates: list[pd.Timestamp] = []
    exit_dates: list[pd.Timestamp] = []
    entry_sides: list[int] = []

    prev = 0
    open_entry: pd.Timestamp | None = None
    open_side = 0
    for ts, raw_sig in exec_signals.items():
        try:
            sig = int(np.sign(float(raw_sig)))
        except Exception:
            sig = 0
        ts = pd.Timestamp(ts)
        if prev == 0 and sig != 0:
            open_entry = ts
            open_side = sig
        elif prev != 0 and sig != prev:
            if open_entry is not None:
                entry_dates.append(open_entry)
                exit_dates.append(ts)
                entry_sides.append(open_side)
            open_entry = ts if sig != 0 else None
            open_side = sig
        prev = sig

    if open_entry is not None and open_side != 0:
        entry_dates.append(open_entry)
        exit_dates.append(pd.Timestamp(idx[-1]))
        entry_sides.append(open_side)

    if not entry_dates:
        return []

    risk_policy = build_risk_policy(
        risk_cfg=cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {},
        backtest_cfg=cfg.get("backtest")
        if isinstance(cfg.get("backtest"), dict)
        else {},
        execution_cfg=cfg.get("execution")
        if isinstance(cfg.get("execution"), dict)
        else {},
    ).sizing
    sig_cfg = cfg.get("signal", {}) if isinstance(cfg.get("signal"), dict) else {}
    entry_z_abs = abs(float(sig_cfg.get("entry_z", sig_cfg.get("min_zscore", 2.0))))
    stop_z_abs = abs(float(sig_cfg.get("stop_z", sig_cfg.get("stop_threshold", 2.0))))
    pair_key = str(cfg.get("_pair_key") or "PAIR")
    y_sym, x_sym = _infer_pair_symbols(cfg, prices)
    y_px = prices["y"] if "y" in prices.columns else None
    x_px = prices["x"] if "x" in prices.columns else None
    z_cache = _coerce_numeric_series(cfg.get("_z_cache"), idx, name="zscore")
    sigma_cache = _coerce_numeric_series(
        cfg.get("_sigma_cache"), idx, name="spread_sigma"
    )
    adv_sum_entry = _sum_adv(adv_t1, adv_t2)

    rows: list[dict[str, Any]] = []
    for entry_ts, exit_ts, sig in zip(entry_dates, exit_dates, entry_sides):
        py0 = _coerce_price(y_px, entry_ts)
        px0 = _coerce_price(x_px, entry_ts)
        py1 = _coerce_price(y_px, exit_ts)
        px1 = _coerce_price(x_px, exit_ts)
        if py0 is None or px0 is None or py1 is None or px1 is None:
            continue

        try:
            beta_entry = float(beta.loc[entry_ts])
        except Exception:
            beta_entry = 1.0
        if not np.isfinite(beta_entry) or beta_entry == 0.0:
            beta_entry = 1.0
        beta_abs = abs(beta_entry)

        gross_entry = float(abs(py0) + beta_abs * abs(px0))
        if not np.isfinite(gross_entry) or gross_entry <= 0.0:
            continue

        entry_pos = _safe_index_pos(idx, pd.Timestamp(entry_ts))
        decision_pos = (
            None if entry_pos is None else max(0, int(entry_pos) - execution_lag_bars)
        )
        decision_ts = (
            pd.Timestamp(idx[decision_pos])
            if decision_pos is not None
            else pd.Timestamp(entry_ts)
        )
        stop_gap_z = _remaining_stop_distance_z(
            decision_ts=decision_ts,
            z_cache=z_cache,
            entry_z_abs=entry_z_abs,
            stop_z_abs=stop_z_abs,
        )
        if stop_gap_z is None:
            continue

        per_unit_risk = _per_unit_stop_risk(
            decision_ts=decision_ts,
            sigma_cache=sigma_cache,
            stop_gap_z=stop_gap_z,
        )
        if per_unit_risk is None:
            continue

        size = size_units_from_risk_budget(
            capital=float(capital),
            risk_per_trade=float(risk_policy.risk_per_trade),
            per_unit_risk=float(per_unit_risk),
            min_units_if_positive=False,
        )
        size = cap_units_by_trade_notional(
            units=int(size),
            capital=float(capital),
            max_trade_pct=float(risk_policy.max_trade_pct),
            per_unit_notional=float(gross_entry),
            min_units_if_positive=False,
        )
        if adv_sum_entry is not None and float(risk_policy.max_participation) > 0.0:
            size = cap_units_by_participation(
                units=int(size),
                max_participation=float(risk_policy.max_participation),
                adv_sum_usd=float(adv_sum_entry),
                per_unit_notional=float(gross_entry),
                require_gt_one_capacity=False,
                min_units_if_positive=False,
            )
        if size <= 0:
            continue

        y_units = float(size) * (1.0 if sig > 0 else -1.0)
        x_units = float(-sig) * float(size) * float(beta_entry)
        notional_y = float(y_units) * float(py0)
        notional_x = float(x_units) * float(px0)
        gross_notional = abs(float(notional_y)) + abs(float(notional_x))
        gross_pnl = y_units * (py1 - py0) + x_units * (px1 - px0)

        rows.append(
            {
                "pair": pair_key,
                "y_symbol": y_sym,
                "x_symbol": x_sym,
                "signal": int(sig),
                "size": int(size),
                "beta_entry": float(beta_entry),
                "y_units": float(y_units),
                "x_units": float(x_units),
                "entry_date": pd.Timestamp(entry_ts),
                "exit_date": pd.Timestamp(exit_ts),
                "entry_price_y": float(py0),
                "entry_price_x": float(px0),
                "exit_price_y": float(py1),
                "exit_price_x": float(px1),
                "notional_y": float(notional_y),
                "notional_x": float(notional_x),
                "gross_notional": float(gross_notional),
                "entry_capital_base": float(capital),
                "adv_sum_entry": float(adv_sum_entry)
                if adv_sum_entry is not None
                else np.nan,
                "decision_date": pd.Timestamp(decision_ts),
                "gross_pnl": float(gross_pnl),
                "holding_days": int(
                    max(
                        1,
                        (
                            pd.Timestamp(exit_ts).normalize()
                            - pd.Timestamp(entry_ts).normalize()
                        ).days,
                    )
                ),
            }
        )
    return rows


@dataclass
class TradeBuilder:
    @staticmethod
    def from_signals(
        *,
        signals: pd.Series,
        prices: pd.DataFrame,
        beta: pd.Series,
        capital: float,
        cfg: dict[str, Any],
        adv_t1: float | None = None,
        adv_t2: float | None = None,
    ) -> pd.DataFrame:
        rows = _build_trade_rows(
            signals=signals,
            prices=prices,
            beta=beta,
            capital=capital,
            cfg=cfg,
            adv_t1=adv_t1,
            adv_t2=adv_t2,
        )
        return pd.DataFrame(rows)


def _trades_to_orders(trades: pd.DataFrame) -> pd.DataFrame:
    return _df_trades_to_orders_df(trades)
