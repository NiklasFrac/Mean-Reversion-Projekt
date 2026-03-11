from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.borrow.accrual import compute_borrow_daily_costs_for_trade_row
from backtest.calendars import map_to_calendar as _cal_map
from backtest.simulators.performance import compute_drawdowns
from backtest.utils.be import _infer_annualization_factor
from backtest.utils.tz import coerce_ts_to_tz, ensure_dtindex_tz

from . import engine_state as _state
from .engine_trades import _get_first_present


def _map_trades_to_daily_pnl(
    trades_eval: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    cfg: Any,
    price_data: Mapping[str, pd.Series],
    borrow_ctx: Any,
) -> tuple[pd.Series, pd.Series, int, int]:
    daily_pnl = pd.Series(0.0, index=calendar, dtype=float)
    daily_gross = pd.Series(0.0, index=calendar, dtype=float)
    mapped_rows = 0
    dropped = 0
    price_cache: dict[str, pd.Series | None] = {}
    policy = str(getattr(cfg, "calendar_mapping", "prior")).lower()
    try:
        _raw = getattr(cfg, "raw_yaml", {}) or {}
        strict_only = bool(
            ((_raw.get("backtest") or {}).get("calendar") or {}).get(
                "strict_only", False
            )
        )
    except Exception:
        strict_only = False
    cal_tz = str(getattr(calendar, "tz", _state._EX_TZ))
    borrow_enabled = False
    if borrow_ctx is not None:
        try:
            borrow_enabled = bool(getattr(borrow_ctx, "enabled", True))
        except Exception:
            borrow_enabled = True

    def _map_ts(ts: pd.Timestamp | None) -> pd.Timestamp | None:
        if ts is None or pd.isna(ts):
            return None
        mapped = _cal_map(ts, calendar, policy)
        if (not strict_only) and mapped is None and policy != "prior":
            mapped = _cal_map(ts, calendar, "prior")
        if (not strict_only) and mapped is None and policy != "nearest":
            mapped = _cal_map(ts, calendar, "nearest")
        if mapped is None:
            return None
        mapped = coerce_ts_to_tz(mapped, cal_tz, naive_is_utc=_state._NAIVE_IS_UTC)
        if mapped not in calendar:
            fallback = _cal_map(mapped, calendar, "nearest")
            if fallback is None:
                return None
            mapped = coerce_ts_to_tz(
                fallback, cal_tz, naive_is_utc=_state._NAIVE_IS_UTC
            )
            if mapped not in calendar:
                return None
        return mapped

    def _price_series(sym: str) -> pd.Series | None:
        if sym in price_cache:
            return price_cache[sym]
        s = price_data.get(sym)
        if s is None or not isinstance(s, pd.Series) or s.empty:
            price_cache[sym] = None
            return None
        if not isinstance(s.index, pd.DatetimeIndex):
            price_cache[sym] = None
            return None
        s = pd.to_numeric(s, errors="coerce")
        idx = cast(pd.DatetimeIndex, s.index)
        try:
            idx = ensure_dtindex_tz(idx, cal_tz, naive_is_utc=_state._NAIVE_IS_UTC)
        except Exception:
            pass
        s = s.copy()
        s.index = idx
        s = s.sort_index()
        s = s.reindex(calendar, method="ffill")
        price_cache[sym] = s
        return s

    def _pick_price(row: pd.Series, keys: tuple[str, ...]) -> float | None:
        for k in keys:
            if k in row and pd.notna(row[k]):
                try:
                    v = float(row[k])
                    if np.isfinite(v):
                        return v
                except Exception:
                    continue
        return None

    def _split_cost(
        row: pd.Series,
        *,
        total_key: str,
        entry_key: str,
        exit_key: str,
    ) -> tuple[float, float]:
        total = _get_first_present(row, (total_key,))
        total = float(total) if total is not None and np.isfinite(float(total)) else 0.0
        entry = _get_first_present(row, (entry_key,))
        exit_ = _get_first_present(row, (exit_key,))
        entry_val = (
            float(entry) if entry is not None and np.isfinite(float(entry)) else None
        )
        exit_val = (
            float(exit_) if exit_ is not None and np.isfinite(float(exit_)) else None
        )

        if entry_val is None and exit_val is None:
            half = 0.5 * total
            return half, half
        if entry_val is None:
            return total - float(exit_val or 0.0), float(exit_val or 0.0)
        if exit_val is None:
            return float(entry_val), total - float(entry_val)
        return float(entry_val), float(exit_val)

    for _, r in trades_eval.iterrows():
        entry_raw = r.get("entry_date")
        exit_raw = r.get("exit_date")
        entry_ts = _map_ts(pd.Timestamp(entry_raw)) if entry_raw is not None else None
        exit_ts = _map_ts(pd.Timestamp(exit_raw)) if exit_raw is not None else None
        if entry_ts is None or exit_ts is None:
            dropped += 1
            continue
        if exit_ts < entry_ts:
            dropped += 1
            continue

        y_sym = r.get("y_symbol") or r.get("t1_symbol") or r.get("leg1_symbol")
        x_sym = r.get("x_symbol") or r.get("t2_symbol") or r.get("leg2_symbol")
        if y_sym is None and x_sym is None:
            y_sym = r.get("symbol") or r.get("asset") or r.get("ticker")
        y_sym = str(y_sym).upper().strip() if y_sym is not None else None
        x_sym = str(x_sym).upper().strip() if x_sym is not None else None

        y_units = r.get("y_units")
        x_units = r.get("x_units")
        if y_units is None or x_units is None or pd.isna(y_units) or pd.isna(x_units):
            size = _get_first_present(r, ("size", "qty", "quantity", "units"))
            sig = _get_first_present(r, ("signal",))
            sig = 1.0 if sig is None else (1.0 if sig >= 0 else -1.0)
            beta = _get_first_present(r, ("beta_entry", "beta"))
            beta = 1.0 if beta is None else float(beta)
            if size is not None and np.isfinite(float(size)) and float(size) != 0.0:
                y_units = abs(float(size)) * (1.0 if sig >= 0 else -1.0)
                x_units = abs(float(size)) * (-1.0 if sig >= 0 else 1.0) * float(beta)

        y_units_f = (
            float(y_units) if y_units is not None and not pd.isna(y_units) else 0.0
        )
        x_units_f = (
            float(x_units) if x_units is not None and not pd.isna(x_units) else 0.0
        )
        if y_units_f == 0.0 and x_units_f == 0.0:
            dropped += 1
            continue

        py_series = _price_series(y_sym) if y_sym else None
        px_series = _price_series(x_sym) if x_sym else None
        if (y_sym and py_series is None) or (x_sym and px_series is None):
            dropped += 1
            continue

        start_idx = int(calendar.get_indexer([entry_ts])[0])
        end_idx = int(calendar.get_indexer([exit_ts])[0])
        if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
            dropped += 1
            continue

        idx_slice = calendar[start_idx : end_idx + 1]
        trade_value = pd.Series(0.0, index=idx_slice, dtype=float)
        gross_slice = pd.Series(0.0, index=idx_slice, dtype=float)

        if y_sym and py_series is not None:
            py0 = _pick_price(r, ("entry_price_y", "price_y", "entry_price"))
            if py0 is None:
                try:
                    py0 = float(py_series.iloc[start_idx])
                except Exception:
                    py0 = None
            py_slice = py_series.iloc[start_idx : end_idx + 1]
            if py0 is None or not np.isfinite(py0):
                dropped += 1
                continue
            if py_slice.isna().any():
                py_slice = py_slice.fillna(py0)
            trade_value = trade_value.add(
                y_units_f * (py_slice - float(py0)), fill_value=0.0
            )
            gross_slice = gross_slice.add(abs(y_units_f) * py_slice, fill_value=0.0)

        if x_sym and px_series is not None and x_units_f != 0.0:
            px0 = _pick_price(r, ("entry_price_x", "price_x", "entry_price"))
            if px0 is None:
                try:
                    px0 = float(px_series.iloc[start_idx])
                except Exception:
                    px0 = None
            px_slice = px_series.iloc[start_idx : end_idx + 1]
            if px0 is None or not np.isfinite(px0):
                dropped += 1
                continue
            if px_slice.isna().any():
                px_slice = px_slice.fillna(px0)
            trade_value = trade_value.add(
                x_units_f * (px_slice - float(px0)), fill_value=0.0
            )
            gross_slice = gross_slice.add(abs(x_units_f) * px_slice, fill_value=0.0)

        if not gross_slice.empty:
            daily_gross = daily_gross.add(gross_slice, fill_value=0.0)

        gross_pnl = _get_first_present(r, ("gross_pnl",))
        if gross_pnl is not None and np.isfinite(float(gross_pnl)):
            try:
                trade_value.iloc[-1] = float(gross_pnl)
            except Exception:
                pass

        delta = trade_value.diff().fillna(trade_value.iloc[0])

        fee_entry, fee_exit = _split_cost(
            r, total_key="fees", entry_key="fees_entry", exit_key="fees_exit"
        )
        slip_entry, slip_exit = _split_cost(
            r,
            total_key="slippage_cost",
            entry_key="slippage_cost_entry",
            exit_key="slippage_cost_exit",
        )
        impact_entry, impact_exit = _split_cost(
            r,
            total_key="impact_cost",
            entry_key="impact_cost_entry",
            exit_key="impact_cost_exit",
        )
        diag_only = bool(r.get("exec_diag_costs_only", False))
        entry_cost = float(fee_entry)
        exit_cost = float(fee_exit)
        if not diag_only:
            entry_cost += float(slip_entry + impact_entry)
            exit_cost += float(slip_exit + impact_exit)

        buyin = _get_first_present(r, ("buyin_penalty_cost",))
        if buyin is not None and np.isfinite(float(buyin)):
            exit_cost += float(buyin)
        emergency_penalty = _get_first_present(r, ("exec_emergency_penalty_cost",))
        if emergency_penalty is not None and np.isfinite(float(emergency_penalty)):
            exit_cost += float(emergency_penalty)

        if len(delta) > 0:
            delta.iloc[0] = float(delta.iloc[0]) + float(entry_cost)
            delta.iloc[-1] = float(delta.iloc[-1]) + float(exit_cost)

        borrow_daily = pd.Series(dtype=float)
        if borrow_enabled:
            try:
                borrow_daily = compute_borrow_daily_costs_for_trade_row(
                    r,
                    calendar=calendar,
                    price_data=price_data,
                    borrow_ctx=borrow_ctx,
                )
            except Exception:
                borrow_daily = pd.Series(dtype=float)

        if borrow_daily is not None and not borrow_daily.empty:
            for d, v in borrow_daily.items():
                if v is None or not np.isfinite(float(v)):
                    continue
                mapped = _map_ts(pd.Timestamp(cast(Any, d)))
                if mapped is None or mapped not in delta.index:
                    continue
                delta.at[mapped] = float(delta.at[mapped]) + float(v)
        else:
            borrow_total = _get_first_present(r, ("borrow_cost",))
            if (
                borrow_total is not None
                and np.isfinite(float(borrow_total))
                and len(delta) > 0
            ):
                total_val = float(borrow_total)
                if total_val != 0.0:
                    delta += total_val / float(len(delta))

        daily_pnl.loc[idx_slice] = (
            daily_pnl.loc[idx_slice].to_numpy() + delta.to_numpy()
        )
        mapped_rows += 1

    return daily_pnl, daily_gross, mapped_rows, dropped


def _compute_equity_and_stats(
    daily_pnl: pd.Series,
    daily_gross: pd.Series,
    *,
    calendar: pd.DatetimeIndex,
    cfg: Any,
    trades_eval: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_pnl = daily_pnl.reindex(calendar).astype(float).fillna(0.0)
    equity = (
        pd.Series(float(cfg.initial_capital), index=calendar, name="equity")
        .add(daily_pnl.cumsum(), fill_value=0.0)
        .astype(float)
    )
    if equity.isna().any():
        equity = equity.ffill().bfill()
    returns = (
        equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    )
    dd, max_dd, _, _ = compute_drawdowns(equity)

    ann = int(cfg.annualization_factor or _infer_annualization_factor(calendar))
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    eps = 1e-12
    denom = sigma if abs(sigma) > eps else (eps if sigma >= 0 else -eps)
    sharpe = float(mu / denom * np.sqrt(ann))

    days = (calendar[-1] - calendar[0]).days if len(calendar) > 1 else 0
    years = max(1, days) / 365.25
    cagr = (
        float((equity.iloc[-1] / float(cfg.initial_capital)) ** (1.0 / years) - 1.0)
        if (np.isfinite(years) and years > 0.0)
        else 0.0
    )
    win_rate = (
        float((trades_eval.get("net_pnl", pd.Series(dtype=float)) > 0).mean())
        if not trades_eval.empty
        else 0.0
    )

    dd_pct = (dd / equity.cummax().replace(0, np.nan)).fillna(0.0).astype(float)
    stats = pd.DataFrame(
        {
            "equity": equity,
            "returns": returns,
            "drawdown": dd.astype(float),
            "drawdown_pct": dd_pct,
        }
    )
    stats.index.name = "date"
    for c in ("equity", "returns", "drawdown", "drawdown_pct"):
        stats[c] = pd.to_numeric(stats[c], errors="coerce")

    stats["Sharpe"] = float(sharpe)
    stats["CAGR"] = float(cagr)
    stats["max_drawdown"] = float(max_dd or 0.0)
    stats["WinRate"] = float(win_rate)
    stats["NumTrades"] = int(len(trades_eval))

    fin_info = {
        "sharpe": float(sharpe),
        "cagr": float(cagr) if np.isfinite(cagr) else float("nan"),
        "max_drawdown": float(max_dd or 0.0),
        "win_rate": float(win_rate),
    }

    return stats, fin_info
