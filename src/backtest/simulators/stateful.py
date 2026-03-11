from __future__ import annotations

import heapq
import logging
from typing import Any, Callable, Mapping, cast

import numpy as np
import pandas as pd

from backtest.borrow.accrual import compute_borrow_daily_costs_for_trade_row
from backtest.calendars import apply_settlement_lag
from backtest.risk_policy import (
    build_risk_policy,
    cap_units_by_participation,
    cap_units_by_trade_notional,
)
from backtest.utils.tz import align_ts_to_index, to_naive_day, to_naive_local

logger = logging.getLogger("backtest.simulators.stateful")


def _clean_price_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.loc[~s.index.isna()].sort_index()
    return s


def _safe_float(val: Any) -> float | None:
    try:
        out = float(val)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _asof_price(series: pd.Series | None, ts: pd.Timestamp) -> float | None:
    if series is None or series.empty:
        return None
    try:
        ser = series
        idx = ser.index
        if not isinstance(idx, pd.DatetimeIndex):
            return None
        t = pd.Timestamp(ts)
        t = align_ts_to_index(t, idx)
        pos = int(idx.searchsorted(t, side="right") - 1)
        if pos < 0:
            return None
        v = float(ser.iloc[pos])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _infer_symbols(row: pd.Series) -> tuple[str | None, str | None]:
    y_sym = row.get("y_symbol") or row.get("t1_symbol") or row.get("leg1_symbol")
    x_sym = row.get("x_symbol") or row.get("t2_symbol") or row.get("leg2_symbol")
    if y_sym is None and x_sym is None:
        sym = row.get("symbol") or row.get("asset") or row.get("ticker")
        y_sym = sym
    y = str(y_sym).upper().strip() if y_sym is not None else None
    x = str(x_sym).upper().strip() if x_sym is not None else None
    return y if y else None, x if x else None


def _infer_units(size: int, signal: int, beta: float) -> tuple[int, int]:
    if size <= 0 or signal == 0:
        return 0, 0
    y_units = int(size) * (1 if signal > 0 else -1)
    beta_eff = float(beta) if np.isfinite(beta) and float(beta) != 0.0 else 1.0
    beta_abs = float(abs(beta_eff))
    x_units_abs = max(1, int(round(float(size) * beta_abs)))
    x_sign = int(np.sign(-float(signal) * beta_eff))
    x_units = int(x_sign if x_sign != 0 else -int(signal)) * int(x_units_abs)
    return int(y_units), int(x_units)


def _make_price_lookup(
    price_data: Mapping[str, pd.Series],
) -> Callable[[str | None, pd.Timestamp], float | None]:
    price_cache: dict[str, pd.Series] = {}

    def _price_for(sym: str | None, ts: pd.Timestamp) -> float | None:
        if sym is None:
            return None
        key = str(sym).upper()
        if key not in price_cache:
            s = price_data.get(key)
            if isinstance(s, pd.Series):
                price_cache[key] = _clean_price_series(s)
            else:
                price_cache[key] = pd.Series(dtype=float)
        return _asof_price(price_cache[key], ts)

    return _price_for


def _normalize_trade_ts(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    try:
        return to_naive_day(t)
    except Exception:
        return pd.Timestamp(to_naive_local(t)).normalize()


def _normalize_day_index(values: Any) -> pd.DatetimeIndex:
    idx_like = pd.to_datetime(values, errors="coerce")
    if isinstance(idx_like, pd.Series):
        idx_like = idx_like.dropna()
        if idx_like.empty:
            return pd.DatetimeIndex([])
        idx = pd.DatetimeIndex(idx_like)
    else:
        idx = pd.DatetimeIndex(idx_like)
    idx = idx[~idx.isna()]
    if idx.empty:
        return pd.DatetimeIndex([])
    return to_naive_day(idx)


def _build_calendar(
    price_data: Mapping[str, pd.Series],
    *,
    entry_dates: pd.Series,
    exit_dates: pd.Series,
) -> pd.DatetimeIndex:
    idxs: list[pd.DatetimeIndex] = []
    for s in price_data.values():
        if (
            isinstance(s, pd.Series)
            and isinstance(s.index, pd.DatetimeIndex)
            and not s.empty
        ):
            idx = _normalize_day_index(s.index)
            if not idx.empty:
                idxs.append(idx)
    trade_idx = _normalize_day_index(pd.concat([entry_dates, exit_dates]))
    if not trade_idx.empty:
        idxs.append(trade_idx)
    if not idxs:
        return pd.DatetimeIndex([])
    out = idxs[0]
    for idx in idxs[1:]:
        out = cast(pd.DatetimeIndex, out.union(idx))
    return pd.DatetimeIndex(out.unique()).sort_values()


def _split_trade_cost(
    row: Mapping[str, Any],
    *,
    total_key: str,
    entry_key: str,
    exit_key: str,
) -> tuple[float, float]:
    total = row.get(total_key, None)
    entry = row.get(entry_key, None)
    exit_ = row.get(exit_key, None)
    total = float(total) if total is not None and np.isfinite(float(total)) else None
    entry_val = (
        float(entry) if entry is not None and np.isfinite(float(entry)) else None
    )
    exit_val = float(exit_) if exit_ is not None and np.isfinite(float(exit_)) else None
    if total is None:
        if entry_val is None and exit_val is None:
            return 0.0, 0.0
        return float(entry_val or 0.0), float(exit_val or 0.0)
    if entry_val is None and exit_val is None:
        half = 0.5 * float(total)
        return half, half
    if entry_val is None:
        return float(total) - float(exit_val or 0.0), float(exit_val or 0.0)
    if exit_val is None:
        return float(entry_val), float(total) - float(entry_val)
    return float(entry_val), float(exit_val)


def _entry_trade_cost(row: Mapping[str, Any]) -> float:
    fee_entry, _ = _split_trade_cost(
        row, total_key="fees", entry_key="fees_entry", exit_key="fees_exit"
    )
    slip_entry, _ = _split_trade_cost(
        row,
        total_key="slippage_cost",
        entry_key="slippage_cost_entry",
        exit_key="slippage_cost_exit",
    )
    impact_entry, _ = _split_trade_cost(
        row,
        total_key="impact_cost",
        entry_key="impact_cost_entry",
        exit_key="impact_cost_exit",
    )
    return float(fee_entry + slip_entry + impact_entry)


def _exit_trade_cost(row: Mapping[str, Any]) -> float:
    _, fee_exit = _split_trade_cost(
        row, total_key="fees", entry_key="fees_entry", exit_key="fees_exit"
    )
    _, slip_exit = _split_trade_cost(
        row,
        total_key="slippage_cost",
        entry_key="slippage_cost_entry",
        exit_key="slippage_cost_exit",
    )
    _, impact_exit = _split_trade_cost(
        row,
        total_key="impact_cost",
        entry_key="impact_cost_entry",
        exit_key="impact_cost_exit",
    )
    exit_cost = float(fee_exit + slip_exit + impact_exit)
    buyin = row.get("buyin_penalty_cost", None)
    if buyin is not None and np.isfinite(float(buyin)):
        exit_cost += float(buyin)
    return float(exit_cost)


def _unrealized_open_trades(
    open_trades: list[dict[str, Any]],
    ts: pd.Timestamp,
    price_for: Callable[[str | None, pd.Timestamp], float | None],
) -> float:
    total = 0.0
    for t in open_trades:
        py = price_for(t.get("y_sym"), ts)
        px = price_for(t.get("x_sym"), ts)
        if py is None:
            py = float(t["entry_py"])
        if px is None:
            px = float(t["entry_px"])
        total += float(t["y_units"]) * (py - float(t["entry_py"])) + float(
            t["x_units"]
        ) * (px - float(t["entry_px"]))
    return float(total)


def _gross_open_trades(
    open_trades: list[dict[str, Any]],
    ts: pd.Timestamp,
    price_for: Callable[[str | None, pd.Timestamp], float | None],
) -> float:
    total = 0.0
    for t in open_trades:
        py = price_for(t.get("y_sym"), ts)
        px = price_for(t.get("x_sym"), ts)
        if py is None:
            py = float(t["entry_py"])
        if px is None:
            px = float(t["entry_px"])
        total += abs(float(t["y_units"])) * abs(float(py)) + abs(
            float(t["x_units"])
        ) * abs(float(px))
    return float(total)


def rescale_trades_stateful(
    trades: pd.DataFrame,
    price_data: Mapping[str, pd.Series],
    *,
    initial_capital: float,
    wf_params: Mapping[int, Mapping[str, float]] | None = None,
    borrow_ctx: Any | None = None,
    settlement_lag_bars: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Rescale trades sequentially using MTM equity at each entry timestamp.

    Uses the existing trade schedule (entry/exit dates) and scales size
    based on current equity vs entry_capital_base. Applies per-trade caps:
      - max_trade_pct (risk)
      - max_participation (liquidity; uses adv_sum_entry if present)
    """
    if trades is None or trades.empty:
        return trades, {"resized": 0, "dropped": 0}

    df = trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"])
    if df.empty:
        return df, {"resized": 0, "dropped": 0}

    df = df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    price_for = _make_price_lookup(price_data)
    calendar = _build_calendar(
        price_data, entry_dates=df["entry_date"], exit_dates=df["exit_date"]
    )
    settle_lag = int(max(0, settlement_lag_bars))

    def _settle_ts(ts: pd.Timestamp) -> pd.Timestamp:
        if settle_lag <= 0 or calendar.empty:
            return ts
        try:
            return apply_settlement_lag(ts, calendar, lag_bars=settle_lag)
        except Exception:
            return ts

    borrow_enabled = (
        bool(getattr(borrow_ctx, "enabled", True)) if borrow_ctx is not None else False
    )

    open_trades: list[dict[str, Any]] = []
    pending_settlements: list[tuple[pd.Timestamp, int, float]] = []
    heapq.heapify(pending_settlements)
    realized_pnl = 0.0
    borrow_total = 0.0
    equity = float(initial_capital)
    prev_unrealized = 0.0
    resized_rows: list[dict[str, Any]] = []
    dropped = 0
    last_ts: pd.Timestamp | None = None

    def _advance_to(ts: pd.Timestamp) -> None:
        nonlocal last_ts, equity, prev_unrealized, realized_pnl, borrow_total
        if last_ts is None or calendar.empty:
            last_ts = ts
            return
        days = calendar[(calendar > last_ts) & (calendar < ts)]
        for dt in days:
            # Apply settlements that become available on/before this day.
            while pending_settlements and pending_settlements[0][0] <= dt:
                _dt_settle, _hid, pnl_val = heapq.heappop(pending_settlements)
                realized_pnl += float(pnl_val)
                equity += float(pnl_val)

            # Close trades that have exited by this day.
            still_open: list[dict[str, Any]] = []
            for t in open_trades:
                if t["exit_ts"] <= dt:
                    py1 = price_for(t["y_sym"], t["exit_ts"])
                    px1 = price_for(t["x_sym"], t["exit_ts"])
                    if py1 is None:
                        py1 = t["entry_py"]
                    if px1 is None:
                        px1 = t["entry_px"]
                    pnl = t["y_units"] * (py1 - t["entry_py"]) + t["x_units"] * (
                        px1 - t["entry_px"]
                    )
                    pnl_net = float(pnl) - float(t.get("exit_cost", 0.0))
                    heapq.heappush(
                        pending_settlements,
                        (_settle_ts(pd.Timestamp(t["exit_ts"])), int(t["id"]), pnl_net),
                    )
                else:
                    still_open.append(t)
            open_trades[:] = still_open

            # MTM update.
            unrealized = _unrealized_open_trades(open_trades, dt, price_for)
            equity += float(unrealized - prev_unrealized)
            prev_unrealized = unrealized

            # Borrow accrual (<=0).
            if borrow_enabled:
                daily_borrow = 0.0
                for t in open_trades:
                    series = t.get("borrow_series")
                    if isinstance(series, pd.Series) and not series.empty:
                        v = series.get(pd.Timestamp(dt), 0.0)
                        if v is not None and np.isfinite(float(v)):
                            daily_borrow += float(v)
                if daily_borrow != 0.0:
                    equity += float(daily_borrow)
                    borrow_total += float(daily_borrow)

        last_ts = ts

    for row_id, row in df.iterrows():
        entry_ts = _normalize_trade_ts(pd.Timestamp(row["entry_date"]))
        exit_ts = _normalize_trade_ts(pd.Timestamp(row["exit_date"]))
        if exit_ts < entry_ts:
            dropped += 1
            continue

        _advance_to(entry_ts)
        # Apply settlements available by entry day (cash available at open).
        while pending_settlements and pending_settlements[0][0] <= entry_ts:
            _dt_settle, _hid, pnl_val = heapq.heappop(pending_settlements)
            realized_pnl += float(pnl_val)
            equity += float(pnl_val)

        unrealized = _unrealized_open_trades(open_trades, entry_ts, price_for)
        equity += float(unrealized - prev_unrealized)
        prev_unrealized = unrealized

        base_size = int(row.get("size", 0) or 0)
        base_cap = float(
            row.get("entry_capital_base", initial_capital) or initial_capital
        )
        if base_size <= 0 or base_cap <= 0 or equity <= 0:
            dropped += 1
            continue
        scale = equity / base_cap
        size = int(np.floor(base_size * scale))
        if size <= 0:
            dropped += 1
            continue

        wf_i = int(row.get("wf_i", -1))
        params = wf_params.get(wf_i, {}) if wf_params is not None else {}
        sizing_policy = build_risk_policy(risk_cfg=params, execution_cfg=params).sizing
        max_trade_pct = float(sizing_policy.max_trade_pct)
        max_participation = float(sizing_policy.max_participation)

        signal = int(row.get("signal", 0) or 0)
        beta_entry = float(row.get("beta_entry", 1.0) or 1.0)

        y_sym, x_sym = _infer_symbols(row)
        py0f = _safe_float(row.get("entry_price_y"))
        px0f = _safe_float(row.get("entry_price_x"))
        py0f = py0f if py0f is not None else price_for(y_sym, entry_ts)
        px0f = px0f if px0f is not None else price_for(x_sym, entry_ts)
        if py0f is None or px0f is None:
            dropped += 1
            continue

        per_unit_value = abs(float(py0f)) + abs(float(px0f)) * abs(float(beta_entry))
        if per_unit_value <= 0.0:
            dropped += 1
            continue

        size = cap_units_by_trade_notional(
            units=int(size),
            capital=float(equity),
            max_trade_pct=float(max_trade_pct),
            per_unit_notional=float(per_unit_value),
            min_units_if_positive=False,
        )

        adv_sum_entry = row.get("adv_sum_entry")
        if (
            max_participation > 0
            and adv_sum_entry is not None
            and np.isfinite(float(adv_sum_entry))
        ):
            size = cap_units_by_participation(
                units=int(size),
                max_participation=float(max_participation),
                adv_sum_usd=float(adv_sum_entry),
                per_unit_notional=float(per_unit_value),
                require_gt_one_capacity=False,
                min_units_if_positive=False,
            )

        if size <= 0:
            dropped += 1
            continue

        y_units, x_units = _infer_units(size, signal, beta_entry)
        if y_units == 0 and x_units == 0:
            dropped += 1
            continue

        py1f = _safe_float(row.get("exit_price_y"))
        px1f = _safe_float(row.get("exit_price_x"))
        py1f = py1f if py1f is not None else price_for(y_sym, exit_ts)
        px1f = px1f if px1f is not None else price_for(x_sym, exit_ts)
        if py1f is None or px1f is None:
            dropped += 1
            continue

        notional_y = float(y_units) * float(py0f)
        notional_x = float(x_units) * float(px0f)
        gross_notional = abs(notional_y) + abs(notional_x)
        gross_pnl = float(y_units) * (float(py1f) - float(py0f)) + float(x_units) * (
            float(px1f) - float(px0f)
        )
        row_map = cast(Mapping[str, Any], row.to_dict())
        entry_cost = _entry_trade_cost(row_map)
        exit_cost = _exit_trade_cost(row_map)

        out_row = row.to_dict()
        out_row["id"] = int(cast(Any, row_id))
        if y_sym is not None:
            out_row["y_symbol"] = str(y_sym)
        if x_sym is not None:
            out_row["x_symbol"] = str(x_sym)
        out_row["size"] = int(size)
        out_row["y_units"] = int(y_units)
        out_row["x_units"] = int(x_units)
        out_row["notional_y"] = float(notional_y)
        out_row["notional_x"] = float(notional_x)
        out_row["gross_notional"] = float(gross_notional)
        out_row["gross_pnl"] = float(gross_pnl)
        out_row["entry_price_y"] = float(py0f)
        out_row["entry_price_x"] = float(px0f)
        out_row["exit_price_y"] = float(py1f)
        out_row["exit_price_x"] = float(px1f)
        out_row["sizing_equity"] = float(equity)
        out_row["sizing_scale"] = float(scale)

        resized_rows.append(out_row)

        equity -= float(entry_cost)

        borrow_series = None
        if borrow_enabled and not calendar.empty:
            try:
                tmp_row = pd.Series(out_row)
                borrow_series = compute_borrow_daily_costs_for_trade_row(
                    tmp_row,
                    calendar=calendar,
                    price_data=price_data,
                    borrow_ctx=borrow_ctx,
                )
                if isinstance(borrow_series, pd.Series) and not borrow_series.empty:
                    idx = to_naive_day(
                        pd.to_datetime(borrow_series.index, errors="coerce")
                    )
                    borrow_series = pd.Series(
                        pd.to_numeric(borrow_series.to_numpy(), errors="coerce"),
                        index=pd.DatetimeIndex(idx),
                    ).dropna()
            except Exception:
                borrow_series = None

        open_trades.append(
            {
                "id": int(cast(Any, row_id)),
                "y_sym": y_sym,
                "x_sym": x_sym,
                "entry_py": float(py0f),
                "entry_px": float(px0f),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "y_units": int(y_units),
                "x_units": int(x_units),
                "exit_cost": float(exit_cost),
                "borrow_series": borrow_series,
            }
        )

    out_df = pd.DataFrame(resized_rows)
    report = {
        "resized": int(len(out_df)),
        "dropped": int(dropped),
        "settlement_lag_bars": int(settle_lag),
        "borrow_total": float(borrow_total),
        "realized_pnl": float(realized_pnl),
    }
    return out_df, report


def replay_trades_mtm(
    trades: pd.DataFrame,
    price_data: Mapping[str, pd.Series],
    *,
    initial_capital: float,
    borrow_ctx: Any | None = None,
    settlement_lag_bars: int = 0,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """
    Replay trades against (possibly perturbed) prices with MTM equity tracking.

    Keeps original trade schedule and size; recomputes entry/exit prices and PnL.
    Returns (equity_series, trades_out, report).
    """
    if trades is None or trades.empty:
        return (
            pd.Series(dtype=float, name="equity"),
            trades,
            {"replayed": 0, "dropped": 0},
        )

    df = trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"])
    if df.empty:
        return pd.Series(dtype=float, name="equity"), df, {"replayed": 0, "dropped": 0}

    df = df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    price_for = _make_price_lookup(price_data)
    calendar = _build_calendar(
        price_data, entry_dates=df["entry_date"], exit_dates=df["exit_date"]
    )
    settle_lag = int(max(0, settlement_lag_bars))

    def _settle_ts(ts: pd.Timestamp) -> pd.Timestamp:
        if settle_lag <= 0 or calendar.empty:
            return ts
        try:
            return apply_settlement_lag(ts, calendar, lag_bars=settle_lag)
        except Exception:
            return ts

    borrow_enabled = (
        bool(getattr(borrow_ctx, "enabled", True)) if borrow_ctx is not None else False
    )

    open_trades: list[dict[str, Any]] = []
    pending_settlements: list[tuple[pd.Timestamp, int, float]] = []
    heapq.heapify(pending_settlements)
    realized_pnl = 0.0
    borrow_total = 0.0
    equity = float(initial_capital)
    prev_unrealized = 0.0
    dropped = 0
    equity_series = pd.Series(index=calendar, dtype=float, name="equity")

    # Pre-build trade rows with pricing and units
    replay_rows: list[dict[str, Any]] = []
    entries_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for row_id, row in df.iterrows():
        entry_ts = _normalize_trade_ts(pd.Timestamp(row["entry_date"]))
        exit_ts = _normalize_trade_ts(pd.Timestamp(row["exit_date"]))
        if exit_ts < entry_ts:
            dropped += 1
            continue

        y_sym, x_sym = _infer_symbols(row)

        # Units: prefer explicit y_units/x_units if available
        y_units = row.get("y_units", None)
        x_units = row.get("x_units", None)
        if (
            y_units is None
            or x_units is None
            or not np.isfinite(float(y_units))
            or not np.isfinite(float(x_units))
        ):
            size = int(row.get("size", 0) or 0)
            signal = int(row.get("signal", 0) or 0)
            beta_entry = float(row.get("beta_entry", 1.0) or 1.0)
            y_units, x_units = _infer_units(size, signal, beta_entry)
        else:
            y_units = int(float(y_units))
            x_units = int(float(x_units))

        if y_units == 0 and x_units == 0:
            dropped += 1
            continue

        py0f = _safe_float(row.get("entry_price_y"))
        px0f = _safe_float(row.get("entry_price_x"))
        py0f = price_for(y_sym, entry_ts) if py0f is None else py0f
        px0f = price_for(x_sym, entry_ts) if px0f is None else px0f
        if py0f is None or px0f is None:
            dropped += 1
            continue

        py1f = _safe_float(row.get("exit_price_y"))
        px1f = _safe_float(row.get("exit_price_x"))
        py1f = price_for(y_sym, exit_ts) if py1f is None else py1f
        px1f = price_for(x_sym, exit_ts) if px1f is None else px1f
        if py1f is None or px1f is None:
            dropped += 1
            continue

        gross_pnl = float(y_units) * (float(py1f) - float(py0f)) + float(x_units) * (
            float(px1f) - float(px0f)
        )

        fees = _safe_float(row.get("fees")) or 0.0
        slip = _safe_float(row.get("slippage_cost")) or 0.0
        impact = _safe_float(row.get("impact_cost")) or 0.0
        buyin = _safe_float(row.get("buyin_penalty_cost")) or 0.0
        borrow_cost = _safe_float(row.get("borrow_cost")) or 0.0
        total_costs = fees + slip + impact + buyin + borrow_cost
        if ("total_costs" in row) and not any(
            k in row
            for k in (
                "fees",
                "slippage_cost",
                "impact_cost",
                "buyin_penalty_cost",
                "borrow_cost",
            )
        ):
            tc = _safe_float(row.get("total_costs"))
            total_costs = float(tc or 0.0)

        net_pnl = float(gross_pnl) - float(total_costs)

        out_row = row.to_dict()
        out_row["entry_date"] = entry_ts
        out_row["exit_date"] = exit_ts
        out_row["y_units"] = int(y_units)
        out_row["x_units"] = int(x_units)
        out_row["entry_price_y"] = float(py0f)
        out_row["entry_price_x"] = float(px0f)
        out_row["exit_price_y"] = float(py1f)
        out_row["exit_price_x"] = float(px1f)
        out_row["gross_pnl"] = float(gross_pnl)
        out_row["net_pnl"] = float(net_pnl)
        out_row["total_costs"] = float(total_costs)
        out_row["borrow_cost"] = float(borrow_cost)

        replay_rows.append(out_row)
        entries_by_date.setdefault(entry_ts, []).append(out_row)

    if calendar.empty:
        return (
            pd.Series(dtype=float, name="equity"),
            pd.DataFrame(replay_rows),
            {"replayed": len(replay_rows), "dropped": dropped},
        )

    for dt in calendar:
        # Apply settlements due
        while pending_settlements and pending_settlements[0][0] <= dt:
            _dt_settle, _hid, pnl_val = heapq.heappop(pending_settlements)
            realized_pnl += float(pnl_val)
            equity += float(pnl_val)

        # Close trades that exit by this day
        if open_trades:
            still_open: list[dict[str, Any]] = []
            for t in open_trades:
                if t["exit_ts"] <= dt:
                    py1 = price_for(t["y_sym"], t["exit_ts"]) or t["entry_py"]
                    px1 = price_for(t["x_sym"], t["exit_ts"]) or t["entry_px"]
                    pnl = t["y_units"] * (py1 - t["entry_py"]) + t["x_units"] * (
                        px1 - t["entry_px"]
                    )
                    pnl_net = float(pnl) - float(t.get("exit_cost", 0.0))
                    heapq.heappush(
                        pending_settlements,
                        (_settle_ts(pd.Timestamp(t["exit_ts"])), int(t["id"]), pnl_net),
                    )
                else:
                    still_open.append(t)
            open_trades[:] = still_open

        # Open new trades
        for entry_row in entries_by_date.get(dt, []):
            entry_cost = _entry_trade_cost(entry_row)
            exit_cost = _exit_trade_cost(entry_row)

            borrow_series = None
            if borrow_enabled and not calendar.empty:
                try:
                    tmp_row = pd.Series(entry_row)
                    borrow_series = compute_borrow_daily_costs_for_trade_row(
                        tmp_row,
                        calendar=calendar,
                        price_data=price_data,
                        borrow_ctx=borrow_ctx,
                    )
                    if isinstance(borrow_series, pd.Series) and not borrow_series.empty:
                        idx = to_naive_day(
                            pd.to_datetime(borrow_series.index, errors="coerce")
                        )
                        borrow_series = pd.Series(
                            pd.to_numeric(borrow_series.to_numpy(), errors="coerce"),
                            index=pd.DatetimeIndex(idx),
                        ).dropna()
                        borrow_sum = float(borrow_series.sum())
                        entry_row["borrow_cost"] = borrow_sum
                        fees = float(entry_row.get("fees", 0.0) or 0.0)
                        slip = float(entry_row.get("slippage_cost", 0.0) or 0.0)
                        impact = float(entry_row.get("impact_cost", 0.0) or 0.0)
                        buyin = float(entry_row.get("buyin_penalty_cost", 0.0) or 0.0)
                        total_costs = fees + slip + impact + buyin + borrow_sum
                        entry_row["total_costs"] = float(total_costs)
                        entry_row["net_pnl"] = float(
                            entry_row.get("gross_pnl", 0.0) or 0.0
                        ) - float(total_costs)
                except Exception:
                    borrow_series = None

            open_trades.append(
                {
                    "id": int(cast(Any, entry_row.get("id", 0))),
                    "y_sym": str(entry_row.get("y_symbol"))
                    if entry_row.get("y_symbol") is not None
                    else None,
                    "x_sym": str(entry_row.get("x_symbol"))
                    if entry_row.get("x_symbol") is not None
                    else None,
                    "entry_py": float(entry_row.get("entry_price_y", 0.0)),
                    "entry_px": float(entry_row.get("entry_price_x", 0.0)),
                    "entry_ts": dt,
                    "exit_ts": entry_row.get("exit_date", dt),
                    "y_units": int(entry_row.get("y_units", 0)),
                    "x_units": int(entry_row.get("x_units", 0)),
                    "exit_cost": float(exit_cost),
                    "borrow_series": borrow_series,
                }
            )
            equity -= float(entry_cost)

        # MTM update
        unrealized = _unrealized_open_trades(open_trades, dt, price_for)
        equity += float(unrealized - prev_unrealized)
        prev_unrealized = unrealized

        # Borrow accrual
        if borrow_enabled:
            daily_borrow = 0.0
            for t in open_trades:
                series = t.get("borrow_series")
                if isinstance(series, pd.Series) and not series.empty:
                    v = series.get(pd.Timestamp(dt), 0.0)
                    if v is not None and np.isfinite(float(v)):
                        daily_borrow += float(v)
            if daily_borrow != 0.0:
                equity += float(daily_borrow)
                borrow_total += float(daily_borrow)

        equity_series.at[dt] = float(equity)

    out_df = pd.DataFrame(replay_rows)
    report = {
        "replayed": int(len(out_df)),
        "dropped": int(dropped),
        "settlement_lag_bars": int(settle_lag),
        "borrow_total": float(borrow_total),
        "realized_pnl": float(realized_pnl),
    }
    return equity_series.dropna(), out_df, report


__all__ = ["rescale_trades_stateful", "replay_trades_mtm"]
