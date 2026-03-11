from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.utils.tz import to_naive_day

__all__ = [
    "compute_borrow_cost_for_trade_row",
    "compute_borrow_daily_costs_for_trade_row",
    "compute_borrow_cost_for_trades_df",
    "compute_borrow_meta_for_trade_row",
]


def _to_ts(x: Any) -> pd.Timestamp | None:
    try:
        t = pd.Timestamp(x)
    except Exception:
        return None
    if pd.isna(t):
        return None
    return t


def _busdays_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """
    Business-day count proxy (>=1), robust to tz-aware timestamps.
    Uses numpy busday_count (end exclusive) and clamps to >=1.
    """
    try:
        a0 = to_naive_day(pd.Timestamp(a)).date()
        b0 = to_naive_day(pd.Timestamp(b)).date()
        n = int(np.busday_count(a0, b0))
        return max(1, n if n > 0 else 1)
    except Exception:
        return 1


def _calendar_days_between(
    a: pd.Timestamp, b: pd.Timestamp, *, include_end: bool
) -> int:
    """
    Calendar-day count proxy (>=1), between normalized dates.
    If include_end=True, counts both endpoints.
    """
    try:
        a0 = to_naive_day(pd.Timestamp(a))
        b0 = to_naive_day(pd.Timestamp(b))
        if include_end:
            d = int((b0 - a0).days) + 1
        else:
            d = int((b0 - a0).days)
        return max(1, d if d > 0 else 1)
    except Exception:
        return 1


def _get_first_float(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except Exception:
                continue
    return None


def _infer_short_leg(row: pd.Series) -> tuple[str | None, float]:
    """
    Returns (short_symbol, short_notional_abs).
    Works for pair rows (y/x) and single-asset rows.
    """
    sig = _get_first_float(row, ("signal", "position", "side_sign"))
    sig = 1.0 if sig is None else (1.0 if float(sig) >= 0 else -1.0)

    y_sym = str(row.get("y_symbol")) if pd.notna(row.get("y_symbol")) else None
    x_sym = str(row.get("x_symbol")) if pd.notna(row.get("x_symbol")) else None
    if y_sym:
        y_sym = y_sym.strip().upper()
    if x_sym:
        x_sym = x_sym.strip().upper()

    ny = _get_first_float(row, ("notional_y", "t1_notional", "leg1_notional"))
    nx = _get_first_float(row, ("notional_x", "t2_notional", "leg2_notional"))
    if ny is not None and nx is not None:
        short_sym = x_sym if sig >= 0 else y_sym
        short_notional = abs(nx) if sig >= 0 else abs(ny)
        return (short_sym, float(short_notional))

    gross = _get_first_float(row, ("gross_notional", "notional", "abs_notional"))
    if gross is not None and np.isfinite(gross) and gross > 0:
        short_sym = x_sym if sig >= 0 else y_sym
        return (short_sym, float(abs(gross) / 2.0))

    # single-asset schema fallback
    sym = row.get("symbol") or row.get("ticker") or row.get("asset")
    sym_s = str(sym).strip().upper() if sym is not None else None
    notional = _get_first_float(row, ("notional", "gross_notional"))
    if notional is not None and np.isfinite(notional) and notional > 0:
        return (sym_s, float(notional))

    size = _get_first_float(row, ("size", "units", "qty", "quantity"))
    px = _get_first_float(
        row, ("entry_price", "price", "px", "entry_price_y", "entry_price_x")
    )
    if size is not None and px is not None and np.isfinite(px) and px > 0:
        return (sym_s, float(abs(size) * abs(px)))

    return (sym_s, 0.0)


def _resolve_rate(borrow_ctx: Any, symbol: str, day: pd.Timestamp) -> float:
    if borrow_ctx is None:
        return 0.0
    resolve_fn = getattr(borrow_ctx, "resolve_borrow_rate", None)
    if callable(resolve_fn):
        try:
            r = resolve_fn(symbol, day)
            return float(r) if r is not None else 0.0
        except Exception:
            return 0.0
    try:
        return float(getattr(borrow_ctx, "default_rate_annual", 0.0))
    except Exception:
        return 0.0


def _day_basis(borrow_ctx: Any) -> int:
    try:
        return int(getattr(borrow_ctx, "day_basis", 252))
    except Exception:
        return 252


def _get_borrow_cfg(borrow_ctx: Any) -> tuple[str, str, bool, int]:
    """
    (accrual_mode, day_count, include_exit_day, min_days)
    """
    try:
        accrual_mode = (
            str(
                getattr(borrow_ctx, "accrual_mode", "entry_notional")
                or "entry_notional"
            )
            .strip()
            .lower()
        )
    except Exception:
        accrual_mode = "entry_notional"
    try:
        day_count = (
            str(getattr(borrow_ctx, "day_count", "busdays") or "busdays")
            .strip()
            .lower()
        )
    except Exception:
        day_count = "busdays"
    try:
        include_exit_day = bool(getattr(borrow_ctx, "include_exit_day", False))
    except Exception:
        include_exit_day = False
    try:
        min_days = int(getattr(borrow_ctx, "min_days", 1))
    except Exception:
        min_days = 1
    return accrual_mode, day_count, include_exit_day, max(0, min_days)


def _asof_price(px: pd.Series, day: pd.Timestamp) -> float | None:
    """
    Price asof normalized day (past-only). Returns None if not available.
    """
    if px is None or not isinstance(px, pd.Series) or px.empty:
        return None
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(px.index, errors="coerce"))
        idx = to_naive_day(idx)
        s = (
            pd.Series(pd.to_numeric(px.to_numpy(), errors="coerce"), index=idx)
            .dropna()
            .sort_index()
        )
        if s.empty:
            return None
        d = to_naive_day(pd.Timestamp(day))
        pos = int(s.index.searchsorted(d, side="right") - 1)
        if pos < 0:
            return None
        val = float(s.iloc[pos])
        return val if np.isfinite(val) and val > 0 else None
    except Exception:
        return None


def _infer_short_symbol_and_units(row: pd.Series) -> tuple[str | None, int | None]:
    """
    Best-effort: return (short_symbol, short_units_abs).
    Uses explicit y_units/x_units when present; otherwise falls back to (symbol, None).
    """
    y_sym = str(row.get("y_symbol")) if pd.notna(row.get("y_symbol")) else None
    x_sym = str(row.get("x_symbol")) if pd.notna(row.get("x_symbol")) else None
    if y_sym:
        y_sym = y_sym.strip().upper()
    if x_sym:
        x_sym = x_sym.strip().upper()

    yu = row.get("y_units")
    xu = row.get("x_units")
    try:
        yu_i = int(round(float(yu))) if yu is not None and pd.notna(yu) else None
    except Exception:
        yu_i = None
    try:
        xu_i = int(round(float(xu))) if xu is not None and pd.notna(xu) else None
    except Exception:
        xu_i = None

    # If we have signed units, pick the negative leg as short.
    if yu_i is not None and xu_i is not None:
        if yu_i < 0 and xu_i >= 0:
            return (y_sym, abs(int(yu_i)))
        if xu_i < 0 and yu_i >= 0:
            return (x_sym, abs(int(xu_i)))

    # Single-leg hint: sometimes only one leg exists in the row.
    if xu_i is not None and xu_i < 0 and x_sym:
        return (x_sym, abs(int(xu_i)))
    if yu_i is not None and yu_i < 0 and y_sym:
        return (y_sym, abs(int(yu_i)))

    # Single-asset fallback
    sym = row.get("symbol") or row.get("ticker") or row.get("asset")
    sym_s = str(sym).strip().upper() if sym is not None else None
    return (sym_s, None)


def _build_day_schedule(
    *,
    entry_day: pd.Timestamp,
    exit_day: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    day_count: str,
    include_exit_day: bool,
    min_days: int,
) -> pd.DatetimeIndex:
    """
    Deterministic daily schedule for borrow accrual.
    Returns a non-empty DatetimeIndex if min_days>0.
    """
    entry_d = to_naive_day(pd.Timestamp(entry_day))
    exit_d = to_naive_day(pd.Timestamp(exit_day))
    end_d = exit_d if include_exit_day else (exit_d - pd.Timedelta(days=1))

    sched = pd.DatetimeIndex([])
    if day_count == "calendar_days":
        try:
            sched = pd.date_range(entry_d, end_d, freq="D")
        except Exception:
            sched = pd.DatetimeIndex([])
    elif day_count == "sessions":
        try:
            cal = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
            cal = to_naive_day(cal)
            sched = cal[(cal >= entry_d) & (cal <= end_d)]
        except Exception:
            sched = pd.DatetimeIndex([])
    else:
        # "busdays": approximate using pandas business-days
        try:
            sched = pd.bdate_range(entry_d, end_d, freq="B")
        except Exception:
            sched = pd.DatetimeIndex([])

    if sched.empty and min_days > 0:
        sched = pd.DatetimeIndex([entry_d])
    return sched


def compute_borrow_meta_for_trade_row(
    row: pd.Series,
    *,
    calendar: pd.DatetimeIndex,
    price_data: Mapping[str, pd.Series] | None,
    borrow_ctx: Any,
) -> dict[str, Any]:
    """
    Compute deterministic borrow metadata for reporting/audits.
    Does not mutate inputs.
    """
    out: dict[str, Any] = {
        "enabled": bool(getattr(borrow_ctx, "enabled", True))
        if borrow_ctx is not None
        else False,
        "accrual_mode": None,
        "day_count": None,
        "include_exit_day": None,
        "min_days": None,
        "day_basis": None,
        "short_symbol": None,
        "short_units_abs": None,
        "rate_entry": None,
        "n_days": None,
        "mtm_price_used": False,
        "mtm_missing_price_days": 0,
    }
    if borrow_ctx is None or row is None:
        return out

    accrual_mode, day_count, include_exit_day, min_days = _get_borrow_cfg(borrow_ctx)
    out.update(
        {
            "accrual_mode": accrual_mode,
            "day_count": day_count,
            "include_exit_day": bool(include_exit_day),
            "min_days": int(min_days),
            "day_basis": int(_day_basis(borrow_ctx)),
        }
    )

    entry = _to_ts(row.get("entry_date"))
    exit_ = _to_ts(row.get("exit_date"))
    if entry is None or exit_ is None:
        return out

    # For daily borrow we only need normalized dates; keep tz semantics stable.
    try:
        entry_d = to_naive_day(pd.Timestamp(entry))
        exit_d = to_naive_day(pd.Timestamp(exit_))
    except Exception:
        return out

    short_sym_units = _infer_short_symbol_and_units(row)
    short_sym = short_sym_units[0]
    short_units_abs = short_sym_units[1]
    out["short_symbol"] = short_sym
    out["short_units_abs"] = (
        int(short_units_abs) if short_units_abs is not None else None
    )

    if short_sym:
        out["rate_entry"] = float(_resolve_rate(borrow_ctx, short_sym, entry_d))

    # Days count only (used for report sanity).
    sched0 = _build_day_schedule(
        entry_day=entry_d,
        exit_day=exit_d,
        calendar=calendar,
        day_count=day_count,
        include_exit_day=bool(include_exit_day),
        min_days=min_days,
    )
    out["n_days"] = (
        int(max(min_days, int(len(sched0))))
        if len(sched0) > 0
        else int(max(min_days, 0))
    )

    # MTM availability info (optional)
    if (
        accrual_mode == "mtm_daily"
        and short_sym
        and short_units_abs
        and price_data is not None
    ):
        px = price_data.get(short_sym)
        if isinstance(px, pd.Series) and not px.empty:
            out["mtm_price_used"] = True
            sched = sched0
            miss = 0
            for d in sched:
                if _asof_price(px, pd.Timestamp(d)) is None:
                    miss += 1
            out["mtm_missing_price_days"] = int(miss)
    return out


def compute_borrow_cost_for_trade_row(
    row: pd.Series,
    *,
    calendar: pd.DatetimeIndex,
    price_data: Mapping[str, pd.Series] | None,
    borrow_ctx: Any,
) -> float:
    """
    Deterministic borrow accrual proxy.

    - Charges only the short leg notional (if inferable).
    - Default legacy mode: uses a single annual rate (as-of entry date) and business-day count.
    - Optional paper-default: MTM daily accrual on calendar-days using last close as-of each day.
    - Returns a negative cash cost (<= 0).
    """
    if borrow_ctx is None or row is None:
        return 0.0

    entry = _to_ts(row.get("entry_date"))
    exit_ = _to_ts(row.get("exit_date"))
    if entry is None or exit_ is None:
        return 0.0

    # Normalize to daily (borrow accrual is a daily ledger).
    try:
        entry_d = to_naive_day(pd.Timestamp(entry))
        exit_d = to_naive_day(pd.Timestamp(exit_))
    except Exception:
        return 0.0

    accrual_mode, day_count, include_exit_day, min_days = _get_borrow_cfg(borrow_ctx)
    basis = _day_basis(borrow_ctx)
    if basis <= 0:
        basis = 252

    short_sym, short_notional_entry = _infer_short_leg(row)
    short_sym_u, short_units_abs = _infer_short_symbol_and_units(row)
    if short_sym is None and short_sym_u is not None:
        short_sym = short_sym_u

    if not short_sym:
        return 0.0

    # Resolve entry-rate first (used for legacy mode and as fallback).
    rate_entry = _resolve_rate(borrow_ctx, short_sym, entry_d)
    if not (np.isfinite(rate_entry) and rate_entry > 0):
        return 0.0

    # -------- Legacy: entry-notional × n_days --------------------------------
    if accrual_mode in {"entry_notional", "simple", "legacy"}:
        days = _get_first_float(row, ("holding_days",))
        if days is not None and np.isfinite(days) and days > 0:
            n_days = int(max(min_days, round(float(days))))
        else:
            if day_count == "calendar_days":
                n_days = _calendar_days_between(
                    entry_d, exit_d, include_end=bool(include_exit_day)
                )
            else:
                n_days = _busdays_between(entry_d, exit_d)
            n_days = int(max(min_days, n_days))

        if not (np.isfinite(short_notional_entry) and short_notional_entry > 0):
            return 0.0
        cost = (
            -(float(rate_entry) / float(basis))
            * float(n_days)
            * float(short_notional_entry)
        )
        return float(cost) if np.isfinite(cost) else 0.0

    # -------- Paper-default: MTM daily accrual --------------------------------
    # Determine the day schedule first.
    sched = _build_day_schedule(
        entry_day=entry_d,
        exit_day=exit_d,
        calendar=calendar,
        day_count=day_count,
        include_exit_day=bool(include_exit_day),
        min_days=min_days,
    )
    if sched.empty:
        return 0.0

    # Prefer MTM notional if we have units + prices; otherwise fall back to entry notional.
    px_series = (
        None if price_data is None else price_data.get(str(short_sym).strip().upper())
    )
    can_mtm = (
        short_units_abs is not None
        and isinstance(px_series, pd.Series)
        and not px_series.empty
    )

    total = 0.0
    for d in sched:
        rate_d = _resolve_rate(borrow_ctx, short_sym, pd.Timestamp(d))
        if not (np.isfinite(rate_d) and rate_d > 0):
            continue

        notional_d = None
        if can_mtm:
            px_series_t = cast(pd.Series, px_series)
            px = _asof_price(px_series_t, pd.Timestamp(d))
            if px is not None:
                units_abs = int(cast(int, short_units_abs))
                notional_d = float(abs(units_abs)) * float(px)

        if notional_d is None:
            # fallback: constant entry notional
            if not (np.isfinite(short_notional_entry) and short_notional_entry > 0):
                continue
            notional_d = float(short_notional_entry)

        total += -(float(rate_d) / float(basis)) * float(notional_d)

    return float(total) if np.isfinite(total) else 0.0


def compute_borrow_daily_costs_for_trade_row(
    row: pd.Series,
    *,
    calendar: pd.DatetimeIndex,
    price_data: Mapping[str, pd.Series] | None,
    borrow_ctx: Any,
) -> pd.Series:
    """
    Daily borrow accrual series (<=0) for one trade.
    Returns empty Series if borrow is not applicable.
    """
    if borrow_ctx is None or row is None:
        return pd.Series(dtype=float)

    entry = _to_ts(row.get("entry_date"))
    exit_ = _to_ts(row.get("exit_date"))
    if entry is None or exit_ is None:
        return pd.Series(dtype=float)

    try:
        entry_d = to_naive_day(pd.Timestamp(entry))
        exit_d = to_naive_day(pd.Timestamp(exit_))
    except Exception:
        return pd.Series(dtype=float)

    accrual_mode, day_count, include_exit_day, min_days = _get_borrow_cfg(borrow_ctx)
    basis = _day_basis(borrow_ctx)
    if basis <= 0:
        basis = 252

    short_sym, short_notional_entry = _infer_short_leg(row)
    short_sym_u, short_units_abs = _infer_short_symbol_and_units(row)
    if short_sym is None and short_sym_u is not None:
        short_sym = short_sym_u

    if not short_sym:
        return pd.Series(dtype=float)

    rate_entry = _resolve_rate(borrow_ctx, short_sym, entry_d)
    if not (np.isfinite(rate_entry) and rate_entry > 0):
        return pd.Series(dtype=float)

    if accrual_mode in {"entry_notional", "simple", "legacy"}:
        days = _get_first_float(row, ("holding_days",))
        if days is not None and np.isfinite(days) and days > 0:
            n_days = int(max(min_days, round(float(days))))
        else:
            if day_count == "calendar_days":
                n_days = _calendar_days_between(
                    entry_d, exit_d, include_end=bool(include_exit_day)
                )
            else:
                n_days = _busdays_between(entry_d, exit_d)
            n_days = int(max(min_days, n_days))

        if not (np.isfinite(short_notional_entry) and short_notional_entry > 0):
            return pd.Series(dtype=float)

        daily_cost = -(float(rate_entry) / float(basis)) * float(short_notional_entry)
        try:
            if day_count == "calendar_days":
                sched = pd.date_range(entry_d, periods=n_days, freq="D")
            else:
                sched = pd.bdate_range(entry_d, periods=n_days, freq="B")
        except Exception:
            sched = pd.DatetimeIndex([])

        if sched.empty and min_days > 0:
            sched = pd.DatetimeIndex([entry_d])
        return pd.Series([daily_cost] * len(sched), index=sched, dtype=float)

    sched = _build_day_schedule(
        entry_day=entry_d,
        exit_day=exit_d,
        calendar=calendar,
        day_count=day_count,
        include_exit_day=bool(include_exit_day),
        min_days=min_days,
    )
    if sched.empty:
        return pd.Series(dtype=float)

    px_series = (
        None if price_data is None else price_data.get(str(short_sym).strip().upper())
    )
    can_mtm = (
        short_units_abs is not None
        and isinstance(px_series, pd.Series)
        and not px_series.empty
    )

    out = []
    for d in sched:
        rate_d = _resolve_rate(borrow_ctx, short_sym, pd.Timestamp(d))
        if not (np.isfinite(rate_d) and rate_d > 0):
            out.append(0.0)
            continue

        notional_d = None
        if can_mtm:
            px_series_t = cast(pd.Series, px_series)
            px = _asof_price(px_series_t, pd.Timestamp(d))
            if px is not None:
                units_abs = int(cast(int, short_units_abs))
                notional_d = float(abs(units_abs)) * float(px)

        if notional_d is None:
            if not (np.isfinite(short_notional_entry) and short_notional_entry > 0):
                out.append(0.0)
                continue
            notional_d = float(short_notional_entry)

        out.append(-(float(rate_d) / float(basis)) * float(notional_d))

    return pd.Series(out, index=sched, dtype=float)


def compute_borrow_cost_for_trades_df(
    trades_df: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    price_data: Mapping[str, pd.Series] | None,
    borrow_ctx: Any,
) -> pd.Series:
    """
    Convenience wrapper returning a Series aligned to trades_df.index (<=0).
    """
    if trades_df is None or trades_df.empty or borrow_ctx is None:
        return pd.Series(
            0.0,
            index=getattr(trades_df, "index", None),
            name="borrow_cost",
            dtype=float,
        )
    out: list[float] = []
    for _, row in trades_df.iterrows():
        out.append(
            float(
                compute_borrow_cost_for_trade_row(
                    row,
                    calendar=calendar,
                    price_data=price_data,
                    borrow_ctx=borrow_ctx,
                )
            )
        )
    return pd.Series(
        out, index=trades_df.index, name="borrow_cost", dtype=float
    ).fillna(0.0)


# Back-compat alias (older engine import path).
_compute_borrow_cost_for_trade_row = compute_borrow_cost_for_trade_row
