from __future__ import annotations

import logging
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.borrow.accrual import compute_borrow_cost_for_trade_row
from backtest.utils.common.prices import price_at_or_prior as _price_at_or_prior
from backtest.config.types import BorrowCtx
from backtest.utils.be import _infer_exit_column
from backtest.utils.tz import coerce_series_to_tz, to_naive_day

from . import engine_state as _state

logger = logging.getLogger("backtest")


def _asof_price_for_ts(s: pd.Series | None, ts: pd.Timestamp) -> float | None:
    """As-of price on/before ts; returns None if missing."""
    return _price_at_or_prior(s, ts, dropna=True)


def _clip_trades_to_eval_window(
    df: pd.DataFrame,
    *,
    e0: pd.Timestamp,
    e1: pd.Timestamp,
    price_data: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Keep entries inside [e0,e1]; clip exits beyond e1 to e1 and mark hard_exit.
    """
    if df is None or df.empty:
        return df, {"dropped": 0, "hard_exits": 0}
    if "entry_date" not in df.columns or "exit_date" not in df.columns:
        return df, {"dropped": 0, "hard_exits": 0}

    entry_in = (df["entry_date"] >= e0) & (df["entry_date"] <= e1)
    dropped = int((~entry_in).sum()) if len(df) else 0
    out = df.loc[entry_in].copy() if dropped else df.copy()
    if out.empty:
        return out, {"dropped": dropped, "hard_exits": 0}

    too_late = out["exit_date"] > e1
    hard_exits = int(too_late.sum()) if len(out) else 0
    if hard_exits <= 0:
        return out, {"dropped": dropped, "hard_exits": 0}

    if "hard_exit" not in out.columns:
        out["hard_exit"] = False
    if "hard_exit_reason" not in out.columns:
        out["hard_exit_reason"] = ""

    for idx, row in out.loc[too_late].iterrows():
        row_idx = cast(Any, idx)
        out.at[row_idx, "hard_exit"] = True
        cur_reason = str(out.at[row_idx, "hard_exit_reason"] or "")
        if "window_end" not in cur_reason:
            out.at[row_idx, "hard_exit_reason"] = (
                cur_reason + (";" if cur_reason else "") + "window_end"
            )

        out.at[row_idx, "exit_date"] = e1

        # Try to recompute exit prices and gross_pnl for pair trades.
        try:
            y_sym = str(row.get("y_symbol") or "").strip().upper()
            x_sym = str(row.get("x_symbol") or "").strip().upper()
            sig = float(row.get("signal", 1.0) or 1.0)
            sig = 1.0 if sig >= 0 else -1.0

            y_units = row.get("y_units", None)
            x_units = row.get("x_units", None)
            if (
                y_units is None
                or x_units is None
                or pd.isna(y_units)
                or pd.isna(x_units)
            ):
                size = row.get("size", row.get("qty", row.get("quantity", None)))
                qty = float(size) if size is not None and not pd.isna(size) else None
                if qty is not None and np.isfinite(qty) and qty != 0:
                    beta = row.get("beta_entry", row.get("beta", 1.0))
                    beta = (
                        float(beta)
                        if beta is not None and np.isfinite(float(beta))
                        else 1.0
                    )
                    y_units = abs(qty) * (1.0 if sig >= 0 else -1.0)
                    x_units = abs(qty) * (-1.0 if sig >= 0 else 1.0) * beta

            y_units_f = (
                float(y_units) if y_units is not None and not pd.isna(y_units) else None
            )
            x_units_f = (
                float(x_units) if x_units is not None and not pd.isna(x_units) else None
            )

            py0 = row.get("entry_price_y", row.get("price_y", None))
            px0 = row.get("entry_price_x", row.get("price_x", None))
            py0f = float(py0) if py0 is not None and not pd.isna(py0) else None
            px0f = float(px0) if px0 is not None and not pd.isna(px0) else None

            py1 = _asof_price_for_ts(price_data.get(y_sym), e1) if y_sym else None
            px1 = _asof_price_for_ts(price_data.get(x_sym), e1) if x_sym else None

            if py1 is not None:
                out.at[row_idx, "exit_price_y"] = float(py1)
            if px1 is not None:
                out.at[row_idx, "exit_price_x"] = float(px1)

            if (
                y_units_f is not None
                and x_units_f is not None
                and py0f is not None
                and px0f is not None
                and py1 is not None
                and px1 is not None
            ):
                gp = float(y_units_f) * (float(py1) - float(py0f)) + float(
                    x_units_f
                ) * (float(px1) - float(px0f))
                out.at[row_idx, "gross_pnl"] = float(gp)
        except Exception:
            pass

    return out, {"dropped": dropped, "hard_exits": hard_exits}


def _normalize_trades(pair: str, trades: Any) -> pd.DataFrame | None:
    """
    Legacy-safe normalization used in the old engine:
    - infer/alias entry/exit columns BEFORE dropping rows
    - make times tz-aware (exchange tz policy)
    - attach 'pair'
    """
    if trades is None:
        return None

    if isinstance(trades, pd.DataFrame):
        df = trades.copy()
    else:
        try:
            df = pd.DataFrame(trades)
        except Exception:
            logger.warning("%s: trades not a DataFrame and not coercible -> skip", pair)
            return None

    if df.empty:
        return None

    entry_cands = [
        "entry_date",
        "entry_dt",
        "entry_ts",
        "entry_time",
        "entry_datetime",
        "open_date",
        "open_dt",
        "open_ts",
        "open_time",
        "open_datetime",
        "timestamp_entry",
        "exec_entry_ts",
        "start",
        "start_time",
        "start_dt",
    ]
    exit_cands = [
        "exit_date",
        "exit_dt",
        "exit_ts",
        "exit_time",
        "exit_datetime",
        "close_date",
        "close_dt",
        "close_ts",
        "close_time",
        "close_datetime",
        "timestamp_exit",
        "exec_exit_ts",
        "end",
        "end_time",
        "end_dt",
        "fill_ts",
    ]

    def _pick(colnames: list[str]) -> str | None:
        for c in colnames:
            if c in df.columns and not df[c].isna().all():
                return c
        return None

    if "entry_date" not in df.columns or df["entry_date"].isna().all():
        src = _pick(entry_cands)
        if src:
            df["entry_date"] = df[src]
        elif isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.columns[0]: "entry_date"})

    if "exit_date" not in df.columns or df["exit_date"].isna().all():
        src = _pick(exit_cands)
        if src:
            df["exit_date"] = df[src]
        else:
            alt = _infer_exit_column(df)
            if alt and alt in df.columns:
                df["exit_date"] = df[alt]

    for c in ("entry_date", "exit_date"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            s = coerce_series_to_tz(s, _state._EX_TZ, naive_is_utc=_state._NAIVE_IS_UTC)
            df[c] = s

    if "entry_date" in df.columns and "exit_date" in df.columns:
        mask_bad = df["entry_date"].isna() | df["exit_date"].isna()
        if mask_bad.any():
            df = df.loc[~mask_bad].copy()
    else:
        logger.warning("%s: missing entry/exit timestamps -> skip", pair)
        return None

    if df.empty:
        return None

    df["pair"] = str(pair)
    return df


def _recompute_holding_days_inplace(df: pd.DataFrame) -> pd.DataFrame:
    if (
        df is None
        or df.empty
        or "entry_date" not in df.columns
        or "exit_date" not in df.columns
    ):
        return df
    entry = to_naive_day(pd.to_datetime(df["entry_date"], errors="coerce"))
    exit_ = to_naive_day(pd.to_datetime(df["exit_date"], errors="coerce"))

    e_day = entry.dt.normalize().values.astype("datetime64[D]")
    x_day = exit_.dt.normalize().values.astype("datetime64[D]")
    out = np.ones(len(df), dtype=float)
    mask = ~(pd.isna(entry).to_numpy() | pd.isna(exit_).to_numpy())
    if bool(mask.any()):
        try:
            out[mask] = np.maximum(1, np.busday_count(e_day[mask], x_day[mask])).astype(
                float
            )
        except Exception:
            out[mask] = 1.0
    if "exec_entry_status" in df.columns:
        blocked = df["exec_entry_status"].astype(str).str.lower().eq("blocked")
        if bool(blocked.any()):
            out[blocked.to_numpy()] = 0.0
    df["holding_days"] = (
        pd.Series(out, index=df.index, dtype=float).fillna(1.0).astype(int)
    )
    return df


def _finalize_costs_and_net(
    trades_df: pd.DataFrame,
    *,
    calendar: pd.DatetimeIndex,
    price_data: Mapping[str, pd.Series],
    borrow_ctx: BorrowCtx | Any | None,
) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return trades_df

    df = trades_df.copy()

    for c in (
        "gross_pnl",
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "buyin_penalty_cost",
        "exec_emergency_penalty_cost",
        "net_pnl",
    ):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    enabled = False
    if borrow_ctx is not None:
        try:
            enabled = bool(getattr(borrow_ctx, "enabled", True))
        except Exception:
            enabled = True
    if enabled:
        try:
            costs: list[float] = []
            for _, row in df.iterrows():
                costs.append(
                    float(
                        compute_borrow_cost_for_trade_row(
                            row,
                            calendar=calendar,
                            price_data=price_data,
                            borrow_ctx=borrow_ctx,
                        )
                    )
                )
            df["borrow_cost"] = pd.Series(costs, index=df.index, dtype=float).fillna(
                0.0
            )
        except Exception as e:
            logger.warning("borrow_cost compute failed: %s — keep existing.", e)

    for c in (
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "buyin_penalty_cost",
        "exec_emergency_penalty_cost",
    ):
        df[c] = -np.abs(pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float))

    diag_only = (
        pd.Series(df.get("exec_diag_costs_only", False), index=df.index)
        .fillna(False)
        .astype(bool)
    )
    slip_net = df["slippage_cost"].where(~diag_only, 0.0)
    impact_net = df["impact_cost"].where(~diag_only, 0.0)
    df["total_costs"] = (
        df["fees"]
        + slip_net
        + impact_net
        + df["borrow_cost"]
        + df["buyin_penalty_cost"]
        + df["exec_emergency_penalty_cost"]
    ).astype(float)
    df["costs"] = np.abs(df["total_costs"]).astype(float)
    df["net_pnl"] = (df["gross_pnl"] + df["total_costs"]).astype(float)
    return df


def _get_first_present(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except Exception:
                continue
    return None


def _infer_leg_payload(
    row: pd.Series,
) -> tuple[tuple[str | None, str | None], tuple[float, float], float]:
    """
    Infer pair legs and notionals sign-consistently from a heterogeneous row.
    Returns: ((y_sym, x_sym), (ny, nx), gross_notional_abs)
    """
    y_sym = None
    x_sym = None
    for a, b in (
        ("y_symbol", "x_symbol"),
        ("t1_symbol", "t2_symbol"),
        ("leg1_symbol", "leg2_symbol"),
    ):
        if a in row or b in row:
            y_sym = str(row.get(a)) if pd.notna(row.get(a)) else None
            x_sym = str(row.get(b)) if pd.notna(row.get(b)) else None
            break

    sig = _get_first_present(row, ("signal",))
    sig = 1.0 if sig is None else (1.0 if float(sig) >= 0 else -1.0)

    ny = _get_first_present(row, ("notional_y", "leg1_notional", "t1_notional"))
    nx = _get_first_present(row, ("notional_x", "leg2_notional", "t2_notional"))

    if ny is not None and nx is not None:
        return (y_sym, x_sym), (float(ny), float(nx)), abs(float(ny)) + abs(float(nx))

    y_units = _get_first_present(row, ("y_units", "units_y"))
    x_units = _get_first_present(row, ("x_units", "units_x"))
    py = _get_first_present(
        row, ("entry_price_y", "price_y", "y_price", "py", "entry_y")
    )
    px = _get_first_present(
        row, ("entry_price_x", "price_x", "x_price", "px", "entry_x")
    )

    if (
        y_units is not None
        and x_units is not None
        and py is not None
        and px is not None
    ):
        ny = float(y_units) * float(py)
        nx = float(x_units) * float(px)
        return (y_sym, x_sym), (ny, nx), abs(ny) + abs(nx)

    size = _get_first_present(row, ("size", "units", "qty", "quantity"))

    if size is not None and py is not None and px is not None:
        ny = abs(size * py) * (1.0 if sig >= 0 else -1.0)
        nx = abs(size * px) * (-1.0 if sig >= 0 else 1.0)
        return (y_sym, x_sym), (ny, nx), abs(ny) + abs(nx)

    gross = _get_first_present(row, ("gross_notional", "notional", "abs_notional"))
    if gross is None and size is not None:
        p_any = _get_first_present(row, ("entry_price", "price", "px"))
        if p_any is not None:
            gross = abs(size * p_any)
    if gross is not None:
        ny = (gross / 2.0) * (1.0 if sig >= 0 else -1.0)
        nx = (gross / 2.0) * (-1.0 if sig >= 0 else 1.0)
        return (y_sym, x_sym), (ny, nx), float(gross)

    return (y_sym, x_sym), (0.0, 0.0), 0.0


def _restore_essential_columns(
    df_out: pd.DataFrame, base_cols: pd.DataFrame
) -> pd.DataFrame:
    """Re-add essential columns if downstream hooks dropped them."""
    if df_out is None or df_out.empty:
        return df_out
    base = base_cols.reindex(df_out.index)
    for c in base.columns:
        if c not in df_out.columns:
            df_out[c] = base[c]
    return df_out


def _finalize_trade_columns(trades_df: pd.DataFrame) -> pd.DataFrame:
    base_num_cols = [
        "borrow_cost",
        "buyin_penalty_cost",
        "exec_emergency_penalty_cost",
        "fees",
        "slippage_cost",
        "impact_cost",
        "total_costs",
        "costs",
        "net_pnl",
    ]
    legacy_diag_cols = [
        "vx_cost_base_bps",
        "vx_cost_impact_bps",
        "vx_fee_bps",
        "vx_total_cost_bps",
        "vx_notional",
        "vx_notional_adv_ratio",
        "vx_fee_clearing",
        "bar_cost_base_bps",
        "bar_cost_impact_bps",
        "bar_fee_bps",
        "bar_total_cost_bps",
        "bar_notional",
        "bar_filled_notional",
        "bar_notional_adv_ratio",
        "bar_participation",
        "bar_pov_utilization",
        "bar_bucket_utilization",
        "bar_fill_ratio_y",
        "bar_fill_ratio_x",
        "bar_filled_shares_y",
        "bar_filled_shares_x",
    ]
    drop_cols = [c for c in legacy_diag_cols if c in trades_df.columns]
    if drop_cols:
        trades_df = trades_df.drop(columns=drop_cols, errors="ignore")

    for c in base_num_cols:
        if c not in trades_df.columns:
            trades_df[c] = 0.0
        trades_df[c] = (
            pd.to_numeric(trades_df[c], errors="coerce").fillna(0.0).astype(float)
        )

    for flag in ("hard_exit", "risk_blocked"):
        if flag in trades_df.columns:
            col_series = trades_df[flag].infer_objects(copy=False)
            if col_series.dtype != bool:
                col_series = col_series.astype("boolean")
            trades_df[flag] = col_series.fillna(False).astype(bool)
        else:
            trades_df[flag] = False
    if "hard_exit_reason" in trades_df.columns:
        trades_df["hard_exit_reason"] = (
            trades_df["hard_exit_reason"].fillna("").astype(str)
        )
    else:
        trades_df["hard_exit_reason"] = ""

    return trades_df
