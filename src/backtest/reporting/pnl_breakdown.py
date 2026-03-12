from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from backtest.utils.io import _safe_write_df
from backtest.utils.metrics import (
    _percentiles,
    arrival_shortfall_bps,
    decision_shortfall_bps,
    markouts_1_5_30m_bps,
)
from backtest.utils.tz import (
    NY_TZ,
    coerce_series_to_tz,
    ensure_dtindex_tz,
    get_ex_tz,
    get_naive_is_utc,
    to_naive_day,
)

# ------------------------------------------------------------
# TZ policy (defaults -> America/New_York; naive interpreted as UTC)
# ------------------------------------------------------------
_EX_TZ: str = get_ex_tz({}, None, default=NY_TZ)
_NAIVE_IS_UTC: bool = get_naive_is_utc()


def _to_ex_tz_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
    return ensure_dtindex_tz(pd.DatetimeIndex(idx), _EX_TZ, naive_is_utc=_NAIVE_IS_UTC)


def _to_ex_tz_series(s: pd.Series) -> pd.Series:
    # SSOT: timezone parsing/localization strategy lives in utils.tz.
    return coerce_series_to_tz(s, _EX_TZ, naive_is_utc=_NAIVE_IS_UTC, utc_hint="auto")


def _normalize_to_settle_date(ts_like: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """
    Convert to exchange tz, normalize to day, drop tz (naive midnight) for stable daily buckets.
    """
    if isinstance(ts_like, pd.Series):
        idx = _to_ex_tz_series(ts_like)
        return to_naive_day(idx)
    # DatetimeIndex
    di = _to_ex_tz_index(pd.DatetimeIndex(ts_like))
    return pd.Series(to_naive_day(di), index=getattr(ts_like, "index", None))


# ------------------------------------------------------------
# Column resolution / normalization (robust to differing logs)
# ------------------------------------------------------------

_COL_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
    # NOTE: Our primary backtest trade log is pair-level (`trades.csv`) with
    # `entry_date` / `exit_date`. For daily buckets we treat `exit_date` as the
    # natural settlement proxy (timestamp of realized PnL).
    "ts": (
        "ts",
        "timestamp",
        "time",
        "datetime",
        "exec_ts",
        "exit_dt",
        "exit_date",
        "entry_date",
    ),
    "symbol": ("symbol", "pair", "ticker", "sym"),
    "side": ("side", "dir", "buy_sell", "bs"),
    "qty": ("qty", "quantity", "size", "shares", "contracts", "q"),
    "exec_px": ("exec_px", "price_exec", "px_exec", "fill_px", "price"),
    "arrival_px": ("arrival_px", "px_arrival", "ref_px", "mid_at_arrival"),
    "decision_px": ("decision_px", "px_decision", "ref_px_decision"),
    "settle_date": ("settle_date", "settle_dt", "date", "exit_date", "exit_dt"),
    "fees": ("fees", "fee", "commissions", "commission"),
    "borrow_cost": ("borrow_cost", "borrow", "stock_borrow", "sb_cost"),
    "slippage_cost": ("slippage_cost", "slippage", "exec_slippage"),
    "impact_cost": ("impact_cost", "market_impact", "impact"),
    "phase": ("phase", "session_phase", "session"),
    "strategy": ("strategy", "strat", "model"),
    "gross_pnl": ("gross_pnl", "pnl_gross", "pnl_raw"),
    "net_pnl": ("net_pnl", "pnl_net", "pnl"),
    "spread_cost": (
        "spread_cost",
        "spread_pnl",
        "realized_spread",
        "realized_spread_pnl",
    ),
}


def _resolve_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for c in _COL_CANDIDATES.get(key, ()):
        if c in df.columns:
            return c
    return None


def _to_ts_index(df: pd.DataFrame, ts_col: Optional[str]) -> pd.DataFrame:
    """
    Set a DatetimeIndex in the exchange tz. If the index already exists, leave it.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = _to_ex_tz_index(cast(pd.DatetimeIndex, out.index))
        return out
    if ts_col and ts_col in df.columns:
        out = df.copy()
        out[ts_col] = _to_ex_tz_series(out[ts_col])
        out = out.set_index(ts_col)
        return out
    return df


def _side_to_num(s: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.Series(s).astype("object")
    side_map = {"BUY": 1, "SELL": -1, "B": 1, "S": -1, "buy": 1, "sell": -1}
    mapped = arr.map(side_map)
    arr = mapped.where(mapped.notna(), arr)
    return pd.to_numeric(arr, errors="coerce").to_numpy(dtype=float)


# ------------------------------------------------------------
# TCA enrichment (optional; NaN-safe; float32 bps)
# ------------------------------------------------------------


def _add_tca_columns(
    df: pd.DataFrame,
    *,
    ts_col: Optional[str],
    side_col: Optional[str],
    exec_col: Optional[str],
    arrival_col: Optional[str],
    decision_col: Optional[str],
    mid: Optional[pd.Series] = None,
) -> pd.DataFrame:
    out = df.copy()

    if not side_col or not exec_col:
        out["tca_decision_bps"] = np.nan
        out["tca_arrival_bps"] = np.nan
        out["mo_t+1m_bps"] = np.nan
        out["mo_t+5m_bps"] = np.nan
        out["mo_t+30m_bps"] = np.nan
        return out

    side = _side_to_num(out[side_col])
    exec_px = pd.to_numeric(out[exec_col], errors="coerce").to_numpy(dtype=float)

    # decision / arrival shortfall
    if decision_col and decision_col in out.columns:
        dec_px = pd.to_numeric(out[decision_col], errors="coerce").to_numpy(dtype=float)
        out["tca_decision_bps"] = decision_shortfall_bps(side, dec_px, exec_px)
    else:
        out["tca_decision_bps"] = np.nan

    if arrival_col and arrival_col in out.columns:
        arr_px = pd.to_numeric(out[arrival_col], errors="coerce").to_numpy(dtype=float)
        out["tca_arrival_bps"] = arrival_shortfall_bps(side, arr_px, exec_px)
    else:
        out["tca_arrival_bps"] = np.nan

    # markouts need a mid series and a datetime index
    if mid is not None:
        aligned = _to_ts_index(out, ts_col)
        if isinstance(aligned.index, pd.DatetimeIndex):
            # Convert event_times to exchange tz first, then match mid tz (if any)
            evt_idx = _to_ex_tz_index(aligned.index)
            mid_idx = mid.index if isinstance(mid.index, pd.DatetimeIndex) else None
            mid_tz = getattr(getattr(mid_idx, "tz", None), "zone", None) or getattr(
                mid_idx, "tz", None
            )
            if mid_tz is not None and mid_idx is not None:
                evt_idx = ensure_dtindex_tz(
                    evt_idx, str(mid_idx.tz)
                )  # align tz with mid
            mo = markouts_1_5_30m_bps(
                side=side, event_times=evt_idx, mid=mid, how="ffill"
            )

            def _mo_col(name_min: str, name_plain: str) -> np.ndarray:
                if name_min in mo.columns:
                    return mo[name_min].to_numpy(dtype=np.float32)
                if name_plain in mo.columns:
                    return mo[name_plain].to_numpy(dtype=np.float32)
                return np.full(len(out), np.nan, dtype=np.float32)

            out["mo_t+1m_bps"] = _mo_col("t+1m", "t+1")
            out["mo_t+5m_bps"] = _mo_col("t+5m", "t+5")
            out["mo_t+30m_bps"] = _mo_col("t+30m", "t+30")
        else:
            out["mo_t+1m_bps"] = np.nan
            out["mo_t+5m_bps"] = np.nan
            out["mo_t+30m_bps"] = np.nan
    else:
        out["mo_t+1m_bps"] = np.nan
        out["mo_t+5m_bps"] = np.nan
        out["mo_t+30m_bps"] = np.nan

    # enforce dtype
    for c in (
        "tca_decision_bps",
        "tca_arrival_bps",
        "mo_t+1m_bps",
        "mo_t+5m_bps",
        "mo_t+30m_bps",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out


# ------------------------------------------------------------
# Materialize standard cost columns (no dummies)
# ------------------------------------------------------------


def _materialize_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create deterministic standard cost columns:
    ['fees','slippage_cost','impact_cost','borrow_cost',
     'buyin_penalty_cost','exec_emergency_penalty_cost'].
    Detected aliases are reused; missing fields default to 0.0.
    """
    out = df.copy()

    def _ensure(std_name: str, key: str) -> None:
        src = _resolve_col(out, key)
        if src and src in out.columns:
            if src == std_name:
                out[std_name] = pd.to_numeric(out[std_name], errors="coerce").fillna(
                    0.0
                )
            else:
                out[std_name] = pd.to_numeric(out[src], errors="coerce").fillna(0.0)
        else:
            out[std_name] = 0.0

    _ensure("fees", "fees")
    _ensure("slippage_cost", "slippage_cost")
    _ensure("impact_cost", "impact_cost")
    _ensure("borrow_cost", "borrow_cost")
    if "buyin_penalty_cost" not in out.columns:
        out["buyin_penalty_cost"] = 0.0
    if "exec_emergency_penalty_cost" not in out.columns:
        out["exec_emergency_penalty_cost"] = 0.0
    out["buyin_penalty_cost"] = pd.to_numeric(
        out["buyin_penalty_cost"], errors="coerce"
    ).fillna(0.0)
    out["exec_emergency_penalty_cost"] = pd.to_numeric(
        out["exec_emergency_penalty_cost"], errors="coerce"
    ).fillna(0.0)
    return out


def _apply_cost_semantics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in (
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "buyin_penalty_cost",
        "exec_emergency_penalty_cost",
    ):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    diag_only = (
        pd.Series(out.get("exec_diag_costs_only", False), index=out.index)
        .fillna(False)
        .astype(bool)
    )
    slip_realized = out["slippage_cost"].where(~diag_only, 0.0)
    impact_realized = out["impact_cost"].where(~diag_only, 0.0)
    out["execution_diagnostic_costs"] = out["slippage_cost"].where(
        diag_only, 0.0
    ) + out["impact_cost"].where(diag_only, 0.0)
    out["total_costs"] = (
        out["fees"]
        + slip_realized
        + impact_realized
        + out["borrow_cost"]
        + out["buyin_penalty_cost"]
        + out["exec_emergency_penalty_cost"]
    )
    if "gross_pnl" not in out.columns:
        out["gross_pnl"] = pd.to_numeric(
            out.get("net_pnl", 0.0), errors="coerce"
        ).fillna(0.0) - pd.to_numeric(out["total_costs"], errors="coerce").fillna(0.0)
    else:
        out["gross_pnl"] = pd.to_numeric(out["gross_pnl"], errors="coerce").fillna(0.0)
    net_ref = out["gross_pnl"] + out["total_costs"]
    if "net_pnl" not in out.columns:
        out["net_pnl"] = net_ref
    else:
        out["net_pnl"] = pd.to_numeric(out["net_pnl"], errors="coerce").fillna(net_ref)
    return out


# ------------------------------------------------------------
# Aggregations / Exports
# ------------------------------------------------------------


def _agg_cost_components(
    df: pd.DataFrame, settle_col: str, by: Sequence[str]
) -> pd.DataFrame:
    """
    Aggregiert Kostenkomponenten pro Tag (und optional Symbol).
    `slippage_cost` und `impact_cost` bleiben DiagnosekanÃ¤le; `total_costs`
    folgt der Runtime-Semantik via `exec_diag_costs_only`.
    Optional 'spread' is also summed when present (separately; NOT in total_costs).
    """
    df = _apply_cost_semantics(df)

    # spread (optional; neutral=0.0)
    if "spread" not in df.columns:
        spread_src = _resolve_col(df, "spread_cost")
        if spread_src and spread_src in df.columns:
            df["spread"] = pd.to_numeric(df[spread_src], errors="coerce").fillna(0.0)
        else:
            df["spread"] = 0.0

    group_cols = list(by) + [settle_col]
    gb = df.groupby(group_cols, dropna=False)

    agg = gb.agg(
        gross_pnl=("gross_pnl", "sum")
        if "gross_pnl" in df.columns
        else ("net_pnl", "sum"),
        fees=("fees", "sum"),
        slippage_cost=("slippage_cost", "sum"),
        impact_cost=("impact_cost", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        buyin_penalty_cost=("buyin_penalty_cost", "sum"),
        exec_emergency_penalty_cost=("exec_emergency_penalty_cost", "sum"),
        execution_diagnostic_costs=("execution_diagnostic_costs", "sum"),
        spread=("spread", "sum"),
        total_costs=("total_costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
    ).reset_index()

    return agg


def _agg_tca_quantiles(
    df: pd.DataFrame, settle_col: str, by: Sequence[str]
) -> pd.DataFrame:
    tcacols = [
        c
        for c in (
            "tca_arrival_bps",
            "tca_decision_bps",
            "mo_t+1m_bps",
            "mo_t+5m_bps",
            "mo_t+30m_bps",
        )
        if c in df.columns
    ]
    if not tcacols:
        return pd.DataFrame()

    group_cols = list(by) + [settle_col]
    rows: List[Dict[str, Any]] = []

    for keys, chunk in df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys, strict=False))
        for c in tcacols:
            arr = pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=float)
            qmap = _percentiles(arr, (0.5, 0.9, 0.95))
            rows.append(
                {
                    **key_map,
                    "metric": c,
                    "p50": float(qmap.get(0.5, np.nan)),
                    "p90": float(qmap.get(0.9, np.nan)),
                    "p95": float(qmap.get(0.95, np.nan)),
                    "mean": float(np.nanmean(arr))
                    if np.isfinite(arr).any()
                    else np.nan,
                    "count": int(np.isfinite(arr).sum()),
                }
            )

    return pd.DataFrame(rows)


def _write_dual(out_path_noext: Path, df: pd.DataFrame) -> None:
    _safe_write_df(out_path_noext.with_suffix(".parquet"), df)
    _safe_write_df(out_path_noext.with_suffix(".csv"), df)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def generate_pnl_breakdown(
    cfg: Dict[str, Any], out: Path, *, mid_prices: Optional[pd.Series] = None
) -> None:
    """
    Konsolidiert trades_te_*.csv (inkl. wf/trades_te_*.csv) und schreibt:
      - pnl_breakdown_daily.{parquet,csv}
      - pnl_breakdown_symbol.{parquet,csv}
      - pnl_components_daily.{parquet,csv}        (Spread, Impact, Fees, Borrow)
      - pnl_components_symbol.{parquet,csv}
      - pnl_tca_daily.{parquet,csv}               (Arrival/Decision/Markouts p50/p90/p95/mean)
      - pnl_tca_symbol.{parquet,csv}

    Parameter
    ---------
    cfg : Dict[str, Any]
        Unused compatibility argument.
    out : Path
        Zielordner.
    mid_prices : Optional[pd.Series]
        Optional mid-price stream (DatetimeIndex) for markouts.
    """
    out.mkdir(parents=True, exist_ok=True)

    # ---- load trades ----
    trade_paths = sorted(glob.glob(str(out / "trades_te_*.csv"))) + sorted(
        glob.glob(str(out / "wf" / "trades_te_*.csv"))
    )
    # Fallback: single-file runner output (runner_backtest writes `trades.csv` at the run root).
    if not trade_paths:
        for p in (out / "trades.csv", out.parent / "trades.csv"):
            if p.exists():
                trade_paths = [str(p)]
                break
    trades_list: List[pd.DataFrame] = []
    for tp in trade_paths:
        try:
            df = pd.read_csv(tp)
            if not df.empty:
                df["__source_file"] = tp
                trades_list.append(df)
        except Exception:
            continue

    if not trades_list:
        return

    t = pd.concat(trades_list, axis=0, ignore_index=True)

    # ---- column mapping / normalization ----
    sym_col = _resolve_col(t, "symbol") or "__sym"
    if sym_col == "__sym":
        t["__sym"] = "ALL"

    # settle date: ensure exchange-tz day buckets
    settle_col = _resolve_col(t, "settle_date")
    if settle_col and settle_col in t.columns:
        # Coerce any settle-like thing to exchange-day (naive midnight).
        t["settle_date"] = _normalize_to_settle_date(t[settle_col])
        settle_col = "settle_date"
    else:
        ts_col_guess = _resolve_col(t, "ts")
        if ts_col_guess and ts_col_guess in t.columns:
            t["settle_date"] = _normalize_to_settle_date(t[ts_col_guess])
        else:
            t["settle_date"] = pd.NaT
        settle_col = "settle_date"

    gross_col = _resolve_col(t, "gross_pnl") or "gross_pnl"
    net_col = _resolve_col(t, "net_pnl") or "net_pnl"
    if gross_col not in t.columns and net_col in t.columns:
        t[gross_col] = t[net_col]
    if net_col not in t.columns and gross_col in t.columns:
        t[net_col] = t[gross_col]
    if gross_col not in t.columns:
        t[gross_col] = 0.0
    if net_col not in t.columns:
        t[net_col] = 0.0

    # ---- TCA enrichment (optional) ----
    ts_col = _resolve_col(t, "ts")
    side_col = _resolve_col(t, "side")
    exec_col = _resolve_col(t, "exec_px")
    arr_col = _resolve_col(t, "arrival_px")
    dec_col = _resolve_col(t, "decision_px")

    t = _add_tca_columns(
        t,
        ts_col=ts_col,
        side_col=side_col,
        exec_col=exec_col,
        arrival_col=arr_col,
        decision_col=dec_col,
        mid=mid_prices,
    )

    # ---- Kostenkolumnen deterministisch anlegen ----
    t = _materialize_cost_columns(t)
    t = _apply_cost_semantics(t)

    # ---- DAILY / SYMBOL (compat outputs) ----
    daily = (
        t.dropna(subset=[settle_col])
        .groupby(settle_col, as_index=False)
        .agg(
            gross_pnl=("gross_pnl", "sum"),
            fees=("fees", "sum"),
            slippage_cost=("slippage_cost", "sum"),
            impact_cost=("impact_cost", "sum"),
            borrow_cost=("borrow_cost", "sum"),
            buyin_penalty_cost=("buyin_penalty_cost", "sum"),
            exec_emergency_penalty_cost=("exec_emergency_penalty_cost", "sum"),
            execution_diagnostic_costs=("execution_diagnostic_costs", "sum"),
            total_costs=("total_costs", "sum"),
            net_pnl=("net_pnl", "sum"),
            trades=("net_pnl", "count"),
        )
    )

    by_symbol = t.groupby(sym_col, as_index=False).agg(
        gross_pnl=("gross_pnl", "sum"),
        fees=("fees", "sum"),
        slippage_cost=("slippage_cost", "sum"),
        impact_cost=("impact_cost", "sum"),
        borrow_cost=("borrow_cost", "sum"),
        buyin_penalty_cost=("buyin_penalty_cost", "sum"),
        exec_emergency_penalty_cost=("exec_emergency_penalty_cost", "sum"),
        execution_diagnostic_costs=("execution_diagnostic_costs", "sum"),
        total_costs=("total_costs", "sum"),
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "count"),
    )

    _write_dual(out / "pnl_breakdown_daily", daily)
    _write_dual(out / "pnl_breakdown_symbol", by_symbol)

    # ---- Components (Spread/Impact/Fees/Borrow) ----
    spread_src = _resolve_col(t, "spread_cost")
    if spread_src and spread_src in t.columns:
        t["spread"] = pd.to_numeric(t[spread_src], errors="coerce").fillna(0.0)
    else:
        t["spread"] = 0.0

    comp_by_day = _agg_cost_components(t, settle_col=settle_col, by=())
    comp_by_sym = _agg_cost_components(t, settle_col=settle_col, by=(sym_col,))

    ordered = [
        "gross_pnl",
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "buyin_penalty_cost",
        "exec_emergency_penalty_cost",
        "execution_diagnostic_costs",
        "spread",
        "total_costs",
        "net_pnl",
        "trades",
    ]
    comp_by_day = comp_by_day[
        [c for c in comp_by_day.columns if c in ordered]
        + [c for c in comp_by_day.columns if c not in ordered]
    ]
    comp_by_sym = comp_by_sym[
        [c for c in comp_by_sym.columns if c in ([sym_col] + ordered)]
        + [c for c in comp_by_sym.columns if c not in ([sym_col] + ordered)]
    ]

    _write_dual(out / "pnl_components_daily", comp_by_day)
    _write_dual(out / "pnl_components_symbol", comp_by_sym)

    # ---- TCA quantiles (p50/p90/p95/mean) by day / by symbol ----
    tca_daily = _agg_tca_quantiles(t, settle_col=settle_col, by=())
    if not tca_daily.empty:
        _write_dual(out / "pnl_tca_daily", tca_daily)

    tca_by_sym = _agg_tca_quantiles(t, settle_col=settle_col, by=(sym_col,))
    if not tca_by_sym.empty:
        _write_dual(out / "pnl_tca_symbol", tca_by_sym)
