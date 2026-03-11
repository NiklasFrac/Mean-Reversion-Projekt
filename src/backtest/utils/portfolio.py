# src/backtest/utils/portfolio.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np
import pandas as pd

# Single Source of Truth for timezones
from backtest.utils.tz import NY_TZ, coerce_series_to_tz, coerce_ts_to_tz

# =============================================================================
# TZ helpers (values vs. index)
# =============================================================================


def _to_ny_ts(x: Any) -> pd.Timestamp:
    """
    Convert a single timestamp into America/New_York (tz-aware).

    Rules:
    - tz-naive → interpret as local wall clock in NY and tz_localize(NY_TZ)
    - tz-aware → tz_convert(NY_TZ)
    """
    return coerce_ts_to_tz(x, NY_TZ)


def _to_ny_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series of timestamps into America/New_York (tz-aware).

    Notes:
    - Uses .dt because we convert the *values*, not the index.
    - tz-naive values are localized to NY_TZ; tz-aware values are converted.
    """
    return coerce_series_to_tz(s, NY_TZ)


# =============================================================================
# Optional: anchor enrichment for MOC workflows
# =============================================================================


def _enrich_orders_for_anchor_blob(
    blob: dict,
    entry_anchor: str,
    exit_anchor: str,
    cal_index: pd.DatetimeIndex | list[Any] | tuple[Any, ...],
) -> dict:
    """
    Mutate a canonical orders blob to carry next_close/MOC metadata when needed.

    - Adds: anchor="next_close", order_type="MOC", tif="DAY", price=None
    - Drops orders at/after the last calendar day (cannot be filled under next_close).
    - Safe no-op if structure doesn't match.

    This is intentionally tolerant and side-effectful (mutation by design).
    """
    try:
        need_close = (entry_anchor == "next_close") or (exit_anchor == "next_close")
        if not need_close or not isinstance(blob, dict):
            return blob

        # Resolve the last calendar day as NY-aware
        last_day_raw = cal_index[-1] if len(cal_index) else None
        last_day = _to_ny_ts(last_day_raw) if last_day_raw is not None else None

        def _iter_orders_lists():
            if "orders" in blob and isinstance(blob["orders"], list):
                yield blob["orders"]
            else:
                for _, v in blob.items():
                    if isinstance(v, Mapping):
                        od = v.get("orders")
                        if isinstance(od, list):
                            yield od

        for lst in _iter_orders_lists():
            for o in lst:
                if not isinstance(o, dict):
                    continue
                o["anchor"] = "next_close"
                o["order_type"] = "MOC"
                o["tif"] = "DAY"
                o["price"] = None  # MOC orders don't carry a limit

                # Drop orders at/after last calendar session
                dt = o.get("dt")
                if dt is None or last_day is None:
                    continue
                ts = _to_ny_ts(pd.to_datetime(dt, errors="coerce"))
                if pd.notna(ts) and ts >= last_day:
                    o["__drop__"] = True

        # Purge dropped
        for lst in _iter_orders_lists():
            keep = [o for o in lst if not o.get("__drop__", False)]
            lst[:] = keep
        return blob
    except Exception:
        return blob


# =============================================================================
# Local helpers (domain-agnostic)
# =============================================================================


def _norm_side(v: Any) -> Optional[str]:
    """
    Normalize side to 'BUY' or 'SELL'.
    Accepts {'BUY','SELL','buy','sell', 1,-1,'1','-1','long','short'}.
    """
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("buy", "long", "b", "1", "+1", "true", "t"):
        return "BUY"
    if s in ("sell", "short", "s", "-1", "0", "false", "f"):
        return "SELL"
    # Numeric fallback
    try:
        f = float(v)
        return "BUY" if f > 0 else "SELL"
    except Exception:
        return None


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(np.floor(float(v)))
    except Exception:
        return default


def _coerce_float(v: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


# =============================================================================
# Trades → Orders mapping (pairs → 4 plain orders)
# =============================================================================


def _df_trades_to_orders_df(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Map a 2-leg pairs trade row into 4 orders:
      - Y entry, X entry, Y exit, X exit

    Contract:
      - Output 'dt' is strictly tz-aware (America/New_York).
      - Prices may be NaN (later filtered if needed).
      - 'symbol' upper-cased; 'side' normalized to {'BUY','SELL'}.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=["dt", "symbol", "side", "qty", "price", "pair", "leg"]
        )

    d = trades.copy()
    lc = {c.lower(): c for c in d.columns}

    def _col(*cands: str, req: bool = False) -> Optional[str]:
        for c in cands:
            if c in d.columns:
                return c
            if c in lc:
                return lc[c]
        if req:
            raise KeyError(f"Missing required column among {cands}")
        return None

    c_entry_dt = cast(
        str,
        _col("entry_date", "entry", "open_date", "open", "timestamp_entry", req=True),
    )
    c_exit_dt = cast(
        str,
        _col(
            "exit_date",
            "exit",
            "close_date",
            "close",
            "exit_dt",
            "close_dt",
            "timestamp_exit",
            req=True,
        ),
    )
    c_sig = _col("signal", "side", "dir")
    c_size = _col("size", "qty", "quantity", "units")
    c_pair = _col("pair")
    c_y = _col("y_symbol", "t1", "asset1", "y")
    c_x = _col("x_symbol", "t2", "asset2", "x")
    c_epy = _col("entry_price_y", "open_y", "y_entry", "y_open")
    c_epx = _col("entry_price_x", "open_x", "x_entry", "x_open")
    c_xpy = _col("exit_price_y", "close_y", "y_exit", "y_close")
    c_xpx = _col("exit_price_x", "close_x", "x_exit", "x_close")

    # Normalize datetime columns to NY tz-aware values
    for c in (c_entry_dt, c_exit_dt):
        if c and c in d.columns:
            try:
                d[c] = _to_ny_series(pd.to_datetime(d[c], errors="coerce"))
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for _, r in d.dropna(subset=[c_entry_dt, c_exit_dt]).iterrows():
        qty = _coerce_int(r.get(c_size, 0), 0)
        if qty <= 0:
            continue

        sig = r.get(c_sig, 0)
        side_norm = _norm_side(sig)
        if side_norm is None:
            try:
                side_norm = "BUY" if float(sig) > 0 else "SELL"
            except Exception:
                side_norm = "BUY"

        pair = str(r.get(c_pair, "PAIR"))

        y_sym = str(r.get(c_y, "Y"))
        x_sym = str(r.get(c_x, "X"))

        t_in = _to_ny_ts(pd.to_datetime(r[c_entry_dt]))
        t_out = _to_ny_ts(pd.to_datetime(r[c_exit_dt]))

        y_in_px = _coerce_float(r.get(c_epy, np.nan))
        x_in_px = _coerce_float(r.get(c_epx, np.nan))
        y_out_px = _coerce_float(r.get(c_xpy, np.nan))
        x_out_px = _coerce_float(r.get(c_xpx, np.nan))

        # entry legs
        y_side_in = side_norm
        x_side_in = "SELL" if y_side_in == "BUY" else "BUY"
        rows.append(
            {
                "dt": t_in,
                "symbol": y_sym,
                "side": y_side_in,
                "qty": qty,
                "price": y_in_px,
                "pair": pair,
                "leg": "Y",
            }
        )
        rows.append(
            {
                "dt": t_in,
                "symbol": x_sym,
                "side": x_side_in,
                "qty": qty,
                "price": x_in_px,
                "pair": pair,
                "leg": "X",
            }
        )
        # exit legs (flatten)
        y_side_out = "SELL" if y_side_in == "BUY" else "BUY"
        x_side_out = "BUY" if x_side_in == "SELL" else "SELL"
        rows.append(
            {
                "dt": t_out,
                "symbol": y_sym,
                "side": y_side_out,
                "qty": qty,
                "price": y_out_px,
                "pair": pair,
                "leg": "Y",
            }
        )
        rows.append(
            {
                "dt": t_out,
                "symbol": x_sym,
                "side": x_side_out,
                "qty": qty,
                "price": x_out_px,
                "pair": pair,
                "leg": "X",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["dt", "symbol", "side", "qty", "price", "pair", "leg"]
        )

    odf = pd.DataFrame(rows)

    # Canonicalize & validate
    odf["dt"] = _to_ny_series(pd.to_datetime(odf["dt"], errors="coerce"))
    odf["symbol"] = odf["symbol"].astype(str).str.upper()
    odf["side"] = odf["side"].map(_norm_side)
    odf["qty"] = odf["qty"].apply(lambda v: _coerce_int(v, 0))
    odf["price"] = odf["price"].apply(lambda v: _coerce_float(v, np.nan))

    # Drop bad / zero lines (MOC with price=None will be dropped here by design)
    odf = odf.dropna(subset=["dt", "symbol", "side", "price"])
    odf = odf[odf["qty"] > 0]

    if odf.empty:
        return pd.DataFrame(
            columns=["dt", "symbol", "side", "qty", "price", "pair", "leg"]
        )

    odf = odf.sort_values("dt", kind="mergesort").reset_index(drop=True)
    return odf[["dt", "symbol", "side", "qty", "price", "pair", "leg"]]


def _orders_df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert normalized orders DataFrame into JSON-safe records."""
    if df is None or df.empty:
        return []
    recs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        price_val = r["price"]
        price_out = None if pd.isna(price_val) else float(price_val)
        recs.append(
            {
                "dt": (
                    pd.Timestamp(r["dt"]).isoformat() if pd.notna(r["dt"]) else None
                ),
                "symbol": str(r["symbol"]),
                "side": str(r["side"]),
                "qty": int(r["qty"]),
                "price": price_out,
                "pair": str(r.get("pair", "")),
                "leg": str(r.get("leg", "")),
            }
        )
    return recs


# =============================================================================
# Public API: normalize strategy output → portfolio blob
# =============================================================================


def _prepare_portfolio_all(
    strat_out: Dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Normalize Strategy-Output into a flat portfolio blob:

        { "orders": [ {dt, symbol, side, qty, price, pair, leg}, ... ] }

    Accepts either already-provided orders (DataFrame or list of dicts)
    OR per-pair trades DataFrames and maps those into orders.

    Signature-compat:
    Ignores extra args/kwargs (e.g. borrow_ctx, availability_df, ...).
    """
    _ = args  # unused
    _ = kwargs  # unused

    orders_frames: List[pd.DataFrame] = []

    for pair, meta in (strat_out or {}).items():
        if not isinstance(meta, Mapping):
            continue

        # 1) Strategy already provides orders (DataFrame)
        od = meta.get("orders")
        if isinstance(od, pd.DataFrame) and not od.empty:
            tmp = od.copy()

            # dt column → NY tz-aware
            if "dt" in tmp:
                tmp["dt"] = _to_ny_series(pd.to_datetime(tmp["dt"], errors="coerce"))

            if "symbol" in tmp:
                tmp["symbol"] = tmp["symbol"].astype(str).str.upper()
            if "side" in tmp:
                tmp["side"] = tmp["side"].map(_norm_side)
            if "pair" not in tmp:
                tmp["pair"] = str(pair)

            orders_frames.append(tmp)
            continue

        # 1b) Strategy provides orders (list of dicts)
        if isinstance(od, list) and od:
            try:
                tmp = pd.DataFrame(od)
                if "pair" not in tmp:
                    tmp["pair"] = str(pair)
                if "dt" in tmp:
                    tmp["dt"] = _to_ny_series(
                        pd.to_datetime(tmp["dt"], errors="coerce")
                    )
                orders_frames.append(tmp)
                continue
            except Exception:
                pass

        # 2) Otherwise: Trades → Orders
        df = meta.get("trades")
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Normalize known datetime columns to NY tz-aware
            for c in (
                "entry_date",
                "exit_date",
                "entry",
                "exit",
                "open_date",
                "close_date",
                "timestamp_entry",
                "timestamp_exit",
            ):
                if c in df.columns:
                    try:
                        df[c] = _to_ny_series(pd.to_datetime(df[c], errors="coerce"))
                    except Exception:
                        pass

            odf = _df_trades_to_orders_df(df)

            # Be defensive: any potential datetime-like columns get normalized
            for col in (
                "dt",
                "entry_date",
                "exit_date",
                "timestamp",
                "order_ts",
                "fill_ts",
            ):
                if col in odf.columns:
                    odf[col] = _to_ny_series(pd.to_datetime(odf[col], errors="coerce"))

            if not odf.empty:
                if "pair" not in odf.columns or odf["pair"].isna().all():
                    odf["pair"] = str(pair)
                orders_frames.append(odf)

    # 3) Merge & produce JSON-safe records
    if not orders_frames:
        return {"orders": []}

    full = (
        pd.concat(orders_frames, ignore_index=True)
        if len(orders_frames) > 1
        else orders_frames[0]
    )

    # Final canonicalization (idempotent)
    need_cols = ["dt", "symbol", "side", "qty", "price", "pair", "leg"]
    for c in need_cols:
        if c not in full.columns:
            if c in ("pair", "leg"):
                full[c] = ""
            else:
                full[c] = np.nan

    full["dt"] = _to_ny_series(pd.to_datetime(full["dt"], errors="coerce"))
    full["symbol"] = full["symbol"].astype(str).str.upper()
    full["side"] = full["side"].map(_norm_side)
    full["qty"] = full["qty"].apply(lambda v: _coerce_int(v, 0))
    full["price"] = full["price"].apply(lambda v: _coerce_float(v, np.nan))
    full["pair"] = full["pair"].astype(str)
    full["leg"] = full["leg"].astype(str)

    # Filters per original semantics: drop rows lacking price or qty<=0
    full = full.dropna(subset=["dt", "symbol", "side", "price"])
    full = full[full["qty"] > 0]
    if full.empty:
        return {"orders": []}

    full = full.sort_values("dt", kind="mergesort").reset_index(drop=True)
    return {"orders": _orders_df_to_records(full)}
