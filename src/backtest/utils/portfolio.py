# src/backtest/utils/portfolio.py
from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from backtest.utils.tz import to_ny_series, to_ny_timestamp


def _norm_side(v: Any) -> Optional[str]:
    """Normalize trade direction to BUY/SELL."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("buy", "long", "b", "1", "+1", "true", "t"):
        return "BUY"
    if s in ("sell", "short", "s", "-1", "0", "false", "f"):
        return "SELL"
    try:
        return "BUY" if float(v) > 0 else "SELL"
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


def _df_trades_to_orders_df(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pair trades into a canonical 4-order representation.

    For each trade row this emits:
    - Y entry
    - X entry
    - Y exit
    - X exit
    """
    columns = ["dt", "symbol", "side", "qty", "price", "pair", "leg"]
    if trades is None or trades.empty:
        return pd.DataFrame(columns=columns)

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

    for c in (c_entry_dt, c_exit_dt):
        try:
            d[c] = to_ny_series(pd.to_datetime(d[c], errors="coerce"))
        except Exception:
            pass

    rows: list[dict[str, Any]] = []
    for _, r in d.dropna(subset=[c_entry_dt, c_exit_dt]).iterrows():
        qty = _coerce_int(r.get(c_size, 0), 0)
        if qty <= 0:
            continue

        y_side_in = _norm_side(r.get(c_sig, 0)) or "BUY"
        x_side_in = "SELL" if y_side_in == "BUY" else "BUY"
        y_side_out = "SELL" if y_side_in == "BUY" else "BUY"
        x_side_out = "BUY" if x_side_in == "SELL" else "SELL"

        pair = str(r.get(c_pair, "PAIR"))
        y_sym = str(r.get(c_y, "Y"))
        x_sym = str(r.get(c_x, "X"))
        t_in = to_ny_timestamp(pd.to_datetime(r[c_entry_dt]))
        t_out = to_ny_timestamp(pd.to_datetime(r[c_exit_dt]))

        rows.extend(
            [
                {
                    "dt": t_in,
                    "symbol": y_sym,
                    "side": y_side_in,
                    "qty": qty,
                    "price": _coerce_float(r.get(c_epy, np.nan)),
                    "pair": pair,
                    "leg": "Y",
                },
                {
                    "dt": t_in,
                    "symbol": x_sym,
                    "side": x_side_in,
                    "qty": qty,
                    "price": _coerce_float(r.get(c_epx, np.nan)),
                    "pair": pair,
                    "leg": "X",
                },
                {
                    "dt": t_out,
                    "symbol": y_sym,
                    "side": y_side_out,
                    "qty": qty,
                    "price": _coerce_float(r.get(c_xpy, np.nan)),
                    "pair": pair,
                    "leg": "Y",
                },
                {
                    "dt": t_out,
                    "symbol": x_sym,
                    "side": x_side_out,
                    "qty": qty,
                    "price": _coerce_float(r.get(c_xpx, np.nan)),
                    "pair": pair,
                    "leg": "X",
                },
            ]
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    odf = pd.DataFrame(rows)
    odf["dt"] = to_ny_series(pd.to_datetime(odf["dt"], errors="coerce"))
    odf["symbol"] = odf["symbol"].astype(str).str.upper()
    odf["side"] = odf["side"].map(_norm_side)
    odf["qty"] = odf["qty"].apply(lambda v: _coerce_int(v, 0))
    odf["price"] = odf["price"].apply(lambda v: _coerce_float(v, np.nan))

    odf = odf.dropna(subset=["dt", "symbol", "side", "price"])
    odf = odf[odf["qty"] > 0]
    if odf.empty:
        return pd.DataFrame(columns=columns)

    odf = odf.sort_values("dt", kind="mergesort").reset_index(drop=True)
    return odf[columns]
