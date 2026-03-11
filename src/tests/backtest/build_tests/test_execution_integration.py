# src/backtest/src/test_execution_integration.py
import math

import numpy as np
import pandas as pd
from backtest_engine import backtest_portfolio_with_yaml_cfg


def _toy_prices(n_days=12):
    idx = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # Y goes up a bit, X down a bit => long Y / short X should profit
    y = pd.Series(np.linspace(100.0, 103.0, n_days), index=idx, name="AAA")
    x = pd.Series(np.linspace(50.0, 48.0, n_days), index=idx, name="BBB")
    return {"AAA": y, "BBB": x}, idx


def _trade_row(idx, entry_i=2, exit_i=6, signal=+1):
    """Builds a single pair trade using index positions."""
    entry_dt = idx[entry_i]
    exit_dt = idx[exit_i]
    return {
        "entry_date": entry_dt,
        "exit_date": exit_dt,
        "y_symbol": "AAA",
        "x_symbol": "BBB",
        "signal": signal,
        # hint prices (used by LOB to recenter; harmless in vectorized)
        "entry_price_y": float(np.nan),  # let LOB take series fallback (more realistic)
        "entry_price_x": float(np.nan),
        "exit_price_y": float(np.nan),
        "exit_price_x": float(np.nan),
        "pair": "AAA/BBB",
    }


def _portfolio(df_trades):
    # backtest_engine expects: mapping pair -> {"trades": DataFrame}
    return {"AAA/BBB": {"trades": df_trades}}


def test_vectorized_execution_end_to_end():
    price_data, idx = _toy_prices()
    tr = _trade_row(idx)

    # For vectorized path the engine uses net_pnl if present, else falls back to gross_pnl - cost_per_trade
    # Provide gross_pnl so equity will move deterministically.
    y_entry = float(price_data["AAA"].loc[tr["entry_date"]])
    y_exit = float(price_data["AAA"].loc[tr["exit_date"]])
    x_entry = float(price_data["BBB"].loc[tr["entry_date"]])
    x_exit = float(price_data["BBB"].loc[tr["exit_date"]])
    gross_pnl = (y_exit - y_entry) + (x_entry - x_exit)  # +1: long y / short x

    df_trades = pd.DataFrame([tr])
    df_trades["gross_pnl"] = float(gross_pnl)

    yaml_cfg = {
        "backtest": {
            "initial_capital": 100000.0,
            "run_mode": "aktuell",  # avoids explicit splits
            "calendar_mapping": "prior",
            "settlement_lag_bars": 0,
        },
        "execution": {
            "mode": "vectorized",
            # optional vectorized controls (safe to ignore if not implemented)
            "use_exec_model": True,
        },
        "costs": {
            "per_trade": 0.0,
            "slippage": 0.0,
            # optional vectorized param block (harmless if unused)
            "execution": {
                "base_slippage": 0.00015,
                "sqrt_coefficient": 0.0,
                "adv_impact_k": 0.5,
                "max_slippage": 0.02,
                "min_fee": 0.0,
                "impact_model": "sqrt",
                "fill_prob_k": 3.0,
                "min_fill_prob": 0.05,
            },
        },
    }

    stats, trades = backtest_portfolio_with_yaml_cfg(
        _portfolio(df_trades), price_data, yaml_cfg
    )

    # 1) No LOB columns in vectorized path
    assert not any(c.startswith("exec_") for c in trades.columns), (
        "Vectorized path must not add LOB exec_* columns"
    )

    # 2) Equity moved by exactly the mapped gross_pnl (no double costs)
    eq0 = yaml_cfg["backtest"]["initial_capital"]
    eq_end = float(stats["equity"].iloc[-1])
    expected = eq0 + float(gross_pnl)
    assert math.isfinite(eq_end) and abs(eq_end - expected) < 1e-6, (
        f"Equity mismatch: got {eq_end}, expected {expected}"
    )

    # 3) Basic sanity
    assert "returns" in stats.columns and len(stats["equity"]) == len(stats["returns"])
    # num trades: 1 exit within window
    assert int(stats.attrs.get("mapped_trades", 1)) >= 1


def test_lob_execution_end_to_end():
    price_data, idx = _toy_prices()
    tr = _trade_row(idx)
    df_trades = pd.DataFrame([tr])

    yaml_cfg = {
        "backtest": {
            "initial_capital": 100000.0,
            "run_mode": "aktuell",
            "calendar_mapping": "prior",
            "settlement_lag_bars": 0,
        },
        "execution": {
            "mode": "lob",
            "lob": {
                "tick": 0.01,
                "levels": 5,
                "size_per_level": 1000,
                "min_spread_ticks": 1,
                "lam": 2.0,
                "max_add": 500,
                "bias_top": 0.7,
                "cancel_prob": 0.15,
                "max_cancel": 500,
                "steps_per_day": 4.0,
                "post_costs": {},  # keep simple here
            },
        },
        "costs": {
            "per_trade": 0.0,
            "slippage": 0.0,
        },
    }

    stats, trades = backtest_portfolio_with_yaml_cfg(
        _portfolio(df_trades), price_data, yaml_cfg
    )

    # 1) LOB columns must be present and finite
    needed = [
        "exec_entry_vwap_y",
        "exec_exit_vwap_y",
        "exec_entry_vwap_x",
        "exec_exit_vwap_x",
        "lob_net_pnl",
    ]
    for c in needed:
        assert c in trades.columns, f"Missing LOB column: {c}"
        v = float(trades[c].iloc[0])
        assert math.isfinite(v), f"LOB column {c} is not finite"

    # 2) net_pnl must be set (override=True) and finite
    assert "net_pnl" in trades.columns
    npnl = float(trades["net_pnl"].iloc[0])
    assert math.isfinite(npnl), "LOB net_pnl should be finite"

    # 3) Equity must reflect mapped LOB PnL (no double costs unless explicitly configured)
    eq0 = yaml_cfg["backtest"]["initial_capital"]
    eq_end = float(stats["equity"].iloc[-1])
    # We can't assert exact value (LOB is simulated), but it should move by ~net_pnl on the exit day
    assert math.isfinite(eq_end)
    # At least check sign consistency: equity should change by same sign as net_pnl (unless zero)
    assert (npnl == 0.0) or ((eq_end - eq0) * npnl > -1e-9), (
        "Equity change inconsistent with LOB net_pnl sign"
    )

    # 4) Basic sanity
    assert int(stats.attrs.get("mapped_trades", 1)) >= 1
