from __future__ import annotations

import pandas as pd

from backtest.run.trade_builder import TradeBuilder
from backtest.simulators import engine
from backtest.simulators import stateful


def test_trade_builder_preserves_negative_beta_sign_in_x_units() -> None:
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        {
            "y": pd.Series(
                [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0],
                index=idx,
            ),
            "x": pd.Series(
                [50.0, 50.5, 51.0, 51.5, 52.0, 51.5, 51.0, 50.5, 50.0, 49.5], index=idx
            ),
        }
    )
    signals = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], index=idx, dtype=int)
    beta = pd.Series(-2.0, index=idx, dtype=float)
    z_cache = pd.Series(0.0, index=idx, dtype=float)
    sigma_cache = pd.Series(1.0, index=idx, dtype=float)
    z_cache.loc[idx[6]] = 1.5
    cfg = {
        "backtest": {"execution_lag_bars": 0},
        "signal": {"entry_z": 1.0, "stop_z": 3.0, "volatility_window": 5},
        "risk": {"risk_per_trade": 0.01, "max_trade_pct": 1.0},
        "execution": {"max_participation": 0.0},
        "_z_cache": z_cache,
        "_sigma_cache": sigma_cache,
    }

    trades = TradeBuilder.from_signals(
        signals=signals,
        prices=prices,
        beta=beta,
        capital=100000.0,
        cfg=cfg,
    )

    assert not trades.empty
    assert float(trades.loc[0, "beta_entry"]) == -2.0
    assert float(trades.loc[0, "y_units"]) > 0.0
    assert float(trades.loc[0, "x_units"]) > 0.0


def test_stateful_unit_inference_preserves_negative_beta_sign() -> None:
    y_units, x_units = stateful._infer_units(size=5, signal=1, beta=-1.5)
    assert y_units == 5
    assert x_units > 0


def test_risk_gating_respects_same_sign_legs_for_negative_beta() -> None:
    df = pd.DataFrame(
        {
            "pair": ["AAA-BBB"],
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-05")],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "beta_entry": [-1.5],
            "notional_y": [1_000.0],
            "notional_x": [1_500.0],
            "y_units": [10.0],
            "x_units": [15.0],
            "entry_price_y": [100.0],
            "entry_price_x": [100.0],
        }
    )
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=10_000.0,
        risk_cfg={
            "max_trade_pct": 1.0,
            "max_gross_exposure": 10.0,
            "max_net_exposure": 0.2,
        },
        price_data={},
    )

    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 1
    else:
        assert rep["blocked"] == 1
        assert out.empty
