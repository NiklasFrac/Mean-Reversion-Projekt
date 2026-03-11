from dataclasses import replace

import pandas as pd

from backtest.config.cfg import BacktestConfig
from backtest.simulators import engine


def test_backtest_portfolio_sets_defaults_and_logs_drops() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 11.0], index=idx),
        "BBB": pd.Series([20.0, 20.5, 21.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "y_symbol": ["AAA", "AAA"],
            "x_symbol": ["BBB", "BBB"],
            "notional_y": [100.0, 100.0],
            "notional_x": [100.0, 100.0],
        }
    )
    portfolio = {"AAA-BBB": {"trades": trades}}
    cfg = replace(
        BacktestConfig(
            splits={"test": {"start": "2024-01-02", "end": "2024-01-04"}},
        ),
        calendar_mapping=None,
        raw_yaml=None,
        exec_mode="skip",
    )

    stats, trades_out = engine.backtest_portfolio(portfolio, price_data, cfg=cfg)
    assert "equity" in stats.columns
    assert not trades_out.empty
