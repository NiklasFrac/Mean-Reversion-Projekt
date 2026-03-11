import pandas as pd

from backtest.simulators.stateful import rescale_trades_stateful


def test_rescale_trades_stateful_scales_with_mtm_equity() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 12.0, 12.0, 12.0], index=idx),
        "BBB": pd.Series([10.0, 10.0, 10.0, 10.0], index=idx),
    }

    trades = pd.DataFrame(
        [
            {
                "pair": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "entry_date": idx[0],
                "exit_date": idx[2],
                "signal": 1,
                "size": 10,
                "beta_entry": 1.0,
                "entry_price_y": 10.0,
                "entry_price_x": 10.0,
                "exit_price_y": 12.0,
                "exit_price_x": 10.0,
                "entry_capital_base": 100.0,
                "wf_i": 0,
            },
            {
                "pair": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "entry_date": idx[1],
                "exit_date": idx[3],
                "signal": 1,
                "size": 10,
                "beta_entry": 1.0,
                "entry_price_y": 12.0,
                "entry_price_x": 10.0,
                "exit_price_y": 12.0,
                "exit_price_x": 10.0,
                "entry_capital_base": 100.0,
                "wf_i": 0,
            },
        ]
    )

    resized, rep = rescale_trades_stateful(
        trades,
        price_data=price_data,
        initial_capital=100.0,
        wf_params={0: {"max_trade_pct": 10.0, "max_participation": 0.0}},
    )

    assert rep["resized"] == 2
    assert rep["dropped"] == 0
    assert int(resized.loc[0, "size"]) == 10
    assert int(resized.loc[1, "size"]) == 12
    assert float(resized.loc[1, "sizing_equity"]) == 120.0


def test_rescale_trades_stateful_respects_settlement_lag() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 12.0, 12.0, 12.0], index=idx),
        "BBB": pd.Series([10.0, 10.0, 10.0, 10.0], index=idx),
    }

    trades = pd.DataFrame(
        [
            {
                "pair": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "entry_date": idx[0],
                "exit_date": idx[1],
                "signal": 1,
                "size": 10,
                "beta_entry": 1.0,
                "entry_price_y": 10.0,
                "entry_price_x": 10.0,
                "exit_price_y": 12.0,
                "exit_price_x": 10.0,
                "entry_capital_base": 100.0,
                "wf_i": 0,
            },
            {
                "pair": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "entry_date": idx[2],
                "exit_date": idx[3],
                "signal": 1,
                "size": 10,
                "beta_entry": 1.0,
                "entry_price_y": 12.0,
                "entry_price_x": 10.0,
                "exit_price_y": 12.0,
                "exit_price_x": 10.0,
                "entry_capital_base": 100.0,
                "wf_i": 0,
            },
        ]
    )

    resized, rep = rescale_trades_stateful(
        trades,
        price_data=price_data,
        initial_capital=100.0,
        wf_params={0: {"max_trade_pct": 10.0, "max_participation": 0.0}},
        settlement_lag_bars=2,
    )

    assert rep["resized"] == 2
    assert rep["dropped"] == 0
    assert int(resized.loc[1, "size"]) == 10
    assert float(resized.loc[1, "sizing_equity"]) == 100.0
