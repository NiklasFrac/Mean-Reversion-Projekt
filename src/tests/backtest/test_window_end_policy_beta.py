import pandas as pd

from backtest.simulators.engine_trades import _clip_trades_to_eval_window


def test_hard_exit_uses_beta_for_units_when_missing() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York")
    y = pd.Series([100, 101, 102, 110, 111], index=idx)
    x = pd.Series([50, 49, 48, 40, 39], index=idx)

    trades = pd.DataFrame(
        {
            "entry_date": [idx[0]],
            "exit_date": [idx[-1]],  # beyond window end
            "signal": [1],
            "size": [10],
            "beta_entry": [2.0],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "y_symbol": ["Y"],
            "x_symbol": ["X"],
            "gross_pnl": [0.0],
        }
    )

    e0 = idx[0]
    e1 = idx[3]  # force exit on this date (price y=110, x=40)
    out, rep = _clip_trades_to_eval_window(
        trades,
        e0=e0,
        e1=e1,
        price_data={"Y": y.rename("Y"), "X": x.rename("X")},
    )

    assert rep["hard_exits"] == 1
    gp = float(out.iloc[0]["gross_pnl"])
    # Expected: y_units=10, x_units=-20 (beta=2)
    # gross_pnl = 10*(110-100) + (-20)*(40-50) = 100 + 200 = 300
    assert gp == 300.0
