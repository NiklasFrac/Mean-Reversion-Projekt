import numpy as np
import pandas as pd

from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg


def test_window_end_clips_and_marks_hard_exit() -> None:
    idx = pd.bdate_range("2024-01-02", periods=8, freq="B")
    tz = "America/New_York"
    idx = idx.tz_localize(tz)
    a = pd.Series(np.arange(len(idx), dtype=float) + 100.0, index=idx, name="AAA")
    b = pd.Series(np.arange(len(idx), dtype=float) + 200.0, index=idx, name="BBB")

    entry_i = 2
    end_i = 5
    entry_dt = idx[entry_i]
    planned_exit_dt = idx[-1]  # beyond eval window end
    e1 = idx[end_i]

    trades = pd.DataFrame(
        {
            "entry_date": [entry_dt],
            "exit_date": [planned_exit_dt],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
            "entry_price_y": [float(a.loc[entry_dt])],
            "entry_price_x": [float(b.loc[entry_dt])],
        }
    )
    portfolio = {"AAA/BBB": {"trades": trades}}
    price_data = {"AAA": a, "BBB": b}

    yaml_cfg = {
        "data": {"calendar_name": "XNYS"},
        "backtest": {
            "initial_capital": 100000.0,
            "calendar_mapping": "prior",
            "settlement_lag_bars": 0,
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[1].date())},
                "test": {"start": str(idx[2].date()), "end": str(e1.date())},
            },
        },
        "execution": {"mode": "lob", "lob": {}},
        "borrow": {"enabled": False},
    }

    _, out = backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio, price_data=price_data, yaml_cfg=yaml_cfg
    )
    assert len(out) == 1
    r = out.iloc[0]

    assert pd.Timestamp(r["exit_date"]) == pd.Timestamp(e1)
    assert bool(r.get("hard_exit", False)) is True
    assert "window_end" in str(r.get("hard_exit_reason", ""))

    exp_gp = float(r["lob_gross_pnl"])
    assert np.isclose(float(r["gross_pnl"]), float(exp_gp), rtol=0, atol=1e-9)
