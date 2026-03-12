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


def test_backtest_portfolio_baseline_intents_size_zero_has_no_ghost_episode() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    price_data = {
        "AAA": pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx),
        "BBB": pd.Series([20.0, 21.0, 22.0, 23.0, 24.0], index=idx),
    }
    pair_prices = pd.DataFrame({"y": price_data["AAA"], "x": price_data["BBB"]})
    portfolio = {
        "AAA-BBB": {
            "intents": pd.DataFrame(
                [
                    {
                        "pair": "AAA-BBB",
                        "signal_date": idx[0],
                        "signal": 1,
                        "z_signal": -2.5,
                        "entry_end": idx[3],
                        "exit_end": idx[4],
                    },
                    {
                        "pair": "AAA-BBB",
                        "signal_date": idx[2],
                        "signal": 1,
                        "z_signal": -2.4,
                        "entry_end": idx[3],
                        "exit_end": idx[4],
                    },
                ]
            ),
            "state": {
                "pair_key": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "prices": pair_prices,
                "beta": pd.Series(1.0, index=idx, dtype=float),
                "z": pd.Series([-2.5, 0.0, -2.4, -0.1, 0.0], index=idx, dtype=float),
                "sigma": pd.Series(1.0, index=idx, dtype=float),
                "entry_z": 2.0,
                "exit_z": 0.5,
                "stop_z": 3.0,
                "max_hold_days": 10,
                "cooldown_days": 2,
                "test_start": idx[0],
                "entry_end": idx[3],
                "exit_end": idx[4],
                "adv_t1": 1_000_000.0,
                "adv_t2": 1_000_000.0,
            },
        }
    }
    cfg = replace(
        BacktestConfig(
            initial_capital=100_000.0,
            splits={
                "train": {
                    "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                    "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                },
                "test": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            },
            exec_mode="light",
            raw_yaml={
                "backtest": {
                    "initial_capital": 100_000.0,
                    "risk_per_trade": 0.0,
                    "execution_lag_bars": 1,
                    "splits": {
                        "train": {
                            "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                            "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                        },
                        "test": {
                            "start": str(idx[0].date()),
                            "end": str(idx[-1].date()),
                        },
                    },
                },
                "execution": {"mode": "light", "light": {"fees": {"per_trade": 0.0}}},
                "risk": {"enabled": False, "risk_per_trade": 0.0},
                "borrow": {"enabled": False},
            },
        ),
        calendar_mapping=None,
    )

    stats, trades_out = engine.backtest_portfolio(portfolio, price_data, cfg=cfg)

    transitions = stats.attrs["state_transitions_df"]
    assert trades_out.empty
    assert list(transitions["reason"]) == ["size_zero", "size_zero"]
    assert "signal_during_cooldown" not in set(transitions["reason"])
