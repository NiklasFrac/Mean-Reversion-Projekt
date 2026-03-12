from __future__ import annotations

import pandas as pd
import pytest

from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg
from backtest.strat import baseline
from backtest.strat.baseline import BaselineZScoreStrategy


def test_baseline_strategy_emits_entry_intents_on_signal_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame(
        {
            "y": pd.Series([100.0 + i for i in range(len(idx))], index=idx),
            "x": pd.Series(
                [90.0 + 0.85 * i + (0.6 if i % 2 else -0.4) for i in range(len(idx))],
                index=idx,
            ),
        }
    )

    monkeypatch.setattr(
        baseline,
        "_entry_intents_from_z",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"signal_date": idx[12], "signal": 1, "z_signal": -2.2}]
        ),
    )

    cfg = {
        "backtest": {
            "initial_capital": 100000.0,
            "risk_per_trade": 0.01,
            "execution_lag_bars": 1,
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[9].date())},
                "test": {"start": str(idx[10].date()), "end": str(idx[-1].date())},
            },
        },
        "strategy": {"name": "baseline"},
        "signal": {
            "entry_z": 1.0,
            "exit_z": 0.2,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "volatility_window": 5,
        },
        "spread_zscore": {"z_window": 5, "z_min_periods": 5},
        "risk": {"enabled": False},
        "execution": {"max_participation": 0.0},
    }

    strat = BaselineZScoreStrategy(cfg, borrow_ctx=None)
    out = strat({"AAA-BBB": {"prices": prices, "meta": {"t1": "AAA", "t2": "BBB"}}})

    meta = out["AAA-BBB"]
    intents = meta["intents"]
    state = meta["state"]

    assert "trades" not in meta
    assert pd.Timestamp(intents.loc[0, "signal_date"]) == idx[12]
    assert int(intents.loc[0, "signal"]) == 1
    assert state["pair_key"] == "AAA-BBB"


def test_baseline_intent_engine_applies_next_session_execution_lag() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = {
        "AAA": pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx),
        "BBB": pd.Series([20.0, 21.0, 22.0, 23.0, 24.0], index=idx),
    }
    pair_prices = pd.DataFrame({"y": prices["AAA"], "x": prices["BBB"]})
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
                    }
                ]
            ),
            "state": {
                "pair_key": "AAA-BBB",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "prices": pair_prices,
                "beta": pd.Series(1.0, index=idx, dtype=float),
                "z": pd.Series([-2.5, -2.2, -0.1, 0.0, 0.0], index=idx, dtype=float),
                "sigma": pd.Series(1.0, index=idx, dtype=float),
                "entry_z": 2.0,
                "exit_z": 0.5,
                "stop_z": 3.0,
                "volatility_window": 3,
                "max_hold_days": 10,
                "cooldown_days": 0,
                "test_start": idx[0],
                "entry_end": idx[3],
                "exit_end": idx[4],
                "adv_t1": 1_000_000.0,
                "adv_t2": 1_000_000.0,
            },
        }
    }
    cfg = {
        "backtest": {
            "initial_capital": 100000.0,
            "risk_per_trade": 0.01,
            "execution_lag_bars": 1,
            "splits": {
                "train": {
                    "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                    "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                },
                "test": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            },
        },
        "execution": {"mode": "light", "light": {"fees": {"per_trade": 0.0}}},
        "risk": {"enabled": False},
        "borrow": {"enabled": False},
    }

    stats, trades = backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=prices,
        yaml_cfg=cfg,
        market_data_panel=None,
        adv_map=None,
    )

    assert len(trades) == 1
    assert pd.Timestamp(trades.loc[0, "entry_date"]).date() == idx[1].date()
    assert pd.Timestamp(trades.loc[0, "exit_date"]).date() == idx[2].date()
    assert float(trades.loc[0, "entry_price_y"]) == pytest.approx(11.0)
    assert float(trades.loc[0, "exit_price_y"]) == pytest.approx(12.0)
    assert stats.attrs["NumTrades"] == 1


def test_baseline_strategy_can_map_pair_z_window_to_state_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame(
        {
            "y": pd.Series([100.0 + i for i in range(len(idx))], index=idx),
            "x": pd.Series(
                [90.0 + 0.85 * i + (0.6 if i % 2 else -0.4) for i in range(len(idx))],
                index=idx,
            ),
        }
    )

    monkeypatch.setattr(
        baseline,
        "_entry_intents_from_z",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"signal_date": idx[12], "signal": 1, "z_signal": -2.2}]
        ),
    )

    cfg = {
        "backtest": {
            "initial_capital": 100000.0,
            "risk_per_trade": 0.01,
            "execution_lag_bars": 1,
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[9].date())},
                "test": {"start": str(idx[10].date()), "end": str(idx[-1].date())},
            },
        },
        "strategy": {
            "name": "baseline",
            "pair_z_window_as_volatility_window": True,
        },
        "signal": {
            "entry_z": 1.0,
            "exit_z": 0.2,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "volatility_window": 5,
        },
        "spread_zscore": {"z_window": 5, "z_min_periods": 5},
        "risk": {"enabled": False},
        "execution": {"max_participation": 0.0},
    }

    strat = BaselineZScoreStrategy(cfg, borrow_ctx=None)
    out = strat(
        {
            "AAA-BBB": {
                "prices": prices,
                "meta": {
                    "t1": "AAA",
                    "t2": "BBB",
                    "cointegration": {"z_window": 7},
                },
            }
        }
    )

    state = out["AAA-BBB"]["state"]
    assert state["volatility_window"] == 7
