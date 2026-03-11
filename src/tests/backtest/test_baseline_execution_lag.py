from __future__ import annotations

import pandas as pd
import pytest

from backtest.strat.baseline import BaselineZScoreStrategy
from backtest.strat import baseline


def test_baseline_strategy_applies_next_session_execution_lag(
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

    signal_pos = pd.Series(0, index=idx, dtype=int)
    signal_pos.loc[idx[12]] = 1
    signal_pos.loc[idx[13]] = 1

    monkeypatch.setattr(
        baseline, "_positions_from_z", lambda *_args, **_kwargs: signal_pos.copy()
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

    trades = out["AAA-BBB"]["trades"]
    assert pd.Timestamp(trades.loc[0, "entry_date"]) == idx[13]
    assert pd.Timestamp(trades.loc[0, "exit_date"]) == idx[15]


def test_baseline_strategy_can_map_pair_z_window_to_volatility_window(
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
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        baseline,
        "_positions_from_z",
        lambda *_args, **_kwargs: pd.Series(0, index=idx, dtype=int),
    )

    def _fake_from_signals(**kwargs: object) -> pd.DataFrame:
        captured["cfg"] = kwargs["cfg"]
        return pd.DataFrame()

    monkeypatch.setattr(
        baseline.TradeBuilder, "from_signals", staticmethod(_fake_from_signals)
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
    strat(
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

    cfg_pair = captured["cfg"]
    assert isinstance(cfg_pair, dict)
    assert cfg_pair["signal"]["volatility_window"] == 7
    assert cfg_pair["spread_zscore"]["z_window"] == 7
