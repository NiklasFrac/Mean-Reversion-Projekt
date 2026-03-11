from __future__ import annotations

import pandas as pd

from backtest.optimize.paper_bo_parts import sim as bo_sim
from backtest.run.trade_builder import (
    TradeBuilder,
    _per_unit_stop_risk,
    _remaining_stop_distance_z,
)
from backtest.strat import baseline


def test_baseline_entry_allows_band_without_stop_collision() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-1.0, -2.5, -2.4, 0.0], index=idx)

    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
    )

    assert pos.tolist() == [0, 1, 1, 0]


def test_baseline_entry_rejects_stop_region_until_fresh_cross() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-1.0, -3.0, -1.0, -2.5], index=idx)

    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
    )

    assert pos.tolist() == [0, 0, 0, 1]


def test_bo_sim_entry_rejects_stop_region() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([1.0, 3.0, 1.0, 2.5], index=idx)

    pos = bo_sim._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
    )

    assert pos.tolist() == [0, 0, 0, -1]


def test_remaining_stop_distance_matches_actual_signal_gap() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    z_cache = pd.Series([2.5, 3.0], index=idx, dtype=float)

    assert (
        _remaining_stop_distance_z(
            decision_ts=idx[0],
            z_cache=z_cache,
            entry_z_abs=2.0,
            stop_z_abs=3.0,
        )
        == 0.5
    )
    assert (
        _remaining_stop_distance_z(
            decision_ts=idx[1],
            z_cache=z_cache,
            entry_z_abs=2.0,
            stop_z_abs=3.0,
        )
        is None
    )


def test_per_unit_stop_risk_rejects_non_positive_gap() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    sigma = pd.Series([1.5], index=idx, dtype=float)

    assert (
        _per_unit_stop_risk(
            decision_ts=idx[0],
            sigma_cache=sigma,
            stop_gap_z=0.0,
        )
        is None
    )


def test_trade_builder_skips_trade_when_decision_z_is_in_stop_region() -> None:
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        {
            "y": pd.Series(
                [100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 107.0, 109.0, 108.0],
                index=idx,
            ),
            "x": pd.Series(
                [50.0, 50.4, 50.8, 50.6, 51.1, 51.0, 51.4, 51.8, 52.1, 52.0], index=idx
            ),
        }
    )
    signals = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 0], index=idx, dtype=int)
    beta = pd.Series(1.0, index=idx, dtype=float)
    z_cache = pd.Series(0.0, index=idx, dtype=float)
    sigma_cache = pd.Series(1.0, index=idx, dtype=float)
    z_cache.loc[idx[7]] = 3.0
    cfg = {
        "backtest": {"execution_lag_bars": 0},
        "signal": {"entry_z": 2.0, "stop_z": 3.0, "volatility_window": 5},
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

    assert trades.empty


def test_trade_builder_preserves_valid_non_collision_entry() -> None:
    idx = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        {
            "y": pd.Series(
                [100.0, 101.0, 103.0, 102.0, 105.0, 104.0, 106.0, 107.0, 109.0, 108.0],
                index=idx,
            ),
            "x": pd.Series(
                [50.0, 50.4, 50.8, 50.6, 51.1, 51.0, 51.4, 51.8, 52.1, 52.0], index=idx
            ),
        }
    )
    signals = pd.Series([0, 0, 0, 0, 0, 0, 0, -1, -1, 0], index=idx, dtype=int)
    beta = pd.Series(1.0, index=idx, dtype=float)
    z_cache = pd.Series(0.0, index=idx, dtype=float)
    sigma_cache = pd.Series(1.0, index=idx, dtype=float)
    z_cache.loc[idx[7]] = 2.5
    cfg = {
        "backtest": {"execution_lag_bars": 0},
        "signal": {"entry_z": 2.0, "stop_z": 3.0, "volatility_window": 5},
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

    assert len(trades) == 1
    assert int(trades.loc[0, "size"]) > 0
    assert pd.Timestamp(trades.loc[0, "decision_date"]) == idx[7]


def test_trade_builder_drops_trade_when_raw_risk_budget_is_below_one_unit() -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "y": pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=idx),
            "x": pd.Series([50.0, 50.5, 51.0, 51.5, 52.0, 52.5], index=idx),
        }
    )
    signals = pd.Series([0, 0, 0, 1, 1, 0], index=idx, dtype=int)
    beta = pd.Series(1.0, index=idx, dtype=float)
    z_cache = pd.Series(0.0, index=idx, dtype=float)
    sigma_cache = pd.Series(10.0, index=idx, dtype=float)
    z_cache.loc[idx[3]] = 2.95
    cfg = {
        "backtest": {"execution_lag_bars": 0},
        "signal": {"entry_z": 2.0, "stop_z": 3.0, "volatility_window": 5},
        "risk": {"risk_per_trade": 0.01, "max_trade_pct": 1.0},
        "execution": {"max_participation": 0.0},
        "_z_cache": z_cache,
        "_sigma_cache": sigma_cache,
    }

    trades = TradeBuilder.from_signals(
        signals=signals,
        prices=prices,
        beta=beta,
        capital=100.0,
        cfg=cfg,
    )

    assert trades.empty
