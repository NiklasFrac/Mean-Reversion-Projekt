from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtest.config import (
    BOConfig,
    BOCVConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    DataConfig,
    MarkovConfig,
    OutputConfig,
    PairSelectionConfig,
    RiskConfig,
    StrategyConfig,
    WalkforwardConfig,
)
from backtest.markov import markov_gate
from backtest.optimize import blocked_folds
from backtest.pair_selection import select_pairs
from backtest.run import run_config
from backtest.engine import run_engine
from backtest.strategy import (
    WindowPlan,
    build_continuous_signals,
    estimate_beta,
    positions_from_z,
    rolling_zscore,
)
from backtest.walkforward import Window, make_windows


def test_beta_and_zscore_are_past_only() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    x = pd.Series([1, 2, 3, 4, 5], index=idx)
    y = 1 + 2 * x
    assert np.isclose(estimate_beta(y, x), 2.0)

    spread = pd.Series([1.0, 2.0, 3.0, 100.0], index=idx[:4])
    z = rolling_zscore(spread, window=2, min_periods=2)
    assert np.isclose(z.iloc[-1], (100.0 - 2.5) / 0.5)


def test_positions_entry_exit_stop_hold_and_cooldown() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    cfg = StrategyConfig(
        entry_z=1.0, exit_z=0.2, stop_z=3.0, max_hold_days=3, cooldown_days=1
    )
    z = pd.Series([-1.2, -0.8, 0.1, -0.5, -1.5, 0.0], index=idx)
    assert positions_from_z(z, cfg).tolist() == [1, 1, 0, 0, 1, 0]


def test_markov_disabled_allows_entries_and_enabled_can_block() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    z = pd.Series([-1.2, -1.1, -1.3, -1.2, -1.4, -1.2, -1.1, -1.3], index=idx)
    train, test = idx[:6], idx[6:]

    assert markov_gate(
        z, train, test, MarkovConfig(enabled=False), entry_z=1.0, exit_z=0.0
    ).all()
    gate = markov_gate(
        z,
        train,
        test,
        MarkovConfig(
            enabled=True,
            min_train_observations=2,
            min_state_observations=1,
            min_revert_prob=0.99,
        ),
        entry_z=1.0,
        exit_z=0.0,
    )
    assert not gate.any()


def test_blocked_folds_are_contiguous_and_disjoint() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    folds = blocked_folds(idx, BOConfig(cv=BOCVConfig(n_blocks=5, k_test_blocks=2)))
    assert folds
    for train, test in folds:
        assert train.intersection(test).empty
        assert test.equals(pd.date_range(test[0], test[-1], freq="D"))


def test_walkforward_rolling_and_expanding_do_not_overlap_tests() -> None:
    idx = pd.date_range("2020-01-01", periods=900, freq="D")
    windows = make_windows(idx, BacktestConfig())
    assert all(
        windows[i].test_end < windows[i + 1].test_start for i in range(len(windows) - 1)
    )
    expanding = make_windows(
        idx, BacktestConfig(walkforward=WalkforwardConfig(train_mode="expanding"))
    )
    assert expanding[0].train_start == expanding[1].train_start


def test_pair_selection_uses_train_window_filters_and_ranking() -> None:
    prices = _cointegrated_prices()
    pairs, metrics = select_pairs(
        prices,
        Window(
            0, prices.index[0], prices.index[199], prices.index[200], prices.index[-1]
        ),
        PairSelectionConfig(
            min_obs=100,
            min_corr=0.50,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=60.0,
            max_hurst=0.90,
            max_pairs=1,
        ),
    )
    assert len(pairs) == 1
    assert {"pair", "corr", "eg_pvalue", "half_life", "hurst"}.issubset(metrics)
    assert metrics["eg_pvalue"].iloc[0] <= 0.20


def test_run_requires_contiguous_test_windows(tmp_path) -> None:
    prices = _cointegrated_prices(periods=160)
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index_label="date")
    cfg = Config(
        data=DataConfig(prices_path=prices_path),
        backtest=BacktestConfig(
            walkforward=WalkforwardConfig(test_months=1, step_months=2)
        ),
        output=OutputConfig(dir=tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="step_months"):
        run_config(cfg)


def test_continuous_signals_hold_old_trade_across_window_and_final_exit() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(prices, first_max_hold=100, second_pairs=False)
    sig = build_continuous_signals(
        prices, plans, RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )
    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[5], pair] == 1
    assert sig.positions.at[prices.index[7], pair] == 1
    assert sig.positions.at[prices.index[-1], pair] == 0

    result = run_engine(
        prices,
        {pair: ("AAA", "BBB")},
        sig.betas,
        sig.positions,
        sig.zscores,
        initial_capital=100_000,
        costs=CostsConfig(fee_bps=0.0, slippage_bps=0.0),
        risk=RiskConfig(max_open_pairs=1, max_pair_weight=0.5),
    )
    assert result.daily["turnover"].iloc[-1] > 0
    assert result.trades.iloc[-1]["position"] == 0


def test_continuous_open_trade_keeps_entry_params_after_window_change() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(prices, first_max_hold=100, second_pairs=True)
    sig = build_continuous_signals(
        prices, plans, RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )
    assert sig.positions.at[prices.index[8], "AAA-BBB"] == 1


def test_continuous_pair_not_reselected_does_not_reenter_after_exit() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(prices, first_max_hold=3, second_pairs=False)
    sig = build_continuous_signals(
        prices, plans, RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )
    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[5], pair] == 1
    assert sig.positions.at[prices.index[6], pair] == 0
    assert sig.positions.loc[prices.index[7] : prices.index[10], pair].eq(0).all()


def test_smoke_cli_writes_core_outputs(tmp_path) -> None:
    prices = _cointegrated_prices(periods=260)
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index_label="date")

    cfg = Config(
        data=DataConfig(prices_path=prices_path),
        pair_selection=PairSelectionConfig(
            min_obs=60,
            min_corr=0.50,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=80.0,
            max_hurst=0.90,
            max_pairs=2,
        ),
        backtest=BacktestConfig(
            initial_capital=100_000,
            walkforward=WalkforwardConfig(
                enabled=True,
                train_mode="rolling",
                train_months=3,
                test_months=1,
                step_months=1,
            ),
        ),
        strategy=StrategyConfig(z_window=10, z_min_periods=3, max_hold_days=20),
        markov=MarkovConfig(enabled=False),
        risk=RiskConfig(max_open_pairs=1, max_pair_weight=0.5, max_drawdown=0.5),
        costs=CostsConfig(fee_bps=0.0, slippage_bps=0.0),
        output=OutputConfig(dir=tmp_path / "out"),
    )
    result = run_config(cfg)
    assert not result.daily.empty
    for name in (
        "summary.json",
        "daily.csv",
        "equity.csv",
        "trades.csv",
        "positions.csv",
        "windows.csv",
        "selected_pairs.csv",
    ):
        assert (tmp_path / "out" / name).exists()
    windows = pd.read_csv(tmp_path / "out" / "windows.csv")
    assert (windows["n_pairs"] > 0).all()
    assert windows["pairs"].str.len().gt(0).all()
    assert not pd.read_csv(tmp_path / "out" / "selected_pairs.csv").empty
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))


def _cointegrated_prices(periods: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    x_log = np.log(50) + np.cumsum(rng.normal(0, 0.01, periods))
    spread = np.zeros(periods)
    for i in range(1, periods):
        spread[i] = 0.75 * spread[i - 1] + rng.normal(0, 0.002)
    y_log = x_log + spread
    z_log = np.log(70) + np.cumsum(rng.normal(0, 0.02, periods))
    return pd.DataFrame(
        {"AAA": np.exp(y_log), "BBB": np.exp(x_log), "CCC": np.exp(z_log)},
        index=idx,
    )


def _continuous_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    x = pd.Series(100 + np.arange(len(idx)), index=idx, dtype=float)
    spread = pd.Series(
        [0, 1, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10], index=idx
    )
    return pd.DataFrame({"AAA": x + spread, "BBB": x}, index=idx)


def _manual_plans(
    prices: pd.DataFrame, *, first_max_hold: int, second_pairs: bool
) -> list[WindowPlan]:
    idx = prices.index
    pair = {"AAA-BBB": ("AAA", "BBB")}
    first = StrategyConfig(
        entry_z=1.5,
        exit_z=0.0,
        stop_z=99.0,
        z_window=3,
        z_min_periods=2,
        max_hold_days=first_max_hold,
        cooldown_days=1,
    )
    second = StrategyConfig(
        entry_z=1.5,
        exit_z=0.0,
        stop_z=99.0,
        z_window=3,
        z_min_periods=2,
        max_hold_days=1,
        cooldown_days=1,
    )
    return [
        WindowPlan(
            Window(0, idx[0], idx[2], idx[3], idx[5]),
            pair,
            first,
            MarkovConfig(enabled=False),
            {"AAA-BBB": 1.0},
        ),
        WindowPlan(
            Window(1, idx[0], idx[5], idx[6], idx[-1]),
            pair if second_pairs else {},
            second,
            MarkovConfig(enabled=False),
            {"AAA-BBB": 1.0} if second_pairs else {},
        ),
    ]
