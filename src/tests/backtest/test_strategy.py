from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest.strategy as strategy_mod
from backtest.config import MarkovConfig, RiskConfig, StrategyConfig
from backtest.strategy import (
    WindowPlan,
    build_continuous_signals,
    estimate_beta,
    estimate_betas,
    positions_from_z,
    rolling_zscore,
    run_baseline,
)
from backtest.walkforward import Window


def test_estimate_beta_uses_intercept_and_complete_rows() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    x = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0], index=idx)
    y = pd.Series([12.0, 14.0, np.nan, 18.0, 20.0, 22.0], index=idx)

    assert np.isclose(estimate_beta(y, x), 2.0)


@pytest.mark.parametrize(
    ("x_values", "y_values"),
    [
        ([1.0], [2.0]),
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        ([1.0, 2.0, 3.0], [5.0, 3.0, 1.0]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 1.0]),
    ],
)
def test_estimate_beta_rejects_unusable_or_nonpositive_relationships(
    x_values: list[float], y_values: list[float]
) -> None:
    idx = pd.date_range("2024-01-01", periods=len(x_values), freq="D")
    x = pd.Series(x_values, index=idx)
    y = pd.Series(y_values, index=idx)

    assert estimate_beta(y, x) is None


def test_estimate_betas_filters_invalid_pairs_in_train_window() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    x = pd.Series(np.arange(1.0, 7.0), index=idx)
    prices = pd.DataFrame(
        {
            "AAA": 5.0 + 2.0 * x,
            "BBB": x,
            "FLAT": 10.0,
            "NEG": 20.0 - x,
        },
        index=idx,
    )
    window = Window(0, idx[0], idx[3], idx[4], idx[5])

    betas = estimate_betas(
        prices,
        {
            "AAA-BBB": ("AAA", "BBB"),
            "FLAT-BBB": ("FLAT", "BBB"),
            "NEG-BBB": ("NEG", "BBB"),
        },
        window,
    )

    assert set(betas) == {"AAA-BBB"}
    assert np.isclose(betas["AAA-BBB"], 2.0)


def test_rolling_zscore_is_past_only_and_handles_insufficient_or_flat_history() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    spread = pd.Series([1.0, 2.0, 3.0, 100.0], index=idx)

    z = rolling_zscore(spread, window=2, min_periods=2)

    assert z.name == "z"
    assert z.iloc[:2].isna().all()
    assert np.isclose(z.iloc[2], (3.0 - 1.5) / 0.5)
    assert np.isclose(z.iloc[3], (100.0 - 2.5) / 0.5)

    flat_history = pd.Series([1.0, 1.0, 2.0], index=idx[:3])
    assert pd.isna(rolling_zscore(flat_history, window=2, min_periods=2).iloc[-1])


@pytest.mark.parametrize(
    ("cfg", "z_values", "expected"),
    [
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                max_hold_days=10,
                cooldown_days=1,
            ),
            [-1.2, -0.8, 0.1, -1.5, -0.5, -1.5],
            [1, 1, 0, 0, 0, 1],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                max_hold_days=10,
                cooldown_days=0,
            ),
            [1.2, 1.4, 3.1, 0.0, -1.2, -3.1],
            [-1, -1, 0, 0, 1, 0],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                max_hold_days=10,
                cooldown_days=0,
            ),
            [-1.2, -0.8, 1.2, 0.0, 1.2, 0.0],
            [1, 1, 0, 0, -1, 0],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                max_hold_days=3,
                cooldown_days=0,
            ),
            [-1.2, -0.9, -0.8, -0.7, -0.6, -0.5],
            [1, 1, 1, 0, 0, 0],
        ),
    ],
)
def test_positions_from_z_handles_core_state_transitions(
    cfg: StrategyConfig, z_values: list[float], expected: list[int]
) -> None:
    idx = pd.date_range("2024-01-01", periods=len(z_values), freq="D")

    positions = positions_from_z(pd.Series(z_values, index=idx), cfg)

    assert positions.dtype == np.dtype("int8")
    assert positions.tolist() == expected


def test_positions_from_z_gate_controls_new_entries_only() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    cfg = StrategyConfig(
        entry_z=1.0, exit_z=0.2, stop_z=3.0, max_hold_days=10, cooldown_days=0
    )
    z = pd.Series([-1.2, -0.5, -1.2, -0.8, 0.1, 1.2, 0.0], index=idx)
    gate = pd.Series([False, True, True, False, False, np.nan, True], index=idx)

    assert positions_from_z(z, cfg, gate).tolist() == [0, 0, 1, 1, 0, -1, 0]


def test_run_baseline_filters_pairs_and_applies_markov_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    x = pd.Series(np.arange(100.0, 107.0), index=idx)
    prices = pd.DataFrame(
        {
            "AAA": 5.0 + 2.0 * x + [0.0, 1.0, 0.0, -1.0, -6.0, -6.0, -6.0],
            "BBB": x,
            "FLAT": 42.0,
        },
        index=idx,
    )
    window = Window(0, idx[0], idx[3], idx[4], idx[6])
    strategy = StrategyConfig(
        entry_z=1.0,
        exit_z=0.0,
        stop_z=99.0,
        z_window=2,
        z_min_periods=2,
        max_hold_days=10,
        cooldown_days=0,
    )
    calls = []

    def block_gate(
        z: pd.Series,
        train: pd.DatetimeIndex,
        test: pd.DatetimeIndex,
        cfg: MarkovConfig,
        *,
        entry_z: float,
        exit_z: float,
    ) -> pd.Series:
        calls.append((z, train, test, cfg, entry_z, exit_z))
        return pd.Series(False, index=test)

    monkeypatch.setattr(strategy_mod, "markov_gate", block_gate)

    out = run_baseline(
        prices,
        {
            "AAA-BBB": ("AAA", "BBB"),
            "FLAT-BBB": ("FLAT", "BBB"),
            "MISSING-BBB": ("MISSING", "BBB"),
        },
        window,
        strategy,
        MarkovConfig(enabled=True),
    )

    assert out.positions.index.equals(idx[4:7])
    assert out.positions.columns.tolist() == ["AAA-BBB"]
    assert out.zscores.columns.tolist() == ["AAA-BBB"]
    assert out.positions["AAA-BBB"].dtype == np.dtype("int8")
    assert out.positions["AAA-BBB"].eq(0).all()
    assert set(out.betas) == {"AAA-BBB"}
    assert np.isfinite(out.betas["AAA-BBB"])

    assert len(calls) == 1
    _, train, test, _, entry_z, exit_z = calls[0]
    assert train.equals(pd.DatetimeIndex(idx[:4]))
    assert test.equals(pd.DatetimeIndex(idx[4:7]))
    assert entry_z == strategy.entry_z
    assert exit_z == strategy.exit_z


def test_continuous_signals_carry_unreselected_trade_and_force_final_exit() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(prices, first_max_hold=100, second_pairs=False)

    sig = build_continuous_signals(
        prices, plans, RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )

    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[3], pair] == 1
    assert sig.positions.at[prices.index[7], pair] == 1
    assert sig.positions.at[prices.index[-1], pair] == 0


def test_continuous_signals_keep_entry_strategy_and_beta_after_reselection() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(
        prices, first_max_hold=100, second_pairs=True, second_beta=2.0
    )

    sig = build_continuous_signals(
        prices, plans, RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )

    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[8], pair] == 1
    assert sig.betas.at[prices.index[8], pair] == 1.0
    assert sig.positions.at[prices.index[-1], pair] == 0


def test_continuous_signals_rank_candidates_under_max_open_pairs() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    prices = _prices_from_spreads(
        idx,
        {
            "A-XA": [0.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
            "B-XB": [0.0, 1.0, 0.5, -8.0, -8.0, -8.0, -8.0],
            "C-XC": [0.0, 1.0, 0.5, -4.0, -4.0, -4.0, -4.0],
        },
    )
    pairs = {"A-XA": ("A", "XA"), "B-XB": ("B", "XB"), "C-XC": ("C", "XC")}
    plan = WindowPlan(
        Window(0, idx[0], idx[2], idx[3], idx[-1]),
        pairs,
        StrategyConfig(
            entry_z=1.0,
            exit_z=0.0,
            stop_z=99.0,
            z_window=3,
            z_min_periods=2,
            max_hold_days=100,
            cooldown_days=0,
        ),
        MarkovConfig(enabled=False),
        {pair: 1.0 for pair in pairs},
    )

    sig = build_continuous_signals(
        prices, [plan], RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )

    assert sig.positions.at[idx[3], "B-XB"] == 1
    assert sig.positions.at[idx[3], "A-XA"] == 0
    assert sig.positions.at[idx[3], "C-XC"] == 0
    assert sig.positions.drop(columns="B-XB").eq(0).all().all()


def test_continuous_signals_gate_and_cooldown_block_reentries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=9, freq="D")
    prices = _prices_from_spreads(
        idx,
        {"AAA-BBB": [0.0, 1.0, 0.5, 0.4, 0.43, 0.39, 0.41, 0.38, 0.4]},
    )
    plan = WindowPlan(
        Window(0, idx[0], idx[2], idx[3], idx[-1]),
        {"AAA-BBB": ("AAA", "BBB")},
        StrategyConfig(
            entry_z=1.0,
            exit_z=0.2,
            stop_z=99.0,
            z_window=2,
            z_min_periods=2,
            max_hold_days=1,
            cooldown_days=2,
        ),
        MarkovConfig(enabled=True),
        {"AAA-BBB": 1.0},
    )

    def block_first_entry(
        z: pd.Series,
        train: pd.DatetimeIndex,
        test: pd.DatetimeIndex,
        cfg: MarkovConfig,
        *,
        entry_z: float,
        exit_z: float,
    ) -> pd.Series:
        del z, train, cfg, entry_z, exit_z
        gate = pd.Series(True, index=test)
        gate.iloc[0] = False
        return gate

    monkeypatch.setattr(strategy_mod, "markov_gate", block_first_entry)

    sig = build_continuous_signals(
        prices, [plan], RiskConfig(max_open_pairs=1, max_pair_weight=0.5)
    )

    pair = "AAA-BBB"
    assert sig.positions.at[idx[3], pair] == 0
    assert sig.positions.at[idx[5], pair] == 1
    assert sig.positions.at[idx[6], pair] == 0
    assert sig.positions.at[idx[7], pair] == 0


def _continuous_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    return _prices_from_spreads(
        idx,
        {
            "AAA-BBB": [
                0.0,
                1.0,
                0.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
                -10.0,
            ]
        },
    )


def _manual_plans(
    prices: pd.DataFrame,
    *,
    first_max_hold: int,
    second_pairs: bool,
    second_beta: float = 1.0,
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
            {"AAA-BBB": second_beta} if second_pairs else {},
        ),
    ]


def _prices_from_spreads(
    idx: pd.DatetimeIndex, spreads: dict[str, list[float]]
) -> pd.DataFrame:
    prices = pd.DataFrame(index=idx)
    for pair, spread_values in spreads.items():
        y_name, x_name = pair.split("-")
        x = pd.Series(100.0 + np.arange(len(idx)), index=idx)
        spread = pd.Series(spread_values, index=idx, dtype=float)
        prices[x_name] = x
        prices[y_name] = x + spread
    return prices
