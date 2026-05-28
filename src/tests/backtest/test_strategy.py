from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.config import StrategyConfig
from backtest.pair_selection import PairMeta
from backtest.strategy import (
    WindowPlan,
    build_continuous_signals,
    log_residual_spread,
    positions_from_z,
    rolling_zscore,
    run_baseline,
)
from backtest.walkforward import Window


def test_log_residual_spread_uses_pair_metadata_alpha_and_beta() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    residual = pd.Series([0.0, 0.2, -0.1, 0.4, -0.3], index=idx)
    alpha, beta = 0.7, 1.4
    prices = _prices_from_log_residuals(idx, {"AAA-BBB": residual}, alpha, beta)
    meta = PairMeta("AAA", "BBB", alpha, beta, 3.0)

    spread = log_residual_spread(prices, idx, meta)

    pd.testing.assert_series_equal(spread, residual.rename("spread"), rtol=1e-12)


def test_rolling_zscore_is_past_only_and_handles_insufficient_or_flat_history() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    spread = pd.Series([1.0, 2.0, 3.0, 100.0], index=idx)

    z = rolling_zscore(spread, window=2)

    assert z.name == "z"
    assert z.iloc[:2].isna().all()
    assert np.isclose(z.iloc[2], (3.0 - 1.5) / 0.5)
    assert np.isclose(z.iloc[3], (100.0 - 2.5) / 0.5)

    flat_history = pd.Series([1.0, 1.0, 2.0], index=idx[:3])
    assert pd.isna(rolling_zscore(flat_history, window=2).iloc[-1])


@pytest.mark.parametrize(
    ("cfg", "z_values", "max_hold_days", "expected"),
    [
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                cooldown_days=1,
            ),
            [-1.2, -0.8, 0.1, -1.5, -0.5, -1.5],
            10,
            [1, 1, 0, 0, 0, 1],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                cooldown_days=0,
            ),
            [1.2, 1.4, 3.1, 0.0, -1.2, -3.1],
            10,
            [-1, -1, 0, 0, 1, 0],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                cooldown_days=0,
            ),
            [-1.2, -0.8, 1.2, 0.0, 1.2, 0.0],
            10,
            [1, 1, 0, 0, -1, 0],
        ),
        (
            StrategyConfig(
                entry_z=1.0,
                exit_z=0.2,
                stop_z=3.0,
                cooldown_days=0,
            ),
            [-1.2, -0.9, -0.8, -0.7, -0.6, -0.5],
            3,
            [1, 1, 1, 0, 0, 0],
        ),
    ],
)
def test_positions_from_z_handles_core_state_transitions(
    cfg: StrategyConfig,
    z_values: list[float],
    max_hold_days: int,
    expected: list[int],
) -> None:
    idx = pd.date_range("2024-01-01", periods=len(z_values), freq="D")

    positions = positions_from_z(
        pd.Series(z_values, index=idx), cfg, max_hold_days=max_hold_days
    )

    assert positions.dtype == np.dtype("int8")
    assert positions.tolist() == expected


def test_positions_from_z_uses_max_hold_override() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    cfg = StrategyConfig(
        entry_z=1.0, exit_z=0.0, stop_z=99.0, cooldown_days=0
    )
    z = pd.Series([-1.2, -0.9, -0.8, -0.7], index=idx)

    assert positions_from_z(z, cfg, max_hold_days=2).tolist() == [1, 1, 0, 0]


def test_run_baseline_filters_pairs_and_generates_zscore_positions() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    residual = pd.Series([0.0, 1.0, 0.0, -1.0, -6.0, -6.0, -6.0], index=idx)
    prices = _prices_from_log_residuals(
        idx,
        {"AAA-BBB": residual},
        alpha=0.9,
        beta=1.7,
    )
    window = Window(0, idx[0], idx[3], idx[4], idx[6])
    strategy = StrategyConfig(
        entry_z=1.0,
        exit_z=0.0,
        stop_z=99.0,
        cooldown_days=0,
    )

    out = run_baseline(
        prices,
        {
            "AAA-BBB": PairMeta("AAA", "BBB", 0.9, 1.7, 2.0),
            "MISSING-BBB": PairMeta("MISSING", "BBB", 0.0, 1.0, 2.0),
        },
        window,
        strategy,
        max_hold_days_by_pair={"AAA-BBB": 2},
    )

    assert out.positions.index.equals(idx[4:7])
    assert out.positions.columns.tolist() == ["AAA-BBB"]
    assert out.zscores.columns.tolist() == ["AAA-BBB"]
    assert out.positions["AAA-BBB"].dtype == np.dtype("int8")
    assert out.positions["AAA-BBB"].tolist() == [1, 1, 0]
    assert set(out.betas) == {"AAA-BBB"}
    assert out.betas["AAA-BBB"] == 1.7
    expected_z = rolling_zscore(residual, 2).reindex(idx[4:7]).rename("AAA-BBB")
    pd.testing.assert_series_equal(out.zscores["AAA-BBB"], expected_z, rtol=1e-12)


def test_run_baseline_derives_z_window_from_half_life_multiplier() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    residual = pd.Series([0.0, 1.0, 0.0, -1.0, -6.0, -6.0, -6.0], index=idx)
    prices = _prices_from_log_residuals(idx, {"AAA-BBB": residual})
    window = Window(0, idx[0], idx[3], idx[4], idx[6])
    strategy = StrategyConfig(
        entry_z=1.0,
        exit_z=0.0,
        stop_z=99.0,
        z_window_multiplier=2.0,
        cooldown_days=0,
    )

    out = run_baseline(
        prices,
        {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 1.0, 1.1)},
        window,
        strategy,
    )

    expected_z = rolling_zscore(residual, 3).reindex(idx[4:7]).rename("AAA-BBB")
    pd.testing.assert_series_equal(out.zscores["AAA-BBB"], expected_z, rtol=1e-12)


@pytest.mark.parametrize(
    ("strategy", "meta", "match"),
    [
        (
            StrategyConfig(z_window_multiplier=0.0),
            PairMeta("AAA", "BBB", 0.0, 1.0, 2.0),
            "z_window_multiplier",
        ),
        (
            StrategyConfig(z_window_multiplier=np.nan),
            PairMeta("AAA", "BBB", 0.0, 1.0, 2.0),
            "z_window_multiplier",
        ),
        (
            StrategyConfig(),
            PairMeta("AAA", "BBB", 0.0, 1.0, 0.0),
            "half_life",
        ),
        (
            StrategyConfig(),
            PairMeta("AAA", "BBB", 0.0, 1.0, np.nan),
            "half_life",
        ),
    ],
)
def test_run_baseline_rejects_invalid_z_window_inputs(
    strategy: StrategyConfig, meta: PairMeta, match: str
) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = _prices_from_spreads(idx, {"AAA-BBB": [0.0, 1.0, 0.0, -1.0, -2.0]})
    window = Window(0, idx[0], idx[2], idx[3], idx[-1])

    with pytest.raises(ValueError, match=match):
        run_baseline(prices, {"AAA-BBB": meta}, window, strategy)


def test_continuous_signals_carry_unreselected_trade_and_force_final_exit() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(prices, first_max_hold=100, second_pairs=False)

    sig = build_continuous_signals(prices, plans)

    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[3], pair] == 1
    assert sig.positions.at[prices.index[7], pair] == 1
    assert sig.positions.at[prices.index[-1], pair] == 0
    assert sig.exit_reasons is not None
    assert sig.exit_reasons.at[prices.index[-1], pair] == "forced_window_end"


def test_continuous_signals_keep_entry_strategy_and_beta_after_reselection() -> None:
    prices = _continuous_prices()
    plans = _manual_plans(
        prices, first_max_hold=100, second_pairs=True, second_beta=2.0
    )

    sig = build_continuous_signals(prices, plans)

    pair = "AAA-BBB"
    assert sig.positions.at[prices.index[8], pair] == 1
    assert sig.betas.at[prices.index[8], pair] == 1.0
    assert sig.positions.at[prices.index[-1], pair] == 0


def test_continuous_signals_opens_all_valid_candidates() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    prices = _prices_from_spreads(
        idx,
        {
            "A-XA": [0.0, 1.0, 0.5, -2.0, -2.0, -2.0, -2.0],
            "B-XB": [0.0, 1.0, 0.5, -8.0, -8.0, -8.0, -8.0],
            "C-XC": [0.0, 1.0, 0.5, -4.0, -4.0, -4.0, -4.0],
        },
    )
    pairs = {
        "A-XA": PairMeta("A", "XA", 0.0, 1.0, 3.0),
        "B-XB": PairMeta("B", "XB", 0.0, 1.0, 3.0),
        "C-XC": PairMeta("C", "XC", 0.0, 1.0, 3.0),
    }
    plan = WindowPlan(
        Window(0, idx[0], idx[2], idx[3], idx[-1]),
        pairs,
        StrategyConfig(
            entry_z=1.0,
            exit_z=0.0,
            stop_z=99.0,
            cooldown_days=0,
        ),
        {pair: 100 for pair in pairs},
    )

    sig = build_continuous_signals(prices, [plan])

    assert sig.positions.at[idx[3], "B-XB"] == 1
    assert sig.positions.at[idx[3], "A-XA"] == 1
    assert sig.positions.at[idx[3], "C-XC"] == 1


def test_continuous_signals_cooldown_blocks_reentries() -> None:
    idx = pd.date_range("2024-01-01", periods=9, freq="D")
    prices = _prices_from_spreads(
        idx,
        {"AAA-BBB": [0.0, 1.0, 0.5, 0.4, 0.43, 0.39, 0.41, 0.38, 0.4]},
    )
    plan = WindowPlan(
        Window(0, idx[0], idx[2], idx[3], idx[-1]),
        {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 1.0, 2.0)},
        StrategyConfig(
            entry_z=1.0,
            exit_z=0.2,
            stop_z=99.0,
            cooldown_days=2,
        ),
        {"AAA-BBB": 1},
    )

    sig = build_continuous_signals(prices, [plan])

    pair = "AAA-BBB"
    assert sig.positions.at[idx[3], pair] == 1
    assert sig.positions.at[idx[4], pair] == 0
    assert sig.positions.at[idx[5], pair] == 0
    assert sig.positions.at[idx[6], pair] == 0
    assert sig.positions.at[idx[7], pair] == 1
    assert sig.exit_reasons is not None
    assert sig.exit_reasons.at[idx[4], pair] == "timeout"


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
    pair = {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 1.0, 3.0)}
    first = StrategyConfig(
        entry_z=1.5,
        exit_z=0.0,
        stop_z=99.0,
        cooldown_days=1,
    )
    second = StrategyConfig(
        entry_z=1.5,
        exit_z=0.0,
        stop_z=99.0,
        cooldown_days=1,
    )
    return [
        WindowPlan(
            Window(0, idx[0], idx[2], idx[3], idx[5]),
            pair,
            first,
            {"AAA-BBB": first_max_hold},
        ),
        WindowPlan(
            Window(1, idx[0], idx[5], idx[6], idx[-1]),
            {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, second_beta, 3.0)}
            if second_pairs
            else {},
            second,
            {"AAA-BBB": 1} if second_pairs else {},
        ),
    ]


def _prices_from_spreads(
    idx: pd.DatetimeIndex, spreads: dict[str, list[float]]
) -> pd.DataFrame:
    return _prices_from_log_residuals(
        idx,
        {
            pair: pd.Series(spread_values, index=idx, dtype=float)
            for pair, spread_values in spreads.items()
        },
    )


def _prices_from_log_residuals(
    idx: pd.DatetimeIndex,
    residuals: dict[str, pd.Series],
    alpha: float = 0.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    prices = pd.DataFrame(index=idx)
    for pair, residual in residuals.items():
        y_name, x_name = pair.split("-")
        x_log = np.log(pd.Series(100.0 + np.arange(len(idx)), index=idx))
        prices[x_name] = np.exp(x_log)
        prices[y_name] = np.exp(alpha + beta * x_log + residual)
    return prices
