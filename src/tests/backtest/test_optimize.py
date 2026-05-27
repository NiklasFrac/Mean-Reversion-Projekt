from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import backtest.optimize as optimize_mod
from backtest.config import (
    BOConfig,
    CostsConfig,
    MarkovConfig,
    RiskConfig,
    StrategyConfig,
)
from backtest.optimize import BAD_SCORE, _bayes_or_random, _with_params
from backtest.strategy import StrategyOutput
from backtest.walkforward import Window


def test_with_params_maps_known_fields_and_preserves_rest() -> None:
    strategy = StrategyConfig(
        entry_z=1.0,
        exit_z=0.1,
        stop_z=4.0,
        z_window=30,
        z_min_periods=7,
        max_hold_half_life_multiplier=1.7,
        cooldown_days=3,
    )
    markov = MarkovConfig(
        enabled=True,
        horizon_days=12,
        min_revert_prob=0.65,
        min_train_observations=20,
    )

    updated_strategy, updated_markov = _with_params(
        strategy,
        markov,
        {
            "entry_z": 1.4,
            "exit_z": 0.25,
            "stop_z": 2.8,
            "horizon_days": 2.6,
            "min_revert_prob": 0.9,
            "z_window": 99,
            "unknown": 123,
        },
    )

    assert updated_strategy.entry_z == 1.4
    assert updated_strategy.exit_z == 0.25
    assert updated_strategy.stop_z == 2.8
    assert updated_strategy.z_window == strategy.z_window
    assert (
        updated_strategy.max_hold_half_life_multiplier
        == strategy.max_hold_half_life_multiplier
    )
    assert updated_markov.min_revert_prob == 0.9
    assert updated_markov.horizon_days == 3
    assert updated_markov.min_train_observations == markov.min_train_observations


def test_with_params_clamps_horizon_and_keeps_defaults() -> None:
    strategy = StrategyConfig(entry_z=1.2)
    markov = MarkovConfig(horizon_days=20, min_revert_prob=0.7)

    updated_strategy, updated_markov = _with_params(
        strategy, markov, {"horizon_days": -4.2}
    )

    assert updated_strategy == strategy
    assert updated_markov.horizon_days == 1
    assert updated_markov.min_revert_prob == markov.min_revert_prob


def test_bayes_or_random_fallback_is_seeded_and_respects_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_random_fallback(monkeypatch)

    def run_once() -> tuple[dict[str, float], float, list[dict[str, float]]]:
        calls: list[dict[str, float]] = []

        def objective(**params: float) -> float:
            calls.append(dict(params))
            return params["entry_z"] * 10.0 + params["stop_z"]

        best_params, best_score = _bayes_or_random(
            objective,
            {"entry_z": (1.0, 2.0), "stop_z": (2.0, 4.0)},
            seed=123,
            init_points=2,
            n_iter=3,
        )
        return best_params, best_score, calls

    first_params, first_score, first_calls = run_once()
    second_params, second_score, second_calls = run_once()

    assert len(first_calls) == 5
    assert first_calls == second_calls
    assert first_params == second_params
    assert first_score == second_score
    for params in first_calls:
        assert 1.0 <= params["entry_z"] <= 2.0
        assert 2.0 <= params["stop_z"] <= 4.0
    assert first_params == max(
        first_calls, key=lambda params: params["entry_z"] * 10.0 + params["stop_z"]
    )


def test_bayes_or_random_fallback_runs_at_least_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_random_fallback(monkeypatch)
    calls = []

    def objective(**params: float) -> float:
        calls.append(params)
        return 1.0

    best_params, best_score = _bayes_or_random(
        objective, {"entry_z": (1.0, 2.0)}, seed=1, init_points=-5, n_iter=0
    )

    assert len(calls) == 1
    assert set(best_params) == {"entry_z"}
    assert best_score == 1.0


@pytest.mark.parametrize(
    "bo",
    [
        BOConfig(enabled=False, ranges={"entry_z": (1.0, 2.0)}),
        BOConfig(enabled=True, ranges={}),
    ],
)
def test_optimize_params_returns_original_configs_when_disabled_or_no_space(
    monkeypatch: pytest.MonkeyPatch, bo: BOConfig
) -> None:
    strategy = StrategyConfig(entry_z=1.1)
    markov = MarkovConfig(min_revert_prob=0.8)

    monkeypatch.setattr(
        optimize_mod,
        "_bayes_or_random",
        lambda *args, **kwargs: pytest.fail("optimizer should not run"),
    )
    monkeypatch.setattr(
        optimize_mod,
        "run_baseline",
        lambda *args, **kwargs: pytest.fail("baseline should not run"),
    )
    monkeypatch.setattr(
        optimize_mod,
        "run_engine",
        lambda *args, **kwargs: pytest.fail("engine should not run"),
    )

    out_strategy, out_markov, trials, best = optimize_mod.optimize_params(
        _prices(),
        {"AAA-BBB": ("AAA", "BBB")},
        _window(),
        strategy,
        markov,
        RiskConfig(),
        CostsConfig(),
        bo,
        initial_capital=100_000.0,
        seed=7,
    )

    assert out_strategy is strategy
    assert out_markov is markov
    assert trials.empty
    assert best == {}


def test_optimize_params_scores_each_candidate_on_full_train_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    pairs = {"AAA-BBB": ("AAA", "BBB")}
    risk = RiskConfig(max_pair_weight=0.5)
    costs = CostsConfig(fee_bps=1.0, slippage_bps=0.5)
    order: list[str] = []
    baseline_calls: list[tuple[Window, StrategyConfig, MarkovConfig]] = []
    engine_calls: list[dict[str, Any]] = []

    def fake_optimizer(objective, space, seed, init_points, n_iter):
        assert space == {
            "entry_z": (1.0, 2.0),
            "min_revert_prob": (0.6, 0.9),
            "horizon_days": (2.0, 4.0),
        }
        assert seed == 42
        assert init_points == 0
        assert n_iter == 0
        first = {
            "entry_z": 1.1,
            "min_revert_prob": 0.7,
            "horizon_days": 2.2,
        }
        second = {
            "entry_z": 1.8,
            "min_revert_prob": 0.85,
            "horizon_days": 3.6,
        }
        objective(**first)
        best_score = objective(**second)
        return second, best_score

    def fake_baseline(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, tuple[str, str]],
        fold: Window,
        strategy: StrategyConfig,
        markov: MarkovConfig,
    ) -> StrategyOutput:
        order.append("baseline")
        assert prices_arg is prices
        assert pairs_arg is pairs
        assert fold.train_start == _window().train_start
        assert fold.test_start == _window().train_start
        assert fold.train_end == _window().train_end
        assert fold.test_end == _window().train_end
        baseline_calls.append((fold, strategy, markov))
        idx = pd.date_range(fold.test_start, fold.test_end, freq="D")
        return StrategyOutput(
            positions=pd.DataFrame({"AAA-BBB": [1] * len(idx)}, index=idx),
            zscores=pd.DataFrame({"AAA-BBB": [strategy.entry_z] * len(idx)}, index=idx),
            betas={"AAA-BBB": 2.0},
        )

    def fake_engine(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, tuple[str, str]],
        betas: pd.DataFrame,
        positions: pd.DataFrame,
        zscores: pd.DataFrame,
        *,
        initial_capital: float,
        costs: CostsConfig,
        risk: RiskConfig,
    ) -> SimpleNamespace:
        order.append("engine")
        assert prices_arg is prices
        assert pairs_arg is pairs
        assert initial_capital == 250_000.0
        assert costs is costs_obj
        assert risk is risk_obj
        assert betas.index.equals(positions.index)
        assert betas["AAA-BBB"].eq(2.0).all()
        engine_calls.append({"positions": positions, "zscores": zscores})
        return SimpleNamespace(summary={"sharpe": float(zscores["AAA-BBB"].iloc[0])})

    costs_obj = costs
    risk_obj = risk
    monkeypatch.setattr(optimize_mod, "_bayes_or_random", fake_optimizer)
    monkeypatch.setattr(optimize_mod, "run_baseline", fake_baseline)
    monkeypatch.setattr(optimize_mod, "run_engine", fake_engine)

    best_strategy, best_markov, trials, best = optimize_mod.optimize_params(
        prices,
        pairs,
        _window(),
        StrategyConfig(exit_z=0.1, stop_z=3.0),
        MarkovConfig(min_revert_prob=0.65, horizon_days=10),
        risk,
        costs,
        BOConfig(
            enabled=True,
            init_points=0,
            n_iter=0,
            ranges={
                "entry_z": (1, 2),
                "min_revert_prob": (0.6, 0.9),
                "horizon_days": (2, 4),
            },
        ),
        initial_capital=250_000.0,
        seed=42,
    )

    assert len(baseline_calls) == 2
    assert len(engine_calls) == len(baseline_calls)
    assert order == ["baseline", "engine"] * len(baseline_calls)
    assert baseline_calls[0][1].entry_z == 1.1
    assert baseline_calls[0][2].horizon_days == 2
    assert baseline_calls[1][1].entry_z == 1.8
    assert baseline_calls[1][2].horizon_days == 4

    assert trials["score"].tolist() == [1.1, 1.8]
    assert trials["entry_z"].tolist() == [1.1, 1.8]
    assert best_strategy.entry_z == 1.8
    assert best_strategy.exit_z == 0.1
    assert best_strategy.stop_z == 3.0
    assert best_markov.min_revert_prob == 0.85
    assert best_markov.horizon_days == 4
    assert best == {
        "score": 1.8,
        "params": {
            "entry_z": 1.8,
            "min_revert_prob": 0.85,
            "horizon_days": 3.6,
        },
    }


def test_optimize_params_records_bad_score_for_invalid_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_baseline(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, tuple[str, str]],
        fold: Window,
        strategy: StrategyConfig,
        markov: MarkovConfig,
    ) -> StrategyOutput:
        del prices_arg, pairs_arg, fold, strategy, markov
        idx = _prices().index
        return StrategyOutput(
            positions=pd.DataFrame({"AAA-BBB": [1] * len(idx)}, index=idx),
            zscores=pd.DataFrame({"AAA-BBB": [1.0] * len(idx)}, index=idx),
            betas={"AAA-BBB": 2.0},
        )

    monkeypatch.setattr(optimize_mod, "run_baseline", fake_baseline)
    monkeypatch.setattr(
        optimize_mod,
        "run_engine",
        lambda *args, **kwargs: SimpleNamespace(summary={"sharpe": np.nan}),
    )

    def fake_optimizer(objective, space, seed, init_points, n_iter):
        del space, seed, init_points, n_iter
        params = {"entry_z": 1.5}
        return params, objective(**params)

    monkeypatch.setattr(optimize_mod, "_bayes_or_random", fake_optimizer)

    best_strategy, best_markov, trials, best = optimize_mod.optimize_params(
        _prices(),
        {"AAA-BBB": ("AAA", "BBB")},
        _window(),
        StrategyConfig(entry_z=1.0),
        MarkovConfig(),
        RiskConfig(),
        CostsConfig(),
        BOConfig(enabled=True, ranges={"entry_z": (1.0, 2.0)}),
        initial_capital=100_000.0,
        seed=3,
    )

    assert best_strategy.entry_z == 1.5
    assert isinstance(best_markov, MarkovConfig)
    assert trials.to_dict("records") == [{"score": BAD_SCORE, "entry_z": 1.5}]
    assert best == {"score": BAD_SCORE, "params": {"entry_z": 1.5}}


def _force_random_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bayes_opt":
            raise ImportError("force random fallback")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    x = pd.Series(100.0 + np.arange(len(idx)), index=idx)
    return pd.DataFrame({"AAA": x + 2.0, "BBB": x}, index=idx)


def _window() -> Window:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    return Window(0, idx[0], idx[-1], idx[0], idx[-1])
