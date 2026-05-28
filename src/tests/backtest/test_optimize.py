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
    GridSearchConfig,
    RiskConfig,
    StrategyConfig,
    load_config,
)
from backtest.optimize import BAD_SCORE, _grid_search, _with_params
from backtest.pair_selection import PairMeta
from backtest.strategy import StrategyOutput
from backtest.walkforward import Window


def test_with_params_maps_known_fields_and_preserves_rest() -> None:
    strategy = StrategyConfig(
        entry_z=1.0,
        exit_z=0.1,
        stop_z=4.0,
        z_window_multiplier=2.0,
        max_hold_half_life_multiplier=1.7,
        cooldown_days=3,
    )

    updated_strategy = _with_params(
        strategy,
        {
            "entry_z": 1.4,
            "exit_z": 0.25,
            "stop_z": 2.8,
            "unknown": 123,
        },
    )

    assert updated_strategy.entry_z == 1.4
    assert updated_strategy.exit_z == 0.25
    assert updated_strategy.stop_z == 2.8
    assert updated_strategy.z_window_multiplier == strategy.z_window_multiplier
    assert (
        updated_strategy.max_hold_half_life_multiplier
        == strategy.max_hold_half_life_multiplier
    )


def test_with_params_ignores_unknown_fields_and_keeps_defaults(tmp_path) -> None:
    strategy = StrategyConfig(entry_z=1.2)

    updated_strategy = _with_params(strategy, {"unknown": -4.2})

    assert updated_strategy == strategy
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
data:
  prices_path: prices.csv
bo:
  validation_fraction: 0.25
gridsearch:
  enabled: true
  values:
    entry_z: [1.5, 2.0]
    exit_z: [0.0, 0.25]
    stop_z: [3.0, 4.0]
""",
        encoding="utf-8",
    )

    assert BOConfig().validation_fraction == 0.3
    cfg = load_config(cfg_path)
    assert cfg.bo.validation_fraction == 0.25
    assert cfg.gridsearch.enabled
    assert cfg.gridsearch.values == {
        "entry_z": [1.5, 2.0],
        "exit_z": [0.0, 0.25],
        "stop_z": [3.0, 4.0],
    }


def test_load_config_rejects_bo_and_gridsearch_enabled(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
data:
  prices_path: prices.csv
bo:
  enabled: true
gridsearch:
  enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bo.enabled and gridsearch.enabled"):
        load_config(cfg_path)


def test_optimize_params_requires_bayes_dependency_when_bo_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_missing_bayes_opt(monkeypatch)

    with pytest.raises(RuntimeError, match="bo.enabled is true"):
        optimize_mod.optimize_params(
            _prices(),
            {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 2.0, 2.0)},
            _window(),
            StrategyConfig(),
            RiskConfig(),
            CostsConfig(),
            BOConfig(enabled=True, ranges={"entry_z": (1.0, 2.0)}),
            GridSearchConfig(),
            initial_capital=100_000.0,
            seed=1,
        )


def test_grid_search_is_deterministic_and_selects_best() -> None:
    calls: list[dict[str, float]] = []

    def objective(**params: float) -> float:
        calls.append(dict(params))
        return params["entry_z"] * 10.0 + params["stop_z"]

    best_params, best_score = _grid_search(
        objective,
        {
            "entry_z": [1.0, 2.0],
            "stop_z": [3.0, 4.0],
        },
    )

    assert calls == [
        {"entry_z": 1.0, "stop_z": 3.0},
        {"entry_z": 1.0, "stop_z": 4.0},
        {"entry_z": 2.0, "stop_z": 3.0},
        {"entry_z": 2.0, "stop_z": 4.0},
    ]
    assert best_params == {"entry_z": 2.0, "stop_z": 4.0}
    assert best_score == 24.0


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

    monkeypatch.setattr(optimize_mod, "_bayesian_optimizer_cls", lambda: object)
    monkeypatch.setattr(
        optimize_mod,
        "_bayes_optimize",
        lambda *args, **kwargs: pytest.fail("optimizer should not run"),
    )
    monkeypatch.setattr(
        optimize_mod,
        "_grid_search",
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

    out_strategy, trials, best = optimize_mod.optimize_params(
        _prices(),
        {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 2.0, 2.0)},
        _window(),
        strategy,
        RiskConfig(),
        CostsConfig(),
        bo,
        GridSearchConfig(),
        initial_capital=100_000.0,
        seed=7,
    )

    assert out_strategy is strategy
    assert trials.empty
    assert best == {}


def test_optimize_params_scores_each_candidate_on_inner_validation_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    pairs = {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 2.0, 2.0)}
    risk = RiskConfig(max_pair_weight=0.5)
    costs = CostsConfig(cost_bps=1.5)
    order: list[str] = []
    baseline_calls: list[tuple[Window, StrategyConfig]] = []
    engine_calls: list[dict[str, Any]] = []

    def fake_optimizer(objective, space, seed, init_points, n_iter):
        assert space == {"entry_z": (1.0, 2.0)}
        assert seed == 42
        assert init_points == 0
        assert n_iter == 0
        first = {"entry_z": 1.1}
        second = {"entry_z": 1.8}
        objective(**first)
        best_score = objective(**second)
        return second, best_score

    def fake_baseline(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, PairMeta],
        fold: Window,
        strategy: StrategyConfig,
    ) -> StrategyOutput:
        order.append("baseline")
        assert prices_arg is prices
        assert pairs_arg is pairs
        assert fold.train_start == _window().train_start
        expected_val_start = _inner_val_start(_window(), 0.3)
        assert fold.train_end == expected_val_start
        assert fold.test_start == expected_val_start
        assert fold.test_end == _window().train_end
        baseline_calls.append((fold, strategy))
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
        assert pairs_arg == {"AAA-BBB": ("AAA", "BBB")}
        assert initial_capital == 250_000.0
        assert costs is costs_obj
        assert risk is risk_obj
        assert betas.index.equals(positions.index)
        assert betas["AAA-BBB"].eq(2.0).all()
        engine_calls.append({"positions": positions, "zscores": zscores})
        return SimpleNamespace(summary={"sharpe": float(zscores["AAA-BBB"].iloc[0])})

    costs_obj = costs
    risk_obj = risk
    monkeypatch.setattr(optimize_mod, "_bayesian_optimizer_cls", lambda: object)
    monkeypatch.setattr(optimize_mod, "_bayes_optimize", fake_optimizer)
    monkeypatch.setattr(optimize_mod, "run_baseline", fake_baseline)
    monkeypatch.setattr(optimize_mod, "run_engine", fake_engine)

    best_strategy, trials, best = optimize_mod.optimize_params(
        prices,
        pairs,
        _window(),
        StrategyConfig(exit_z=0.1, stop_z=3.0),
        risk,
        costs,
        BOConfig(
            enabled=True,
            init_points=0,
            n_iter=0,
            ranges={
                "entry_z": (1, 2),
            },
        ),
        GridSearchConfig(),
        initial_capital=250_000.0,
        seed=42,
    )

    assert len(baseline_calls) == 2
    assert len(engine_calls) == len(baseline_calls)
    assert order == ["baseline", "engine"] * len(baseline_calls)
    assert baseline_calls[0][1].entry_z == 1.1
    assert baseline_calls[1][1].entry_z == 1.8

    assert trials["score"].tolist() == [1.1, 1.8]
    assert trials["entry_z"].tolist() == [1.1, 1.8]
    assert best_strategy.entry_z == 1.8
    assert best_strategy.exit_z == 0.1
    assert best_strategy.stop_z == 3.0
    assert best == {
        "score": 1.8,
        "params": {"entry_z": 1.8},
    }


def test_optimize_params_scores_gridsearch_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    pairs = {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 2.0, 2.0)}

    def fake_baseline(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, PairMeta],
        fold: Window,
        strategy: StrategyConfig,
    ) -> StrategyOutput:
        del prices_arg, pairs_arg
        idx = pd.date_range(fold.test_start, fold.test_end, freq="D")
        score = strategy.entry_z * 10.0 + strategy.stop_z - strategy.exit_z
        return StrategyOutput(
            positions=pd.DataFrame({"AAA-BBB": [1] * len(idx)}, index=idx),
            zscores=pd.DataFrame({"AAA-BBB": [score] * len(idx)}, index=idx),
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
        del prices_arg, pairs_arg, betas, positions, initial_capital, costs, risk
        return SimpleNamespace(summary={"sharpe": float(zscores["AAA-BBB"].iloc[0])})

    monkeypatch.setattr(
        optimize_mod,
        "_bayes_optimize",
        lambda *args, **kwargs: pytest.fail("bayes optimizer should not run"),
    )
    monkeypatch.setattr(optimize_mod, "run_baseline", fake_baseline)
    monkeypatch.setattr(optimize_mod, "run_engine", fake_engine)

    best_strategy, trials, best = optimize_mod.optimize_params(
        prices,
        pairs,
        _window(),
        StrategyConfig(entry_z=0.5, exit_z=0.4, stop_z=2.0),
        RiskConfig(),
        CostsConfig(),
        BOConfig(enabled=False),
        GridSearchConfig(
            enabled=True,
            values={
                "entry_z": [1.0, 2.0],
                "exit_z": [0.1],
                "stop_z": [3.0, 4.0],
            },
        ),
        initial_capital=100_000.0,
        seed=42,
    )

    assert trials[["entry_z", "exit_z", "stop_z", "score"]].to_dict(
        "records"
    ) == [
        {"entry_z": 1.0, "exit_z": 0.1, "stop_z": 3.0, "score": 12.9},
        {"entry_z": 1.0, "exit_z": 0.1, "stop_z": 4.0, "score": 13.9},
        {"entry_z": 2.0, "exit_z": 0.1, "stop_z": 3.0, "score": 22.9},
        {"entry_z": 2.0, "exit_z": 0.1, "stop_z": 4.0, "score": 23.9},
    ]
    assert best_strategy.entry_z == 2.0
    assert best_strategy.exit_z == 0.1
    assert best_strategy.stop_z == 4.0
    assert best == {
        "score": 23.9,
        "params": {"entry_z": 2.0, "exit_z": 0.1, "stop_z": 4.0},
    }


def test_optimize_params_records_bad_score_for_invalid_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_baseline(
        prices_arg: pd.DataFrame,
        pairs_arg: dict[str, PairMeta],
        fold: Window,
        strategy: StrategyConfig,
    ) -> StrategyOutput:
        del prices_arg, pairs_arg, fold, strategy
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

    monkeypatch.setattr(optimize_mod, "_bayesian_optimizer_cls", lambda: object)
    monkeypatch.setattr(optimize_mod, "_bayes_optimize", fake_optimizer)

    best_strategy, trials, best = optimize_mod.optimize_params(
        _prices(),
        {"AAA-BBB": PairMeta("AAA", "BBB", 0.0, 2.0, 2.0)},
        _window(),
        StrategyConfig(entry_z=1.0),
        RiskConfig(),
        CostsConfig(),
        BOConfig(enabled=True, ranges={"entry_z": (1.0, 2.0)}),
        GridSearchConfig(),
        initial_capital=100_000.0,
        seed=3,
    )

    assert best_strategy.entry_z == 1.5
    assert trials.to_dict("records") == [{"score": BAD_SCORE, "entry_z": 1.5}]
    assert best == {"score": BAD_SCORE, "params": {"entry_z": 1.5}}


def _force_missing_bayes_opt(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bayes_opt":
            raise ImportError("force missing bayes_opt")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    x = pd.Series(100.0 + np.arange(len(idx)), index=idx)
    return pd.DataFrame({"AAA": x + 2.0, "BBB": x}, index=idx)


def _window() -> Window:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    return Window(0, idx[0], idx[-1], idx[0], idx[-1])


def _inner_val_start(window: Window, validation_fraction: float) -> pd.Timestamp:
    return window.train_start + (window.train_end - window.train_start) * (
        1.0 - validation_fraction
    )
