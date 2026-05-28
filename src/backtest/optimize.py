from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import (
    BOConfig,
    CostsConfig,
    GridSearchConfig,
    RiskConfig,
    StrategyConfig,
)
from backtest.engine import run_engine
from backtest.pair_selection import PairMeta
from backtest.strategy import run_baseline
from backtest.walkforward import Window

BAD_SCORE = -1e6


def optimize_params(
    prices: pd.DataFrame,
    pairs: dict[str, PairMeta],
    window: Window,
    strategy: StrategyConfig,
    risk: RiskConfig,
    costs: CostsConfig,
    bo: BOConfig,
    gridsearch: GridSearchConfig,
    *,
    initial_capital: float,
    seed: int,
    max_hold_days_by_pair: dict[str, int] | None = None,
) -> tuple[StrategyConfig, pd.DataFrame, dict[str, Any]]:
    if bo.enabled and gridsearch.enabled:
        raise ValueError("bo.enabled and gridsearch.enabled cannot both be true")
    if bo.enabled:
        _bayesian_optimizer_cls()
    space = {k: tuple(map(float, v)) for k, v in bo.ranges.items()}
    grid = {k: [float(x) for x in v] for k, v in gridsearch.values.items()}
    if gridsearch.enabled:
        if not grid or any(not values for values in grid.values()):
            return strategy, pd.DataFrame(), {}
    elif not bo.enabled or not space:
        return strategy, pd.DataFrame(), {}

    inner_val_start = window.train_start + (
        window.train_end - window.train_start
    ) * (1.0 - float(bo.validation_fraction))
    scoring_window = Window(
        window.i,
        window.train_start,
        inner_val_start,
        inner_val_start,
        window.train_end,
    )
    pair_cols = {pair: (meta.y, meta.x) for pair, meta in pairs.items()}
    trials: list[dict[str, Any]] = []

    def objective(**params: float) -> float:
        strat = _with_params(strategy, params)
        baseline_kwargs = (
            {"max_hold_days_by_pair": max_hold_days_by_pair}
            if max_hold_days_by_pair
            else {}
        )
        sig = run_baseline(
            prices,
            pairs,
            scoring_window,
            strat,
            **baseline_kwargs,
        )
        pos = sig.positions
        betas = pd.DataFrame(sig.betas, index=pos.index)
        res = run_engine(
            prices,
            pair_cols,
            betas,
            pos,
            sig.zscores,
            initial_capital=initial_capital,
            costs=costs,
            risk=risk,
        )
        try:
            score = float(res.summary.get("sharpe", BAD_SCORE))
        except (TypeError, ValueError):
            score = BAD_SCORE
        if not np.isfinite(score):
            score = BAD_SCORE
        trials.append({"score": score, **{k: float(v) for k, v in params.items()}})
        return score

    if gridsearch.enabled:
        best_params, best_score = _grid_search(objective, grid)
    else:
        best_params, best_score = _bayes_optimize(
            objective, space, seed, bo.init_points, bo.n_iter
        )
    best_strategy = _with_params(strategy, best_params)
    return (
        best_strategy,
        pd.DataFrame(trials),
        {
            "score": best_score,
            "params": best_params,
        },
    )


def _with_params(strategy: StrategyConfig, params: dict[str, float]) -> StrategyConfig:
    return replace(
        strategy,
        entry_z=float(params.get("entry_z", strategy.entry_z)),
        exit_z=float(params.get("exit_z", strategy.exit_z)),
        stop_z=float(params.get("stop_z", strategy.stop_z)),
    )


def _bayes_optimize(objective, space, seed, init_points, n_iter):
    BayesianOptimization = _bayesian_optimizer_cls()

    opt = BayesianOptimization(f=objective, pbounds=space, random_state=seed, verbose=0)
    opt.maximize(init_points=max(0, init_points), n_iter=max(0, n_iter))
    best = opt.max or {"params": {}, "target": BAD_SCORE}
    return dict(best["params"]), float(best["target"])


def _bayesian_optimizer_cls():
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as exc:
        raise RuntimeError(
            "bo.enabled is true, but bayesian-optimization is not installed. "
            "Install it with `uv sync --extra backtest`."
        ) from exc
    return BayesianOptimization


def _grid_search(objective, values):
    keys = list(values)
    best_params, best_score = {}, BAD_SCORE
    for candidate in product(*(values[k] for k in keys)):
        params = {k: float(v) for k, v in zip(keys, candidate)}
        score = objective(**params)
        if score > best_score:
            best_params, best_score = params, score
    return best_params, best_score
