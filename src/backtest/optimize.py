from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import (
    BOConfig,
    CostsConfig,
    MarkovConfig,
    RiskConfig,
    StrategyConfig,
)
from backtest.engine import run_engine
from backtest.risk import cap_positions
from backtest.strategy import run_baseline
from backtest.walkforward import Window

BAD_SCORE = -1e6


def blocked_folds(
    index: pd.DatetimeIndex, bo: BOConfig
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    idx = index.drop_duplicates().sort_values()
    bounds = np.linspace(0, len(idx), bo.cv.n_blocks + 1, dtype=int)
    folds = []
    for i in range(0, bo.cv.n_blocks - bo.cv.k_test_blocks + 1):
        left, right = bounds[i], bounds[i + bo.cv.k_test_blocks]
        test = idx[left:right]
        if bo.cv.purge:
            test = test[bo.cv.purge : max(bo.cv.purge, len(test) - bo.cv.purge)]
        train_mask = np.ones(len(idx), dtype=bool)
        train_mask[
            max(0, left - bo.cv.purge) : min(
                len(idx), right + int(bo.cv.embargo * len(test))
            )
        ] = False
        train = idx[train_mask]
        if len(train) and len(test):
            folds.append((train, test))
    return folds


def optimize_params(
    prices: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    window: Window,
    strategy: StrategyConfig,
    markov: MarkovConfig,
    risk: RiskConfig,
    costs: CostsConfig,
    bo: BOConfig,
    *,
    initial_capital: float,
    seed: int,
) -> tuple[StrategyConfig, MarkovConfig, pd.DataFrame, dict[str, Any]]:
    space = {k: tuple(map(float, v)) for k, v in bo.ranges.items()}
    if not bo.enabled or not space:
        return strategy, markov, pd.DataFrame(), {}

    folds = blocked_folds(
        pd.DatetimeIndex(prices.loc[window.train_start : window.train_end].index), bo
    )
    trials: list[dict[str, Any]] = []

    def objective(**params: float) -> float:
        strat, mark = _with_params(strategy, markov, params)
        scores = []
        for train, test in folds:
            fold = Window(0, train[0], train[-1], test[0], test[-1])
            sig = run_baseline(prices, pairs, fold, strat, mark)
            pos = cap_positions(sig.positions, sig.zscores, risk)
            betas = pd.DataFrame(sig.betas, index=pos.index)
            res = run_engine(
                prices,
                pairs,
                betas,
                pos,
                sig.zscores,
                initial_capital=initial_capital,
                costs=costs,
                risk=risk,
            )
            scores.append(float(res.summary.get("sharpe", BAD_SCORE)))
        score = float(np.median(scores)) if scores else BAD_SCORE
        trials.append({"score": score, **{k: float(v) for k, v in params.items()}})
        return score if np.isfinite(score) else BAD_SCORE

    best_params, best_score = _bayes_or_random(
        objective, space, seed, bo.init_points, bo.n_iter
    )
    best_strategy, best_markov = _with_params(strategy, markov, best_params)
    return (
        best_strategy,
        best_markov,
        pd.DataFrame(trials),
        {
            "score": best_score,
            "params": best_params,
        },
    )


def _with_params(
    strategy: StrategyConfig, markov: MarkovConfig, params: dict[str, float]
) -> tuple[StrategyConfig, MarkovConfig]:
    return (
        replace(
            strategy,
            entry_z=float(params.get("entry_z", strategy.entry_z)),
            exit_z=float(params.get("exit_z", strategy.exit_z)),
            stop_z=float(params.get("stop_z", strategy.stop_z)),
        ),
        replace(
            markov,
            min_revert_prob=float(
                params.get("min_revert_prob", markov.min_revert_prob)
            ),
            horizon_days=max(
                1, int(round(params.get("horizon_days", markov.horizon_days)))
            ),
        ),
    )


def _bayes_or_random(objective, space, seed, init_points, n_iter):
    try:
        from bayes_opt import BayesianOptimization

        opt = BayesianOptimization(
            f=objective, pbounds=space, random_state=seed, verbose=0
        )
        opt.maximize(init_points=max(0, init_points), n_iter=max(0, n_iter))
        best = opt.max or {"params": {}, "target": BAD_SCORE}
        return dict(best["params"]), float(best["target"])
    except Exception:
        rng = np.random.default_rng(seed)
        best_params, best_score = {}, BAD_SCORE
        for _ in range(max(1, init_points + n_iter)):
            params = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}
            score = objective(**params)
            if score > best_score:
                best_params, best_score = params, score
        return best_params, best_score
