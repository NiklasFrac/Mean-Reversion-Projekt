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
from backtest.strategy import run_baseline
from backtest.walkforward import Window

BAD_SCORE = -1e6


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
    max_hold_days_by_pair: dict[str, int] | None = None,
) -> tuple[StrategyConfig, MarkovConfig, pd.DataFrame, dict[str, Any]]:
    space = {k: tuple(map(float, v)) for k, v in bo.ranges.items()}
    if not bo.enabled or not space:
        return strategy, markov, pd.DataFrame(), {}

    scoring_window = Window(
        window.i,
        window.train_start,
        window.train_end,
        window.train_start,
        window.train_end,
    )
    trials: list[dict[str, Any]] = []

    def objective(**params: float) -> float:
        strat, mark = _with_params(strategy, markov, params)
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
            mark,
            **baseline_kwargs,
        )
        pos = sig.positions
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
        try:
            score = float(res.summary.get("sharpe", BAD_SCORE))
        except (TypeError, ValueError):
            score = BAD_SCORE
        if not np.isfinite(score):
            score = BAD_SCORE
        trials.append({"score": score, **{k: float(v) for k, v in params.items()}})
        return score

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
