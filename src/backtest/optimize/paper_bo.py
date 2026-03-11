"""
Paper-aligned Bayesian Optimization (BO) for the conservative backtest mode.

Facade module: implementation is split into paper_bo_parts/* while keeping the
original public API and test hooks intact.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from backtest.optimize.paper_bo_parts import core as _core
from backtest.optimize.paper_bo_parts import cv as _cv
from backtest.optimize.paper_bo_parts import pipeline as _pipeline
from backtest.optimize.paper_bo_parts import realistic as _realistic
from backtest.optimize.paper_bo_parts import sim as _sim

logger = logging.getLogger("backtest.paper_bo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)

BayesianOptimization: Any = _core.BayesianOptimization
_BAYES_OK: bool = _core._BAYES_OK
BAD_SCORE: float = _core.BAD_SCORE

_candidate_id = _core._candidate_id
_safe_int = _core._safe_int
_safe_float = _core._safe_float
_append_trial_row = _core._append_trial_row
_log_trial = _core._log_trial
_persist_trials_json = _core._persist_trials_json
_load_trials_json = _core._load_trials_json
_register_previous_trials = _core._register_previous_trials

BOCVConfig = _cv.BOCVConfig
_parse_bo_mode = _cv._parse_bo_mode
_parse_bo_cv = _cv._parse_bo_cv
_fold_score_from_pnl = _cv._fold_score_from_pnl
_fold_score_with_refit = _cv._fold_score_with_refit

RealisticConfig = _realistic.RealisticConfig
_parse_realistic_cfg = _realistic._parse_realistic_cfg
_fold_score_realistic = _realistic._fold_score_realistic
apply_bo_params_to_cfg = _realistic.apply_bo_params_to_cfg

_portfolio_pnl_equal_weight = _sim._portfolio_pnl_equal_weight
_precompute_spreads = _sim._precompute_spreads
_simulate_stage_pnl = _sim._simulate_stage_pnl
_simulate_stage_pnl_refit = _sim._simulate_stage_pnl_refit

run_paper_bo_conservative = _pipeline.run_paper_bo_conservative


def _bayes_optimize(
    *,
    out_dir,
    stage: str,
    pbounds: Mapping[str, tuple[float, float]],
    objective,
    seed: int,
    init_points: int,
    n_iter: int,
    patience: int = 0,
) -> tuple[dict[str, float], float]:
    # Keep monkeypatch behavior from tests by syncing the core module's globals.
    _core.BayesianOptimization = BayesianOptimization
    _core._BAYES_OK = _BAYES_OK
    return _core._bayes_optimize(
        out_dir=out_dir,
        stage=stage,
        pbounds=pbounds,
        objective=objective,
        seed=seed,
        init_points=init_points,
        n_iter=n_iter,
        patience=patience,
    )


__all__ = [
    "BAD_SCORE",
    "BayesianOptimization",
    "_BAYES_OK",
    "_bayes_optimize",
    "_candidate_id",
    "_safe_int",
    "_safe_float",
    "_append_trial_row",
    "_log_trial",
    "_persist_trials_json",
    "_load_trials_json",
    "_register_previous_trials",
    "BOCVConfig",
    "_parse_bo_mode",
    "_parse_bo_cv",
    "_fold_score_from_pnl",
    "_fold_score_with_refit",
    "RealisticConfig",
    "_parse_realistic_cfg",
    "_fold_score_realistic",
    "apply_bo_params_to_cfg",
    "_portfolio_pnl_equal_weight",
    "_precompute_spreads",
    "_simulate_stage_pnl",
    "_simulate_stage_pnl_refit",
    "run_paper_bo_conservative",
]
