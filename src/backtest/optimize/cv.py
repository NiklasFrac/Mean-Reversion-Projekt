from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.config.types import BOCVConfig
from backtest.optimize.cpcv import (
    _compute_block_boundaries,
    _embargo_len,
    _train_indices_with_purge_embargo,
    _validate_params,
    cpcv_splits,
)
from backtest.simulators.performance import compute_performance

from .fast_objective import _portfolio_pnl_equal_weight, _simulate_stage_pnl_refit
from .search import BAD_SCORE, _log_trial


def _build_cv_folds(
    calendar: pd.DatetimeIndex,
    *,
    cv: BOCVConfig,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = int(len(calendar))
    if n <= 0:
        return []
    _validate_cv_config(cv, n_samples=n)
    if cv.scheme == "cpcv":
        folds = _validation_indices_from_cpcv(
            n,
            n_blocks=cv.n_blocks,
            k_test_blocks=cv.k_test_blocks,
            purge=cv.purge,
            embargo=cv.embargo,
            max_folds=cv.max_folds,
            shuffle=cv.shuffle,
            seed=seed,
        )
    else:
        folds = _validation_indices_from_blocked(
            n,
            n_blocks=cv.n_blocks,
            k_test_blocks=cv.k_test_blocks,
            purge=cv.purge,
            embargo=cv.embargo,
            max_folds=cv.max_folds,
            shuffle=cv.shuffle,
            seed=seed,
        )
    if not folds:
        raise ValueError("CV configuration produced no usable folds.")
    return folds


def _aggregate_scores(scores: list[float], *, mode: str, trim_pct: float) -> float:
    vals = np.asarray(
        [s for s in scores if s is not None and np.isfinite(s)], dtype=float
    )
    if vals.size == 0:
        return BAD_SCORE
    if mode == "median":
        return float(np.median(vals))
    if mode == "trimmed_mean":
        k = int(np.floor(trim_pct * vals.size))
        vals.sort()
        core = vals[k : vals.size - k] if (vals.size - 2 * k) > 0 else vals
        return float(np.mean(core))
    return float(np.mean(vals))


def _validate_cv_config(cv: BOCVConfig, *, n_samples: int) -> None:
    _validate_params(
        int(cv.n_blocks),
        int(cv.k_test_blocks),
        int(cv.purge),
        float(cv.embargo),
    )
    if (cv.max_folds is not None) and (int(cv.max_folds) <= 0):
        raise ValueError("cv.max_folds must be > 0 when provided.")
    if not (0.0 <= float(cv.trim_pct) < 0.5):
        raise ValueError("cv.trim_pct must be in [0.0, 0.5).")
    if int(n_samples) < int(cv.n_blocks):
        raise ValueError(
            f"cv.n_blocks={int(cv.n_blocks)} exceeds available samples={int(n_samples)}."
        )


def _trim_range_indices(
    left: int,
    right: int,
    *,
    purge: int,
    embargo: float | int,
) -> np.ndarray:
    blen = right - left
    if blen <= 0:
        return np.empty(0, dtype=np.int64)
    emb = _embargo_len(embargo, blen)
    l2 = min(right, left + int(max(0, purge)))
    r2 = max(l2, right - int(max(0, purge)) - int(max(0, emb)))
    if r2 <= l2:
        return np.empty(0, dtype=np.int64)
    return np.arange(l2, r2, dtype=np.int64)


def _trim_test_segments(
    test_idx: np.ndarray,
    *,
    purge: int,
    embargo: float | int,
) -> list[np.ndarray]:
    parts: list[np.ndarray] = []
    for seg in _split_consecutive_indices(test_idx):
        left = int(seg[0])
        right = int(seg[-1]) + 1
        trimmed = _trim_range_indices(left, right, purge=purge, embargo=embargo)
        if trimmed.size > 0:
            parts.append(trimmed)
    return parts


def _validation_indices_from_cpcv(
    n_samples: int,
    *,
    n_blocks: int,
    k_test_blocks: int,
    purge: int,
    embargo: float,
    max_folds: int | None,
    shuffle: bool,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0:
        return []
    _validate_params(int(n_blocks), int(k_test_blocks), int(purge), float(embargo))
    if int(n_samples) < int(n_blocks):
        raise ValueError(
            f"cv.n_blocks={int(n_blocks)} exceeds available samples={int(n_samples)}."
        )
    out: list[tuple[np.ndarray, np.ndarray]] = []

    for sp in cpcv_splits(
        n_samples=int(n_samples),
        n_blocks=int(n_blocks),
        k_test_blocks=int(k_test_blocks),
        purge=int(max(0, purge)),
        embargo=float(embargo),
        max_splits=max_folds,
        shuffle=bool(shuffle),
        random_state=int(seed),
    ):
        test_idx = np.asarray(sp.test_idx, dtype=np.int64)
        if test_idx.size == 0:
            continue
        parts = _trim_test_segments(test_idx, purge=purge, embargo=embargo)
        if not parts:
            continue
        test_idx = np.concatenate(parts)
        train_idx = np.asarray(sp.train_idx, dtype=np.int64)
        out.append((train_idx, test_idx))
    return out


def _validation_indices_from_blocked(
    n_samples: int,
    *,
    n_blocks: int,
    k_test_blocks: int,
    purge: int,
    embargo: float,
    max_folds: int | None,
    shuffle: bool,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0:
        return []
    _validate_params(int(n_blocks), int(k_test_blocks), int(purge), float(embargo))
    if int(n_samples) < int(n_blocks):
        raise ValueError(
            f"cv.n_blocks={int(n_blocks)} exceeds available samples={int(n_samples)}."
        )
    boundaries = _compute_block_boundaries(n_samples, n_blocks)
    candidates = list(range(0, int(n_blocks) - int(k_test_blocks) + 1))
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(candidates)
    if max_folds is not None and max_folds > 0 and len(candidates) > max_folds:
        candidates = candidates[: int(max_folds)]

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in candidates:
        left = int(boundaries[i])
        right = int(boundaries[i + int(k_test_blocks)])
        test_idx = _trim_range_indices(left, right, purge=purge, embargo=embargo)
        if test_idx.size == 0:
            continue
        test_blocks = list(range(int(i), int(i + int(k_test_blocks))))
        train_idx = _train_indices_with_purge_embargo(
            boundaries,
            test_blocks,
            purge=int(max(0, purge)),
            embargo=float(embargo),
        )
        out.append((train_idx, test_idx))
    return out


def _fold_score_from_pnl(
    pnl: pd.Series,
    *,
    calendar: pd.DatetimeIndex,
    initial_capital: float,
    cv: BOCVConfig,
    seed: int,
    component: str,
    out_dir: Path,
    params_for_log: Mapping[str, Any],
) -> float:
    pnl = (
        pd.to_numeric(pnl, errors="coerce").reindex(calendar).fillna(0.0).astype(float)
    )
    if pnl.empty:
        return BAD_SCORE

    if not cv.enabled:
        perf = compute_performance(pnl, float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        _log_trial(out_dir, component=component, params=params_for_log, score=sc)
        return sc if np.isfinite(sc) else BAD_SCORE

    folds = _build_cv_folds(calendar, cv=cv, seed=seed)
    scores: list[float] = []
    for j, (train_idx, test_idx) in enumerate(folds):
        if test_idx.size == 0:
            continue
        dates = calendar[test_idx]
        perf = compute_performance(pnl.loc[dates], float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        is_score = None
        if train_idx.size > 0:
            train_dates = calendar[train_idx]
            perf_is = compute_performance(pnl.loc[train_dates], float(initial_capital))
            is_score_raw = float(perf_is.get("sharpe", BAD_SCORE))
            if np.isfinite(is_score_raw) and is_score_raw != BAD_SCORE:
                is_score = is_score_raw
        if np.isfinite(sc) and sc != BAD_SCORE:
            scores.append(sc)
            _log_trial(
                out_dir,
                component=component,
                params=params_for_log,
                fold_id=j,
                is_score=is_score,
                oos_score=sc,
            )

    agg = _aggregate_scores(scores, mode=cv.aggregate, trim_pct=cv.trim_pct)
    _log_trial(out_dir, component=component, params=params_for_log, score=agg)
    return agg if np.isfinite(agg) else BAD_SCORE


def _split_consecutive_indices(idx: np.ndarray) -> list[np.ndarray]:
    if idx.size == 0:
        return []
    parts: list[list[int]] = [[int(idx[0])]]
    for v in idx[1:]:
        vi = int(v)
        if vi == parts[-1][-1] + 1:
            parts[-1].append(vi)
        else:
            parts.append([vi])
    return [np.asarray(p, dtype=np.int64) for p in parts if p]


def _fold_score_with_refit(
    *,
    per_pair_prices: Mapping[str, Mapping[str, pd.Series]],
    calendar: pd.DatetimeIndex,
    initial_capital: float,
    cv: BOCVConfig,
    seed: int,
    component: str,
    out_dir: Path,
    params_for_log: Mapping[str, Any],
    z_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_days: int,
    cooldown_days: int,
    cfg: Mapping[str, Any],
    markov_overrides: Mapping[str, Any] | None = None,
) -> float:
    if not cv.enabled:
        pnl_by_pair = _simulate_stage_pnl_refit(
            per_pair_prices=per_pair_prices,
            train_dates=calendar,
            z_window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_hold_days=max_hold_days,
            cooldown_days=cooldown_days,
            cfg=cfg,
            calendar=calendar,
            markov_overrides=markov_overrides,
        )
        pnl = _portfolio_pnl_equal_weight(pnl_by_pair, calendar)
        perf = compute_performance(pnl, float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        _log_trial(out_dir, component=component, params=params_for_log, score=sc)
        return sc if np.isfinite(sc) else BAD_SCORE

    folds = _build_cv_folds(calendar, cv=cv, seed=seed)
    scores: list[float] = []
    for j, (train_idx, test_idx) in enumerate(folds):
        if test_idx.size == 0:
            continue
        train_dates = calendar[train_idx] if train_idx.size > 0 else calendar[:0]
        test_dates = calendar[test_idx]
        fold_pnl_by_pair: dict[str, pd.Series] = {}
        for seg in _split_consecutive_indices(test_idx):
            eval_dates = calendar[seg]
            pnl_seg = _simulate_stage_pnl_refit(
                per_pair_prices=per_pair_prices,
                train_dates=train_dates,
                z_window=z_window,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
                max_hold_days=max_hold_days,
                cooldown_days=cooldown_days,
                cfg=cfg,
                calendar=calendar,
                eval_dates=eval_dates,
                markov_overrides=markov_overrides,
            )
            for k, s in pnl_seg.items():
                if k in fold_pnl_by_pair:
                    fold_pnl_by_pair[k] = fold_pnl_by_pair[k].add(s, fill_value=0.0)
                else:
                    fold_pnl_by_pair[k] = s
        if not fold_pnl_by_pair:
            continue
        pnl = _portfolio_pnl_equal_weight(fold_pnl_by_pair, calendar)
        perf = compute_performance(pnl.loc[test_dates], float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        is_score = None
        if train_idx.size > 0:
            pnl_train_by_pair: dict[str, pd.Series] = {}
            for seg in _split_consecutive_indices(train_idx):
                eval_dates = calendar[seg]
                pnl_seg = _simulate_stage_pnl_refit(
                    per_pair_prices=per_pair_prices,
                    train_dates=train_dates,
                    z_window=z_window,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    stop_z=stop_z,
                    max_hold_days=max_hold_days,
                    cooldown_days=cooldown_days,
                    cfg=cfg,
                    calendar=calendar,
                    eval_dates=eval_dates,
                    markov_overrides=markov_overrides,
                )
                for k, s in pnl_seg.items():
                    if k in pnl_train_by_pair:
                        pnl_train_by_pair[k] = pnl_train_by_pair[k].add(
                            s, fill_value=0.0
                        )
                    else:
                        pnl_train_by_pair[k] = s
            if pnl_train_by_pair:
                pnl_train = _portfolio_pnl_equal_weight(pnl_train_by_pair, calendar)
                perf_is = compute_performance(
                    pnl_train.loc[train_dates], float(initial_capital)
                )
                is_score_raw = float(perf_is.get("sharpe", BAD_SCORE))
                if np.isfinite(is_score_raw) and is_score_raw != BAD_SCORE:
                    is_score = is_score_raw
        if np.isfinite(sc) and sc != BAD_SCORE:
            scores.append(sc)
            _log_trial(
                out_dir,
                component=component,
                params=params_for_log,
                fold_id=j,
                is_score=is_score,
                oos_score=sc,
            )

    agg = _aggregate_scores(scores, mode=cv.aggregate, trim_pct=cv.trim_pct)
    _log_trial(out_dir, component=component, params=params_for_log, score=agg)
    return agg if np.isfinite(agg) else BAD_SCORE
