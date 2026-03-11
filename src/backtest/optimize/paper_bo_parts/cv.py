from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.optimize.cpcv import cpcv_splits
from backtest.optimize import cv_blocks as _cvb
from backtest.simulators.performance import compute_performance

from .core import BAD_SCORE, _log_trial, _safe_float, _safe_int
from .sim import _portfolio_pnl_equal_weight, _simulate_stage_pnl_refit


@dataclass(frozen=True, slots=True)
class BOCVConfig:
    enabled: bool
    scheme: str  # "cpcv" | "blocked" | "none"
    n_blocks: int
    k_test_blocks: int
    purge: int
    embargo: float
    max_folds: int | None
    aggregate: str  # "median" | "mean" | "trimmed_mean"
    trim_pct: float
    shuffle: bool


_VALID_BO_MODES = {"fast", "realistic"}
_REMOVED_BO_MODES = {"lob_rescore", "lob", "expensive"}


def _bo_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    bo = cfg.get("bo", {})
    return bo if isinstance(bo, Mapping) else {}


def _parse_bo_mode(cfg: Mapping[str, Any]) -> str:
    bo = _bo_cfg(cfg)
    mode = str(bo.get("mode", "fast")).strip().lower() or "fast"
    if mode in _REMOVED_BO_MODES:
        raise ValueError(
            f"bo.mode={mode!r} is no longer supported. Use 'realistic' instead."
        )
    if mode not in _VALID_BO_MODES:
        keys = ", ".join(sorted(_VALID_BO_MODES))
        raise ValueError(f"Unsupported bo.mode={mode!r}. Expected one of: {keys}.")
    return mode


def _validate_bo_schema(cfg: Mapping[str, Any]) -> None:
    bo = _bo_cfg(cfg)
    if "cv" in bo:
        raise ValueError(
            "bo.cv is no longer supported. Use bo.fast.cv or bo.realistic.cv."
        )
    if "rescore" in bo:
        raise ValueError(
            "bo.rescore is no longer supported. realistic mode evaluates all BO candidates directly."
        )


def _mode_bo_cfg(cfg: Mapping[str, Any], *, mode: str) -> Mapping[str, Any]:
    _validate_bo_schema(cfg)
    bo = _bo_cfg(cfg)
    raw = bo.get(mode, {})
    return raw if isinstance(raw, Mapping) else {}


def _parse_bo_cv(cfg: Mapping[str, Any], *, mode: str | None = None) -> BOCVConfig:
    mode_eff = mode or _parse_bo_mode(cfg)
    mode_cfg = _mode_bo_cfg(cfg, mode=mode_eff)
    cv = mode_cfg.get("cv", {}) if isinstance(mode_cfg.get("cv"), Mapping) else {}

    scheme = str(cv.get("scheme", cv.get("mode", "none"))).strip().lower()
    enabled = bool(cv.get("enabled", False) and scheme in {"cpcv", "blocked"})

    n_blocks = _safe_int(cv.get("n_blocks", 8), 8)
    k_test = _safe_int(cv.get("k_test_blocks", 2), 2)
    purge = _safe_int(cv.get("purge", 0), 0)
    embargo = _safe_float(cv.get("embargo", 0.0), 0.0)
    mf_raw = cv.get("max_folds", None)
    try:
        max_folds = int(mf_raw) if mf_raw is not None else None
    except Exception:
        max_folds = None

    agg = str(cv.get("aggregate", "median")).strip().lower()
    if agg not in {"median", "mean", "trimmed_mean"}:
        agg = "median"
    trim = float(np.clip(_safe_float(cv.get("trim_pct", 0.1), 0.1), 0.0, 0.4))
    shuffle = bool(cv.get("shuffle", False))

    n_blocks = max(2, int(n_blocks))
    k_test = max(1, min(int(k_test), n_blocks - 1))

    return BOCVConfig(
        enabled=enabled,
        scheme=scheme,
        n_blocks=n_blocks,
        k_test_blocks=k_test,
        purge=max(0, int(purge)),
        embargo=max(0.0, float(embargo)),
        max_folds=max_folds,
        aggregate=agg,
        trim_pct=trim,
        shuffle=shuffle,
    )


def _build_cv_folds(
    calendar: pd.DatetimeIndex,
    *,
    cv: BOCVConfig,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = int(len(calendar))
    if cv.scheme == "cpcv":
        return _validation_indices_from_cpcv(
            n,
            n_blocks=cv.n_blocks,
            k_test_blocks=cv.k_test_blocks,
            purge=cv.purge,
            embargo=cv.embargo,
            max_folds=cv.max_folds,
            shuffle=cv.shuffle,
            seed=seed,
        )
    return _validation_indices_from_blocked(
        n,
        n_blocks=cv.n_blocks,
        k_test_blocks=cv.k_test_blocks,
        purge=cv.purge,
        embargo=cv.embargo,
        max_folds=cv.max_folds,
        shuffle=cv.shuffle,
        seed=seed,
    )


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


# CV block helpers (SSOT in backtest.optimize.cv_blocks)
_compute_block_boundaries = _cvb._compute_block_boundaries
_embargo_len = _cvb._embargo_len
_train_indices_with_purge_embargo = _cvb._train_indices_with_purge_embargo
_trim_block_indices = _cvb._trim_block_indices
_trim_range_indices = _cvb._trim_range_indices


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
    if n_blocks < 2 or k_test_blocks < 1 or k_test_blocks >= n_blocks:
        return []
    if n_samples < n_blocks:
        return []
    boundaries = _compute_block_boundaries(n_samples, n_blocks)
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
        # Infer block ids touched by test_idx
        test_blocks: list[int] = []
        for b in range(int(n_blocks)):
            left = int(boundaries[b])
            right = int(boundaries[b + 1])
            if right <= left:
                continue
            if np.any((test_idx >= left) & (test_idx < right)):
                test_blocks.append(b)
        parts = [
            _trim_block_indices(boundaries, b, purge=purge, embargo=embargo)
            for b in test_blocks
        ]
        parts = [p for p in parts if p.size > 0]
        if not parts:
            continue
        test_idx = np.concatenate(parts)
        train_idx = _train_indices_with_purge_embargo(
            boundaries,
            test_blocks,
            purge=int(max(0, purge)),
            embargo=float(embargo),
        )
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
    if n_blocks < 2 or k_test_blocks < 1 or k_test_blocks >= n_blocks:
        return []
    if n_samples < n_blocks:
        return []
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
    if not folds:
        perf = compute_performance(pnl, float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        _log_trial(out_dir, component=component, params=params_for_log, score=sc)
        return sc if np.isfinite(sc) else BAD_SCORE

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
        )
        pnl = _portfolio_pnl_equal_weight(pnl_by_pair, calendar)
        perf = compute_performance(pnl, float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        _log_trial(out_dir, component=component, params=params_for_log, score=sc)
        return sc if np.isfinite(sc) else BAD_SCORE

    folds = _build_cv_folds(calendar, cv=cv, seed=seed)
    if not folds:
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
        )
        pnl = _portfolio_pnl_equal_weight(pnl_by_pair, calendar)
        perf = compute_performance(pnl, float(initial_capital))
        sc = float(perf.get("sharpe", BAD_SCORE))
        _log_trial(out_dir, component=component, params=params_for_log, score=sc)
        return sc if np.isfinite(sc) else BAD_SCORE

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
