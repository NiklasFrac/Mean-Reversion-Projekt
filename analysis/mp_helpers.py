"""Multiprocessing helpers kept at module level for pickling."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np

from analysis.bootstrap_fdr import bootstrap_pair_stats
from analysis.threading_control import set_thread_limits


def _worker_init_for_mp(
    blas_threads: Optional[int], numexpr_threads: Optional[int]
) -> None:
    try:
        set_thread_limits(blas_threads=blas_threads, numexpr_threads=numexpr_threads)
    except Exception:
        pass


def _bootstrap_worker(
    idx: int,
    pair_name: str,
    cors: Optional[np.ndarray],
    thr1: float,
    null_mean: float,
    n_resamples: int,
    seed: int,
    block_size: Optional[int],
    ci_level: float,
) -> tuple[int, str, dict[str, Any]]:
    if cors is None or not np.any(~np.isnan(np.asarray(cors))):
        boot: dict[str, Any] = {
            "mean_corr": float("nan"),
            "pct_above_thr1": float("nan"),
            "mean_corr_ci": (np.nan, np.nan),
            "pct1_ci": (np.nan, np.nan),
            "pval": 1.0,
        }
        return idx, pair_name, boot

    boot = bootstrap_pair_stats(
        np.asarray(cors, dtype=float),
        thr1=thr1,
        null_mean=null_mean,
        n_resamples=n_resamples,
        rng_seed=seed,
        block_size=block_size,
        ci_level=float(ci_level),
    )
    return idx, pair_name, boot


def _bootstrap_batch_worker(
    tasks: list[tuple[int, str, Optional[np.ndarray]]],
    thr1: float,
    null_mean: float,
    n_resamples: int,
    base_seed: int,
    block_size: Optional[int],
    ci_level: float,
) -> list[tuple[int, str, dict[str, Any]]]:
    results: list[tuple[int, str, dict[str, Any]]] = []
    for _local_i, (idx, pair_name, cors) in enumerate(tasks):
        h = int(
            hashlib.sha256((pair_name + str(base_seed)).encode()).hexdigest()[:8], 16
        )
        seed = (base_seed ^ h) & 0x7FFFFFFF
        results.append(
            _bootstrap_worker(
                idx,
                pair_name,
                cors,
                thr1,
                null_mean,
                n_resamples,
                seed,
                block_size,
                ci_level,
            )
        )
    return results


__all__ = ["_worker_init_for_mp", "_bootstrap_worker", "_bootstrap_batch_worker"]
