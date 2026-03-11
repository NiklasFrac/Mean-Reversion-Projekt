"""Bootstrap statistics and FDR utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np


def _stationary_bootstrap_indices(
    m: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    if m <= 0:
        return np.empty(0, dtype=int)
    if block_size <= 1:
        return np.asarray(rng.integers(0, m, size=m), dtype=int)
    p = 1.0 / float(block_size)
    idx = np.empty(m, dtype=int)
    starts = rng.integers(0, m, size=m)
    pos = int(starts[0])
    for t in range(m):
        if t == 0 or rng.random() < p:
            pos = int(starts[t])
        else:
            pos = (pos + 1) % m
        idx[t] = pos
    return idx


def bootstrap_pair_stats(
    cors: np.ndarray,
    thr1: float,
    null_mean: float,
    n_resamples: int,
    rng_seed: int,
    block_size: Optional[int] = None,
    *,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Stationary block bootstrap over window correlations."""
    cors = np.asarray(cors).ravel()
    mask = ~np.isnan(cors)
    if not np.any(mask):
        return {}
    vals = cors[mask]
    m = len(vals)
    obs_mean = float(np.mean(vals))
    obs_pct1 = float((vals >= thr1).mean() * 100.0)

    if not (0.0 < float(ci_level) < 1.0):
        raise ValueError("ci_level must be in (0, 1)")

    block_size_eff = int(block_size) if block_size is not None else 1
    block_size_eff = max(1, block_size_eff)

    n_resamples_eff = max(1, int(n_resamples))

    rng = np.random.default_rng(rng_seed)
    means = np.empty(n_resamples_eff, dtype=float)
    pct1s = np.empty(n_resamples_eff, dtype=float)

    for b in range(n_resamples_eff):
        if block_size_eff > 1:
            idx = _stationary_bootstrap_indices(m, block_size_eff, rng)
        else:
            idx = rng.integers(0, m, size=m)
        sample = vals[idx]
        means[b] = float(np.mean(sample))
        pct1s[b] = float((sample >= thr1).mean() * 100.0)

    alpha = 1.0 - float(ci_level)
    q_lo = 100.0 * (alpha / 2.0)
    q_hi = 100.0 * (1.0 - alpha / 2.0)

    mean_ci = (float(np.percentile(means, q_lo)), float(np.percentile(means, q_hi)))
    pct1_ci = (float(np.percentile(pct1s, q_lo)), float(np.percentile(pct1s, q_hi)))

    shifted = vals - obs_mean + null_mean
    means_null = np.empty(n_resamples_eff, dtype=float)
    for b in range(n_resamples_eff):
        if block_size_eff > 1:
            idx = _stationary_bootstrap_indices(m, block_size_eff, rng)
        else:
            idx = rng.integers(0, m, size=m)
        sample0 = shifted[idx]
        means_null[b] = float(np.mean(sample0))

    p = (float((means_null >= obs_mean).sum()) + 1.0) / (n_resamples_eff + 1.0)
    p = min(max(p, 0.0), 1.0)

    return {
        "mean_corr": obs_mean,
        "mean_corr_ci": mean_ci,
        "pct_above_thr1": obs_pct1,
        "pct1_ci": pct1_ci,
        "pval": p,
        "k_windows": int(m),
        "n_resamples_eff": int(n_resamples_eff),
        "block_size_eff": int(block_size_eff),
        "ci_level": float(ci_level),
    }


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals, dtype=float)[idx]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = sorted_p <= thresh
    if not np.any(below):
        return [False] * n
    max_i = int(np.where(below)[0].max())
    cutoff = float(sorted_p[max_i])
    return [float(p) <= cutoff for p in pvals]


def benjamini_yekutieli(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    c_n = sum(1.0 / i for i in range(1, n + 1))
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals, dtype=float)[idx]
    thresh = (np.arange(1, n + 1) / n) * (alpha / c_n)
    below = sorted_p <= thresh
    if not np.any(below):
        return [False] * n
    max_i = int(np.where(below)[0].max())
    cutoff = float(sorted_p[max_i])
    return [float(p) <= cutoff for p in pvals]


__all__ = [
    "_stationary_bootstrap_indices",
    "bootstrap_pair_stats",
    "benjamini_hochberg",
    "benjamini_yekutieli",
]
