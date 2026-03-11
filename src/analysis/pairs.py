"""Pair extraction and clustering helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from analysis.logging_config import logger
from analysis.utils import _canon_pair


def list_high_pairs_vectorized(
    corr: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    if corr.empty or corr.shape[0] < 2:
        return []
    arr = corr.values
    iu = np.triu_indices_from(arr, k=1)
    vals = arr[iu]
    mask = vals >= threshold
    if not np.any(mask):
        return []
    cols = np.asarray(corr.columns, dtype=object)
    lefts = cols[iu[0][mask]]
    rights = cols[iu[1][mask]]
    corr_vals = vals[mask]
    pairs_raw = list(
        zip(lefts.astype(str).tolist(), rights.astype(str).tolist(), corr_vals.tolist())
    )
    pairs: list[tuple[str, str, float]] = []
    for a, b, v in pairs_raw:
        a_c, b_c = _canon_pair(a, b)
        pairs.append((a_c, b_c, float(v)))
    pairs.sort(key=lambda x: -x[2])
    logger.info("Found %d pairs >= %.3f", len(pairs), threshold)
    return pairs


def pairs_df_from_corr(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["left", "right", "corr"])
    arr = corr.values
    iu = np.triu_indices_from(arr, k=1)
    vals = arr[iu]
    mask = vals >= threshold
    if not np.any(mask):
        return pd.DataFrame(columns=["left", "right", "corr"])
    cols = np.asarray(corr.columns, dtype=object).astype(str)
    lefts = cols[iu[0][mask]]
    rights = cols[iu[1][mask]]
    ab = [_canon_pair(a, b) for a, b in zip(lefts, rights)]
    df = pd.DataFrame(
        {"left": [x[0] for x in ab], "right": [x[1] for x in ab], "corr": vals[mask]}
    )
    return df.sort_values("corr", ascending=False).reset_index(drop=True)


def hierarchical_clustering_from_corr(
    corr: pd.DataFrame, method: str = "average"
) -> np.ndarray:
    if corr.empty:
        raise ValueError("Empty correlation matrix")
    dist = np.sqrt(np.clip(2 * (1 - corr.values), 0, None))
    if not np.isfinite(dist).all():
        raise ValueError("Non-finite entries in correlation distance matrix")
    vec = squareform(dist)
    Z = linkage(vec, method=method)
    return Z


__all__ = [
    "list_high_pairs_vectorized",
    "pairs_df_from_corr",
    "hierarchical_clustering_from_corr",
]
