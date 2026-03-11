"""Rolling correlation metrics over candidate pairs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analysis.logging_config import logger
from analysis.numerics import compute_shrink_corr
from analysis.utils import _canon_pair, _pct_label


def rolling_pair_metrics_fast(
    logrets: pd.DataFrame,
    candidate_pairs: Sequence[tuple[str, str]],
    window: int,
    step: int,
    thr1: float,
    thr2: float,
    *,
    pd_tol: float = 1e-12,
    pd_floor: float = 1e-8,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute per-window correlations on the union of candidate assets."""
    if logrets.empty or not candidate_pairs:
        return pd.DataFrame(), {}

    all_cols = set(map(str, logrets.columns))
    pairs_tuples: list[tuple[str, str]] = []
    names: list[str] = []
    seen: set[str] = set()
    for a, b in candidate_pairs:
        a_s, b_s = _canon_pair(a, b)
        if a_s in all_cols and b_s in all_cols:
            name = f"{a_s}-{b_s}"
            if name in seen:
                continue
            seen.add(name)
            pairs_tuples.append((a_s, b_s))
            names.append(name)
    if not pairs_tuples:
        return pd.DataFrame(), {}

    n = len(logrets)
    windows = [(s, s + window) for s in range(0, n - window + 1, step)]
    k = len(windows)
    if k == 0:
        return pd.DataFrame(), {}

    cors_per_pair = np.full((len(pairs_tuples), k), np.nan, dtype=float)

    assets_union = sorted({x for ab in pairs_tuples for x in ab})
    available_assets = [c for c in assets_union if c in logrets.columns]

    for wi, (s, e) in enumerate(windows):
        seg = logrets.iloc[s:e]
        if seg.shape[0] < max(10, int(window * 0.5)):
            continue
        seg_small = seg.loc[:, available_assets]
        if seg_small.shape[1] < 2:
            continue
        corr_seg = compute_shrink_corr(seg_small, pd_tol=pd_tol, pd_floor=pd_floor)
        cols_set = set(map(str, corr_seg.columns))
        for pi, (a, b) in enumerate(pairs_tuples):
            if a in cols_set and b in cols_set:
                v = corr_seg.at[a, b]
                if isinstance(v, (float, int, np.floating, np.integer)):
                    cors_per_pair[pi, wi] = float(v)

    valid = ~np.isnan(cors_per_pair)
    valid_counts = valid.sum(axis=1)
    means = np.divide(
        np.nansum(cors_per_pair, axis=1),
        valid_counts,
        out=np.full(len(pairs_tuples), np.nan, dtype=float),
        where=valid_counts > 0,
    )

    def pct_ge(thr: float) -> np.ndarray:
        ge = (cors_per_pair >= thr) & valid
        cnt = ge.sum(axis=1)
        return (
            np.divide(
                cnt,
                valid_counts,
                out=np.zeros_like(cnt, dtype=float),
                where=valid_counts > 0,
            )
            * 100.0
        )

    pct1 = pct_ge(thr1)
    pct2 = pct_ge(thr2)

    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)
    df_summary = pd.DataFrame(
        {
            "pair": names,
            "mean_corr_raw": means,
            f"{lbl1}_raw": pct1,
            f"{lbl2}_raw": pct2,
        }
    )
    df_summary["mean_corr"] = np.round(df_summary["mean_corr_raw"], 4)
    df_summary[lbl1] = np.round(df_summary[f"{lbl1}_raw"], 2)
    df_summary[lbl2] = np.round(df_summary[f"{lbl2}_raw"], 2)
    df_summary = df_summary.sort_values(lbl2, ascending=False).reset_index(drop=True)

    zero_valid_mask = valid_counts == 0
    if bool(np.any(zero_valid_mask)):
        bad_pairs = [names[i] for i, m in enumerate(zero_valid_mask) if bool(m)]
        preview = ", ".join(bad_pairs[:10])
        logger.info(
            "Pairs with no valid rolling windows: %d%s",
            len(bad_pairs),
            f" (e.g., {preview} ...)" if len(bad_pairs) > 0 else "",
        )

    window_cors = {names[i]: cors_per_pair[i, :] for i in range(len(names))}
    return df_summary, window_cors


__all__ = ["rolling_pair_metrics_fast"]
