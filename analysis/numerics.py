"""Numerical helpers: shrinkage covariance, correlations, and returns cleaning."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf  # type: ignore[import-untyped]

from analysis.logging_config import logger


def _nearest_positive_definite(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, a_min=eps, a_max=None)
    A_pd = (vecs * vals) @ vecs.T
    out = (A_pd + A_pd.T) / 2.0
    return np.asarray(out, dtype=float)


def compute_shrink_corr(
    returns: pd.DataFrame,
    fix_pd: bool = True,
    pd_fix_fn: Callable[..., np.ndarray] | None = None,
    *,
    pd_tol: float = 1e-12,
    pd_floor: float = 1e-8,
) -> pd.DataFrame:
    if returns.empty or returns.shape[1] == 0:
        return pd.DataFrame()
    if returns.shape[0] < 2:
        return pd.DataFrame()

    cols = pd.Index(map(str, returns.columns))
    X = returns.to_numpy(dtype=float, copy=False)
    _finite_mask = np.isfinite(X)
    if not bool(_finite_mask.all()):
        # The paper-methodology assumes dense returns; keep the function defensive for library usage.
        logger.warning(
            "Non-finite returns encountered; dropping affected rows for shrinkage fit."
        )
        keep_rows = np.isfinite(X).all(axis=1)
        X = X[keep_rows]
        if X.shape[0] < 2:
            return pd.DataFrame()

    # STEP 2 (paper): Ledoit–Wolf shrinkage covariance on R.
    try:
        lw = LedoitWolf().fit(X)
        sigma = np.asarray(lw.covariance_, dtype=float)
    except Exception as e:
        logger.warning(
            "LedoitWolf fit failed (%s); falling back to empirical covariance.", e
        )
        sigma = np.cov(X, rowvar=False, bias=True)
        sigma = np.asarray(sigma, dtype=float)

    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(
            f"Invalid covariance shape from shrinkage estimator: {sigma.shape}"
        )

    sigma = (sigma + sigma.T) / 2.0

    # PD safeguard on Σ (paper): eigenvalue clipping to >= pd_floor when min eig < pd_tol.
    if fix_pd:
        fixer = pd_fix_fn or _nearest_positive_definite
        try:
            min_eig = float(np.linalg.eigvalsh(sigma).min())
        except Exception:
            min_eig = float("nan")
        if not np.isfinite(min_eig) or min_eig < float(pd_tol):
            sigma = fixer(sigma, eps=float(pd_floor))
            sigma = (sigma + sigma.T) / 2.0

    # Convert Σ to correlation C.
    diag = np.diag(sigma)
    d = np.sqrt(np.clip(diag, float(pd_floor), None))
    corr = sigma / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    corr = np.clip(corr, -1.0, 1.0)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)

    return pd.DataFrame(corr, index=cols, columns=cols)


def compute_log_returns(
    df: pd.DataFrame,
    min_positive_frac: float = 0.99,
    max_nan_frac_cols: float = 0.01,
    drop_policy_rows: str = "any",
    min_return_std: float = 0.0,
) -> pd.DataFrame:
    """Compute log-returns with robust cleaning."""
    if df.empty:
        return pd.DataFrame()
    df = df.replace([np.inf, -np.inf], np.nan)

    pos_frac = (df > 0).mean(axis=0).fillna(0.0)
    keep_pos = pos_frac >= min_positive_frac
    if not bool(keep_pos.all()):
        logger.info(
            "Dropping %d cols (positive-price fraction < %.2f%%)",
            int((~keep_pos).sum()),
            min_positive_frac * 100,
        )
    df = df.loc[:, keep_pos]

    arr = df.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_arr = np.log(arr)
    logret = pd.DataFrame(
        np.diff(log_arr, axis=0),
        index=df.index[1:],
        columns=df.columns,
    )
    logret = logret.replace([np.inf, -np.inf], np.nan)

    col_nan_frac = logret.isna().mean(axis=0)
    keep_cols = col_nan_frac <= max_nan_frac_cols
    if not bool(keep_cols.all()):
        logger.info(
            "Dropping %d cols (NaN fraction > %.2f%%)",
            int((~keep_cols).sum()),
            max_nan_frac_cols * 100,
        )
    logret = logret.loc[:, keep_cols]

    before = len(logret)
    dpr = str(drop_policy_rows).lower()
    if dpr == "any":
        logret = logret.dropna(axis=0, how="any")
    elif dpr == "all":
        logret = logret.dropna(axis=0, how="all")
    elif isinstance(dpr, str) and dpr.startswith("pct:"):
        thr = float(dpr.split(":", 1)[1])
        row_nan_frac = logret.isna().mean(axis=1)
        logret = logret.loc[row_nan_frac <= thr]
    else:
        logger.warning(
            "Unknown drop_policy_rows=%r, falling back to 'any'", drop_policy_rows
        )
        logret = logret.dropna(axis=0, how="any")

    dropped = before - len(logret)
    if dropped:
        logger.info("Dropped %d rows due to NaNs (row policy=%s)", dropped, dpr)

    if min_return_std and min_return_std > 0:
        std = logret.std(axis=0, ddof=0).fillna(0.0)
        keep = std > float(min_return_std)
        if not bool(keep.all()):
            logger.info(
                "Dropping %d cols (return std <= %g)",
                int((~keep).sum()),
                float(min_return_std),
            )
        logret = logret.loc[:, keep]

    rows_with_nan = int(logret.isna().any(axis=1).sum()) if not logret.empty else 0
    logger.info(
        "Computed log returns shape %s (rows_with_nan=%d)", logret.shape, rows_with_nan
    )
    return logret


__all__ = ["_nearest_positive_definite", "compute_shrink_corr", "compute_log_returns"]
