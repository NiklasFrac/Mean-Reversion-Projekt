from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# =============================================================================
# Basic helpers (tightened for NaN-safety and typing)
# =============================================================================


def _percentiles(
    arr: NDArray[np.floating[Any]], qs: Iterable[float]
) -> Dict[float, float]:
    """
    Compute percentiles; `qs` may be given as fractions in [0,1] or [0,100].
    NaNs/inf are ignored.

    Returns a mapping {q -> value} where q mirrors the input (fraction or 0-100).
    """
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {float(q): float("nan") for q in qs}

    qs_list = [float(q) for q in qs]
    pcts = [q * 100.0 if q <= 1.0 else q for q in qs_list]
    vals = np.percentile(a, pcts).astype(float).tolist()
    return {q: v for q, v in zip(qs_list, vals, strict=False)}


def _ks_2samp(
    x: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov–Smirnov statistic with a standard asymptotic p-value approximation.
    Pure NumPy, no SciPy dependency. NaNs/Infs are dropped.

    Returns
    -------
    D : float
        KS statistic in [0,1].
    p : float
        Approximate two-sided p-value in [0,1].
    """
    xx = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    yy = np.sort(np.asarray(y, dtype=float)[np.isfinite(y)])
    nx, ny = xx.size, yy.size
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    i = j = 0
    cdf1 = 0.0
    cdf2 = 0.0
    d = 0.0
    while i < nx and j < ny:
        if xx[i] <= yy[j]:
            i += 1
            cdf1 = i / nx
        else:
            j += 1
            cdf2 = j / ny
        d_ij = abs(cdf1 - cdf2)
        if d_ij > d:
            d = d_ij

    en = np.sqrt(nx * ny / (nx + ny))
    # Smirnov approximation, clipped for numerical sanity
    p = float(2.0 * np.exp(-2.0 * (en * d) ** 2))
    p = float(min(1.0, max(0.0, p)))
    return float(d), p


def _roc_pr(
    scores: NDArray[np.floating[Any]], labels01: NDArray[np.integer[Any]]
) -> Dict[str, Any]:
    """
    ROC + PR curves and AUCs for binary labels {0,1}. Scores can be any real number.
    NaNs/inf in either array are dropped pairwise.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels01, dtype=int)
    mask = np.isfinite(s) & np.isfinite(y)
    s = s[mask]
    y = y[mask]
    if s.size == 0:
        return {"roc": [], "pr": [], "auc_roc": 0.0, "auc_pr": 0.0}

    order = np.argsort(-s, kind="mergesort")  # stable for ties
    s = s[order]
    y = y[order]

    P = float(y.sum())
    N = float(y.size - y.sum())

    tp = 0.0
    fp = 0.0
    tpr_list: List[float] = []
    fpr_list: List[float] = []
    prec_list: List[float] = []
    rec_list: List[float] = []

    last_score: float | None = None
    for i in range(y.size):
        if (last_score is None) or (s[i] != last_score):
            if i > 0:
                tpr = tp / max(P, 1.0)
                fpr = fp / max(N, 1.0)
                prec = tp / max(tp + fp, 1.0)
                rec = tpr
                tpr_list.append(float(tpr))
                fpr_list.append(float(fpr))
                prec_list.append(float(prec))
                rec_list.append(float(rec))
            last_score = float(s[i])

        if y[i] == 1:
            tp += 1.0
        else:
            fp += 1.0

    # add last point
    tpr = tp / max(P, 1.0)
    fpr = fp / max(N, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tpr
    tpr_list.append(float(tpr))
    fpr_list.append(float(fpr))
    prec_list.append(float(prec))
    rec_list.append(float(rec))

    auc_roc = float(np.trapezoid(y=np.array(tpr_list), x=np.array(fpr_list)))
    auc_pr = float(np.trapezoid(y=np.array(prec_list), x=np.array(rec_list)))
    return {
        "roc": list(zip(fpr_list, tpr_list)),
        "pr": list(zip(rec_list, prec_list)),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }


# =============================================================================
# Vectorized TCA metrics (Decision/Arrival Shortfall + Markouts)
# =============================================================================


def _sanitize_side(
    side: NDArray[np.floating[Any]] | Sequence[float] | pd.Series | pd.Index,
) -> NDArray[np.float32]:
    """
    Normalize trade side/qty to {-1, +1}. Zeros or NaNs are mapped to 0 (ignored in bps).
    """
    s = np.asarray(side, dtype=float)
    out = np.sign(s)
    # map extremely small values to 0 to avoid noisy signs
    out[np.abs(s) < 1e-12] = 0.0
    # keep only -1,0,1
    out[out > 0] = 1.0
    out[out < 0] = -1.0
    return out.astype(np.float32, copy=False)


def _bps(
    numer: NDArray[np.floating[Any]], denom: NDArray[np.floating[Any]]
) -> NDArray[np.float32]:
    """
    Compute 10k * numer/denom safely, returning float32.
    """
    num = np.asarray(numer, dtype=float)
    den = np.asarray(denom, dtype=float)
    out = np.full(shape=num.shape, fill_value=np.nan, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 0.0)
    out[ok] = 1e4 * (num[ok] / den[ok])
    return out.astype(np.float32, copy=False)


def decision_shortfall_bps(
    side: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    decision_price: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    exec_price: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
) -> NDArray[np.float32]:
    """
    Decision Shortfall in basis points:
      buy  (+1): (exec - decision) / decision * 10_000
      sell (-1): (decision - exec) / decision * 10_000
    Vectorized & NaN-safe. Returns float32 array.

    If side==0 the contribution is 0 (ignored).
    """
    s = _sanitize_side(side)
    dec = np.asarray(decision_price, dtype=float)
    exe = np.asarray(exec_price, dtype=float)
    raw = s * (exe - dec)
    return _bps(raw, dec)


def arrival_shortfall_bps(
    side: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    arrival_price: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    exec_price: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
) -> NDArray[np.float32]:
    """
    Arrival Shortfall in basis points:
      buy  (+1): (exec - arrival) / arrival * 10_000
      sell (-1): (arrival - exec) / arrival * 10_000
    """
    s = _sanitize_side(side)
    arr = np.asarray(arrival_price, dtype=float)
    exe = np.asarray(exec_price, dtype=float)
    raw = s * (exe - arr)
    return _bps(raw, arr)


@dataclass(slots=True)
class MarkoutSpec:
    """
    Configuration for markout computation.

    horizon: pandas Timedelta like "1min", "5min", "30min"
    how: 'ffill' (asof last known price) or 'nearest'
    """

    horizon: pd.Timedelta
    how: str = "ffill"  # 'ffill' (asof) or 'nearest'


def _lookup_prices(
    mid: pd.Series,
    when: pd.DatetimeIndex,
    horizon: pd.Timedelta,
    how: str = "ffill",
) -> Tuple[pd.Series, pd.Series]:
    """
    For each event time `t` in `when`, get mid[t0] and mid[t0+h].
    If event timestamps don't fall exactly on `mid.index`, use:
      - how='ffill'  -> asof (last known price <= t)
      - how='nearest'-> nearest tick

    Returns (p_now, p_fut) as two aligned Series indexed by `when`.
    """
    if not isinstance(mid.index, pd.DatetimeIndex):
        raise TypeError("mid must be indexed by a DatetimeIndex")

    mid = pd.to_numeric(mid, errors="coerce").astype(float).sort_index()
    when = pd.DatetimeIndex(when).sort_values()
    # as-of lookup for now prices
    if how == "nearest":
        p_now = mid.reindex(when, method="nearest")
    else:
        p_now = mid.reindex(when, method="ffill")

    fut_times = when + horizon
    if how == "nearest":
        p_fut = mid.reindex(fut_times, method="nearest")
    else:
        p_fut = mid.reindex(fut_times, method="ffill")

    p_now.index = when
    p_fut.index = when
    return p_now, p_fut


def markouts_bps(
    side: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    event_times: pd.DatetimeIndex,
    mid: pd.Series,
    horizons: Sequence[pd.Timedelta | str] = (
        pd.Timedelta("1min"),
        pd.Timedelta("5min"),
        pd.Timedelta("30min"),
    ),
    *,
    how: str = "ffill",
) -> pd.DataFrame:
    """
    Vectorized markouts (PnL in bps from entry point) for multiple horizons.

    For each event time t and side (+1 buy, -1 sell), computes:
      bps_h = 10_000 * side * (mid[t+h] - mid[t]) / mid[t]

    Parameters
    ----------
    side : array-like
        Trade direction or signed quantity; reduced to {-1, 0, +1}.
    event_times : DatetimeIndex
        Timestamps of fills/signals.
    mid : pd.Series
        Mid-price time series indexed by DatetimeIndex.
    horizons : list of Timedelta or str
        Horizons (e.g., ["1min", "5min", "30min"]).
    how : {'ffill', 'nearest'}
        Matching policy for price lookup.

    Returns
    -------
    DataFrame[float32], index aligned with `event_times`; columns like 't+1min', 't+5min', ...
    NaNs when price not available at t or t+h.
    """
    s = _sanitize_side(side)
    if len(event_times) != s.shape[0]:
        raise ValueError("`side` and `event_times` must have the same length")

    ev = pd.DatetimeIndex(event_times)
    out_cols: Dict[str, NDArray[np.float32]] = {}

    for hz in horizons:
        td = pd.Timedelta(hz)
        col_name = f"t+{td.components.minutes if td.components.days == 0 else str(td)}"
        p0, pH = _lookup_prices(mid, ev, td, how=how)
        raw = s * (
            np.asarray(pH.values, dtype=float) - np.asarray(p0.values, dtype=float)
        )
        out_cols[col_name] = _bps(raw, np.asarray(p0.values, dtype=float))

    df = pd.DataFrame(out_cols, index=ev)
    # make a clean, predictable column naming for typical minute horizons
    rename_map = {
        c: c.replace("t+0:01:00", "t+1m")
        .replace("t+0:05:00", "t+5m")
        .replace("t+0:30:00", "t+30m")
        for c in df.columns
    }
    df = df.rename(columns=rename_map)
    # standardize dtype
    df = df.astype(np.float32)
    return df


# =============================================================================
# Convenience wrappers for your exact t+{1,5,30}m use case
# =============================================================================


def markouts_1_5_30m_bps(
    side: Sequence[float] | NDArray[np.floating[Any]] | pd.Series,
    event_times: pd.DatetimeIndex,
    mid: pd.Series,
    *,
    how: str = "ffill",
) -> pd.DataFrame:
    """
    Convenience call: returns columns ['t+1m','t+5m','t+30m'] as float32.
    """
    return markouts_bps(
        side=side,
        event_times=event_times,
        mid=mid,
        horizons=(pd.Timedelta("1min"), pd.Timedelta("5min"), pd.Timedelta("30min")),
        how=how,
    )
