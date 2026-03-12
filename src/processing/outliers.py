from __future__ import annotations

from typing import Iterable, cast

import numpy as np
import pandas as pd


def safe_log(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        df = x.apply(pd.to_numeric, errors="coerce").astype("float64")
        mask = (df > 0) & np.isfinite(df.to_numpy())
        if not isinstance(mask, pd.DataFrame):
            mask = pd.DataFrame(mask, index=df.index, columns=df.columns)
        return cast(pd.DataFrame, np.log(df.where(mask)))
    s = pd.to_numeric(x, errors="coerce").astype("float64")
    mask = (s > 0) & np.isfinite(s)
    return cast(pd.Series, np.log(s.where(mask)))


def _causal_returns(
    series: pd.Series,
    *,
    use_log_returns: bool,
    max_gap_bars: int | None = None,
) -> pd.Series:
    """
    Compute returns against the last valid observation (causal), so missing bars do not
    suppress return-based checks.

    If max_gap_bars is set, returns are only computed when the previous valid observation
    is within (max_gap_bars + 1) bars; otherwise the return is NaN.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if use_log_returns:
        log_s = cast(pd.Series, safe_log(s))
        prev = log_s.ffill().shift(1)
        r = log_s - prev
        valid_mask = log_s.notna()
    else:
        prev = s.ffill().shift(1).replace(0.0, np.nan)
        r = (s / prev) - 1.0
        valid_mask = s.notna()

    r = r.replace([np.inf, -np.inf], np.nan)

    if max_gap_bars is not None:
        max_gap_bars = int(max_gap_bars)
        pos = pd.Series(np.arange(len(s), dtype="float64"), index=s.index)
        prev_pos = pos.where(valid_mask).ffill().shift(1)
        dist = pos - prev_pos
        allowed = dist <= float(max_gap_bars + 1)
        r = r.where(allowed)

    return r


def robust_outlier_mask_causal(
    series: pd.Series,
    *,
    zscore: float = 8.0,
    window: int = 21,
    use_log_returns: bool = True,
    max_gap_bars: int | None = None,
) -> pd.Series:
    """
    Causal (backward-looking) Median/MAD heuristic on returns.
    No center=True -> no future information.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    if s.dropna().size < 5:
        return pd.Series(False, index=s.index, dtype=bool)

    r = _causal_returns(
        s, use_log_returns=bool(use_log_returns), max_gap_bars=max_gap_bars
    )

    minp = max(5, window // 2)
    med = r.rolling(window, center=False, min_periods=minp).median()
    mad = (r - med).abs().rolling(
        window, center=False, min_periods=minp
    ).median() * 1.4826

    z = (r - med) / mad.replace(0.0, np.nan)
    return (z.abs() > float(zscore)).fillna(False).astype(bool)


def scrub_outliers_causal(
    series: pd.Series,
    *,
    zscore: float = 8.0,
    window: int = 21,
    use_log_returns: bool = True,
    exclude_dates: Iterable[pd.Timestamp] | None = None,
    max_gap_bars: int | None = None,
) -> tuple[pd.Series, int]:
    """
    Marks the target values P_t of returns detected as outliers as NaN.
    Optional: exclude_dates (e.g. ex-dividend/-split days) -> no marking there.
    """
    mask = robust_outlier_mask_causal(
        series,
        zscore=zscore,
        window=window,
        use_log_returns=use_log_returns,
        max_gap_bars=max_gap_bars,
    )
    if exclude_dates:
        ex = pd.DatetimeIndex(list(exclude_dates))
        idx = cast(pd.DatetimeIndex, series.index)
        mask.loc[idx.normalize().isin(ex.normalize())] = False

    if not bool(mask.any()):
        return series, 0
    s = pd.to_numeric(series, errors="coerce").astype("float64").copy()
    s.loc[mask[mask].index] = np.nan
    s[s <= 0] = np.nan
    return s, int(mask.sum())
