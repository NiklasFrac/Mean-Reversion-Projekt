from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from backtest.utils.tz import align_ts_to_index


def get_tickers_from_meta(data: dict[str, Any]) -> tuple[str, str] | None:
    if not isinstance(data, dict):
        return None
    nested = data.get("meta")
    if isinstance(nested, dict):
        res = get_tickers_from_meta(nested)
        if res:
            return res
    for a_k, b_k in (
        ("t1", "t2"),
        ("y", "x"),
        ("t1_ticker", "t2_ticker"),
        ("asset1", "asset2"),
    ):
        a, b = data.get(a_k), data.get(b_k)
        if isinstance(a, str) and isinstance(b, str) and a and b:
            return a, b
    pair = data.get("pair")
    if isinstance(pair, str) and ("-" in pair or "/" in pair):
        sep = "-" if "-" in pair else "/"
        left, right = [p.strip() for p in pair.split(sep, 1)]
        if left and right:
            return left, right
    return None


def coerce_ts_like_index(ts: Any, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return align_ts_to_index(ts, idx)


def resolve_train_index(
    cfg: dict[str, Any],
    *,
    idx: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> pd.DatetimeIndex:
    raw_dates = cfg.get("_bo_train_dates")
    if isinstance(raw_dates, (list, tuple, pd.Index, pd.DatetimeIndex)) and raw_dates:
        mapped: list[pd.Timestamp] = []
        for raw in raw_dates:
            try:
                ts = coerce_ts_like_index(pd.to_datetime(raw), idx)
            except Exception:
                continue
            mapped.append(pd.Timestamp(ts))
        if mapped:
            out = pd.DatetimeIndex(mapped).drop_duplicates().sort_values()
            out = out.intersection(idx)
            if not out.empty:
                return out
    return idx[(idx >= train_start) & (idx <= train_end)]


def estimate_beta_ols_with_intercept(
    y: pd.Series,
    x: pd.Series,
    *,
    ols_cls: Callable[..., Any] = OLS,
    add_constant_fn: Callable[..., Any] = add_constant,
) -> float:
    beta, _ = estimate_beta_ols_with_intercept_details(
        y,
        x,
        ols_cls=ols_cls,
        add_constant_fn=add_constant_fn,
    )
    return float(beta) if beta is not None else 1.0


def estimate_beta_ols_with_intercept_details(
    y: pd.Series,
    x: pd.Series,
    *,
    ols_cls: Callable[..., Any] = OLS,
    add_constant_fn: Callable[..., Any] = add_constant,
) -> tuple[float | None, str | None]:
    yy = pd.to_numeric(y, errors="coerce")
    xx = pd.to_numeric(x, errors="coerce")
    m = yy.notna() & xx.notna()
    if int(m.sum()) < 2:
        return None, "beta_estimation_failed"
    yv = yy.loc[m].to_numpy(dtype=float, copy=False)
    xv = xx.loc[m].to_numpy(dtype=float, copy=False)
    try:
        X = add_constant_fn(xv)
        res = ols_cls(yv, X).fit()
        beta = float(res.params[-1])
        if not np.isfinite(beta):
            return None, "beta_estimation_failed"
        if beta <= 0.0:
            return None, "beta_non_positive"
        return beta, None
    except Exception:
        return None, "beta_estimation_failed"


def rolling_zscore_stats_past_only(
    spread: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = pd.to_numeric(spread, errors="coerce")
    base = s.shift(1)
    win = int(window)
    minp = int(min_periods)
    m = base.rolling(win, min_periods=minp).mean().rename("spread_mean")
    sd = (
        base.rolling(win, min_periods=minp)
        .std(ddof=0)
        .replace(0.0, np.nan)
        .rename("spread_sigma")
    )
    z = ((s - m) / sd).rename("zscore")
    return z, m, sd


def rolling_zscore_past_only(
    spread: pd.Series, *, window: int, min_periods: int
) -> pd.Series:
    z, _, _ = rolling_zscore_stats_past_only(
        spread, window=window, min_periods=min_periods
    )
    return z


def prior_train_history(
    train_index: pd.DatetimeIndex, *, eval_index: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    train_idx = pd.DatetimeIndex(train_index).drop_duplicates().sort_values()
    eval_idx = pd.DatetimeIndex(eval_index).drop_duplicates().sort_values()
    if train_idx.empty or eval_idx.empty:
        return train_idx[:0]
    return train_idx[train_idx < eval_idx.min()]


def rolling_zscore_on_allowed_dates(
    spread: pd.Series,
    *,
    allowed_index: pd.DatetimeIndex,
    window: int,
    min_periods: int,
    full_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    allowed = pd.DatetimeIndex(allowed_index).drop_duplicates().sort_values()
    if full_index is None:
        base_index = pd.DatetimeIndex(spread.index)
    else:
        base_index = pd.DatetimeIndex(full_index)

    out = pd.Series(np.nan, index=base_index, dtype=float, name="zscore")
    if allowed.empty:
        return out

    s_allowed = pd.to_numeric(spread.reindex(allowed), errors="coerce")
    s_allowed = s_allowed.dropna()
    if s_allowed.empty:
        return out

    z_allowed, _, _ = rolling_zscore_stats_past_only(
        s_allowed, window=window, min_periods=min_periods
    )
    out.loc[z_allowed.index] = pd.to_numeric(z_allowed, errors="coerce").astype(float)
    return out.rename("zscore")


def rolling_zscore_stats_on_allowed_dates(
    spread: pd.Series,
    *,
    allowed_index: pd.DatetimeIndex,
    window: int,
    min_periods: int,
    full_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    allowed = pd.DatetimeIndex(allowed_index).drop_duplicates().sort_values()
    if full_index is None:
        base_index = pd.DatetimeIndex(spread.index)
    else:
        base_index = pd.DatetimeIndex(full_index)

    z_out = pd.Series(np.nan, index=base_index, dtype=float, name="zscore")
    mean_out = pd.Series(np.nan, index=base_index, dtype=float, name="spread_mean")
    sigma_out = pd.Series(np.nan, index=base_index, dtype=float, name="spread_sigma")
    if allowed.empty:
        return z_out, mean_out, sigma_out

    s_allowed = pd.to_numeric(spread.reindex(allowed), errors="coerce")
    s_allowed = s_allowed.dropna()
    if s_allowed.empty:
        return z_out, mean_out, sigma_out

    z_allowed, mean_allowed, sigma_allowed = rolling_zscore_stats_past_only(
        s_allowed,
        window=window,
        min_periods=min_periods,
    )
    z_out.loc[z_allowed.index] = pd.to_numeric(z_allowed, errors="coerce").astype(float)
    mean_out.loc[mean_allowed.index] = pd.to_numeric(
        mean_allowed, errors="coerce"
    ).astype(float)
    sigma_out.loc[sigma_allowed.index] = pd.to_numeric(
        sigma_allowed, errors="coerce"
    ).astype(float)
    return z_out, mean_out, sigma_out


def frozen_zscore(
    spread: pd.Series, *, train_index: pd.DatetimeIndex
) -> tuple[pd.Series, bool]:
    z, _, _, ok = frozen_zscore_stats(spread, train_index=train_index)
    return z, ok


def frozen_zscore_stats(
    spread: pd.Series,
    *,
    train_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    train = pd.to_numeric(spread.reindex(train_index), errors="coerce").dropna()
    if train.empty:
        nan = pd.Series(np.nan, index=spread.index, dtype=float)
        return (
            spread.rename("zscore"),
            nan.rename("spread_mean"),
            nan.rename("spread_sigma"),
            False,
        )
    center = float(train.mean())
    scale = float(train.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        nan = pd.Series(np.nan, index=spread.index, dtype=float)
        return (
            spread.rename("zscore"),
            nan.rename("spread_mean"),
            nan.rename("spread_sigma"),
            False,
        )
    s = pd.to_numeric(spread, errors="coerce")
    mean = pd.Series(center, index=spread.index, dtype=float, name="spread_mean")
    sigma = pd.Series(scale, index=spread.index, dtype=float, name="spread_sigma")
    z = ((s - mean) / sigma).rename("zscore")
    return z, mean, sigma, True
