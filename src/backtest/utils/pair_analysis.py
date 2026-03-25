from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint

from backtest.utils.tz import align_ts_to_index

# ---------- Logging ----------
logger = logging.getLogger("backtest.utils.pair_analysis")
logger.addHandler(logging.NullHandler())

# ---------- Pair Selection Defaults ----------
DEFAULT_COINT_ALPHA: float = 0.05
DEFAULT_PREFILTER_MIN_OBS: int = 30
DEFAULT_HALF_LIFE_MIN_DAYS: float = 5.0
DEFAULT_HALF_LIFE_MAX_DAYS: float = 60.0
DEFAULT_MAX_HOLD_MULTIPLE: float = 2.0
DEFAULT_MIN_DERIVED_DAYS: int = 5


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


def prior_train_history(
    train_index: pd.DatetimeIndex, *, eval_index: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    train_idx = pd.DatetimeIndex(train_index).drop_duplicates().sort_values()
    eval_idx = pd.DatetimeIndex(eval_index).drop_duplicates().sort_values()
    if train_idx.empty or eval_idx.empty:
        return train_idx[:0]
    return train_idx[train_idx < eval_idx.min()]


# ---------- OLS Beta Helpers ----------
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


# ---------- Spread and Z-Score Helpers ----------
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


def compute_spread_zscore(
    y: pd.Series,
    x: pd.Series,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Classic spread/z-score calculation, consistent with the baseline strategy:

      - static OLS hedge ratio beta (y_t = alpha + beta x_t + epsilon_t)
      - spread S_t = y_t - beta x_t
      - rolling mean/std z-score with window z_window
    """
    cfg = dict(cfg or {})
    z_window = int(cfg.get("z_window", 30))
    z_min_periods = int(cfg.get("z_min_periods", max(z_window // 2, 1)))

    yy = pd.Series(pd.to_numeric(y, errors="coerce"), index=y.index)
    xx = pd.Series(pd.to_numeric(x, errors="coerce"), index=x.index)
    idx = yy.index.intersection(xx.index)
    yy = yy.reindex(idx).ffill().bfill()
    xx = xx.reindex(idx).ffill().bfill()

    mask = yy.notna() & xx.notna()
    y_reg = yy[mask]
    x_reg = xx[mask]

    if len(y_reg) < 2:
        spread = (yy - xx).rename("spread")
        m = spread.rolling(z_window, min_periods=z_min_periods).mean()
        s = (
            spread.rolling(z_window, min_periods=z_min_periods)
            .std(ddof=0)
            .replace(0.0, np.nan)
        )
        z = ((spread - m) / s).rename("zscore")
        beta_series = pd.Series(1.0, index=spread.index, name="beta")
        return spread, z, beta_series

    beta_hat, beta_reason = estimate_beta_ols_with_intercept_details(
        y_reg,
        x_reg,
        ols_cls=OLS,
        add_constant_fn=add_constant,
    )
    if beta_hat is None:
        if beta_reason == "beta_non_positive":
            logger.debug(
                "OLS hedge ratio was non-positive in compute_spread_zscore; using beta=1.0 fallback"
            )
        else:
            logger.debug("OLS hedge ratio failed in compute_spread_zscore")
        beta_hat = 1.0

    spread = (yy - beta_hat * xx).rename("spread")
    m = spread.rolling(z_window, min_periods=z_min_periods).mean()
    s = (
        spread.rolling(z_window, min_periods=z_min_periods)
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    z = ((spread - m) / s).rename("zscore")

    beta_series = pd.Series(beta_hat, index=spread.index, name="beta")
    return spread, z, beta_series


# ---------- Cointegration and Half-Life Helpers ----------
def safe_coint(x: pd.Series, y: pd.Series, alpha: float = DEFAULT_COINT_ALPHA) -> bool:
    """Engle-Granger cointegration; False on errors/too little data."""
    xx = pd.Series(pd.to_numeric(x, errors="coerce"), index=x.index).dropna()
    yy = pd.Series(pd.to_numeric(y, errors="coerce"), index=y.index).dropna()
    if xx.empty or yy.empty:
        return False
    try:
        return bool(coint(xx, yy)[1] < alpha)
    except Exception as exc:
        logger.debug("safe_coint failed: %s", exc)
        return False


def resolve_half_life_cfg(
    half_life_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(half_life_cfg or {})
    min_days = float(cfg.get("min_days", DEFAULT_HALF_LIFE_MIN_DAYS))
    max_days = float(cfg.get("max_days", DEFAULT_HALF_LIFE_MAX_DAYS))
    max_hold_multiple = float(cfg.get("max_hold_multiple", DEFAULT_MAX_HOLD_MULTIPLE))
    min_derived_days = int(cfg.get("min_derived_days", DEFAULT_MIN_DERIVED_DAYS))

    if not np.isfinite(min_days) or min_days <= 0.0:
        raise ValueError("pair_prefilter.half_life.min_days must be > 0")
    if not np.isfinite(max_days) or max_days < min_days:
        raise ValueError("pair_prefilter.half_life.max_days must be >= min_days")
    if not np.isfinite(max_hold_multiple) or max_hold_multiple <= 0.0:
        raise ValueError("pair_prefilter.half_life.max_hold_multiple must be > 0")
    if int(min_derived_days) < 1:
        raise ValueError("pair_prefilter.half_life.min_derived_days must be >= 1")

    return {
        "min_days": float(min_days),
        "max_days": float(max_days),
        "max_hold_multiple": float(max_hold_multiple),
        "min_derived_days": int(min_derived_days),
    }


def _diag_template(
    *, passed: bool, reject_reason: str | None, n_obs: int
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reject_reason": reject_reason,
        "n_obs": int(max(0, n_obs)),
        "eg_pvalue": np.nan,
        "beta": np.nan,
        "lambda": np.nan,
        "half_life": np.nan,
        "z_window": None,
        "max_hold_days": None,
    }


def evaluate_pair_cointegration(
    prices: pd.DataFrame,
    *,
    coint_alpha: float = DEFAULT_COINT_ALPHA,
    min_obs: int = DEFAULT_PREFILTER_MIN_OBS,
    half_life_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a pair with Engle-Granger first and optional half-life diagnostics.

    If `half_life_cfg` is omitted, the function behaves like the legacy EG-only
    prefilter and does not require AR(1)/half-life estimation to pass.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] < 2:
        return _diag_template(passed=False, reject_reason="invalid_input", n_obs=0)

    df = prices.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.shape[1] < 2:
        return _diag_template(passed=False, reject_reason="invalid_input", n_obs=0)

    y_raw = df.iloc[:, 0]
    x_raw = df.iloc[:, 1]
    y, x = y_raw.align(x_raw, join="inner")
    mask = y.notna() & x.notna()
    y = y.loc[mask].astype(float)
    x = x.loc[mask].astype(float)
    n_obs = int(len(y))

    if y.empty or x.empty:
        return _diag_template(passed=False, reject_reason="invalid_input", n_obs=n_obs)

    n_min = max(2, int(min_obs))
    if n_obs < n_min:
        return _diag_template(
            passed=False, reject_reason="min_obs_not_met", n_obs=n_obs
        )

    out = _diag_template(passed=False, reject_reason="eg_failed", n_obs=n_obs)
    try:
        _stat, eg_pvalue, _crit = coint(y, x)
        out["eg_pvalue"] = float(eg_pvalue)
    except Exception as exc:
        logger.debug("evaluate_pair_cointegration: EG test failed: %s", exc)
        out["reject_reason"] = "eg_error"
        return out

    if not np.isfinite(float(out["eg_pvalue"])) or float(out["eg_pvalue"]) >= float(
        coint_alpha
    ):
        return out

    beta_hat, beta_reason = estimate_beta_ols_with_intercept_details(
        y,
        x,
        ols_cls=OLS,
        add_constant_fn=add_constant,
    )
    if beta_hat is None:
        if beta_reason == "beta_estimation_failed":
            logger.debug("evaluate_pair_cointegration: beta estimation failed")
        out["reject_reason"] = str(beta_reason or "beta_estimation_failed")
        return out
    out["beta"] = float(beta_hat)

    if half_life_cfg is None:
        out["passed"] = True
        out["reject_reason"] = None
        return out

    hl_cfg = resolve_half_life_cfg(half_life_cfg)

    resid = (y - beta_hat * x).astype(float)
    lag = resid.shift(1)
    delta = resid.diff()
    ar_df = pd.DataFrame({"lag": lag, "delta": delta}).dropna()
    if ar_df.shape[0] < 2:
        out["reject_reason"] = "half_life_non_finite"
        return out

    lag_vals = ar_df["lag"].to_numpy(dtype=float, copy=False)
    delta_vals = ar_df["delta"].to_numpy(dtype=float, copy=False)
    denom = float(np.dot(lag_vals, lag_vals))
    if not np.isfinite(denom) or denom <= 0.0:
        out["reject_reason"] = "half_life_non_finite"
        return out

    lambda_hat = float(np.dot(lag_vals, delta_vals) / denom)
    out["lambda"] = float(lambda_hat)

    if lambda_hat >= 0.0:
        out["reject_reason"] = "lambda_non_negative"
        return out

    base = 1.0 + float(lambda_hat)
    if not np.isfinite(base) or base <= 0.0:
        out["reject_reason"] = "lambda_invalid_domain"
        return out

    try:
        half_life = float(-np.log(2.0) / np.log(base))
    except Exception:
        half_life = float("nan")
    out["half_life"] = float(half_life)

    if not np.isfinite(half_life) or half_life <= 0.0:
        out["reject_reason"] = "half_life_non_finite"
        return out
    if half_life < float(hl_cfg["min_days"]):
        out["reject_reason"] = "half_life_too_fast"
        return out
    if half_life > float(hl_cfg["max_days"]):
        out["reject_reason"] = "half_life_too_slow"
        return out

    min_derived = int(hl_cfg["min_derived_days"])
    z_window_raw = int(round(float(half_life)))
    max_hold_raw = int(round(float(hl_cfg["max_hold_multiple"]) * float(half_life)))
    out["z_window"] = int(max(min_derived, z_window_raw))
    out["max_hold_days"] = int(max(min_derived, max_hold_raw))
    out["passed"] = True
    out["reject_reason"] = None
    return out


def pair_prefilter(
    prices: pd.DataFrame,
    *,
    coint_alpha: float = DEFAULT_COINT_ALPHA,
    min_obs: int = DEFAULT_PREFILTER_MIN_OBS,
    half_life_cfg: Mapping[str, Any] | None = None,
) -> bool:
    """
    Coarse filter for pairs: minimal QC + Engle-Granger cointegration test.

    - expects a DataFrame with at least two columns (first = y, second = x)
    - numeric cleanup + alignment
    - minimum length
    - Engle-Granger via safe_coint
    """
    result = evaluate_pair_cointegration(
        prices,
        coint_alpha=float(coint_alpha),
        min_obs=int(min_obs),
        half_life_cfg=half_life_cfg,
    )
    return bool(result.get("passed", False))


__all__ = [
    "DEFAULT_COINT_ALPHA",
    "DEFAULT_PREFILTER_MIN_OBS",
    "DEFAULT_HALF_LIFE_MIN_DAYS",
    "DEFAULT_HALF_LIFE_MAX_DAYS",
    "DEFAULT_MAX_HOLD_MULTIPLE",
    "DEFAULT_MIN_DERIVED_DAYS",
    "coerce_ts_like_index",
    "resolve_train_index",
    "prior_train_history",
    "estimate_beta_ols_with_intercept",
    "estimate_beta_ols_with_intercept_details",
    "rolling_zscore_stats_past_only",
    "rolling_zscore_past_only",
    "rolling_zscore_on_allowed_dates",
    "rolling_zscore_stats_on_allowed_dates",
    "frozen_zscore",
    "frozen_zscore_stats",
    "compute_spread_zscore",
    "safe_coint",
    "resolve_half_life_cfg",
    "evaluate_pair_cointegration",
    "pair_prefilter",
]
