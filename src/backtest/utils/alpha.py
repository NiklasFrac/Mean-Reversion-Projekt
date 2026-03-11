# helpers (Tier-1: fit→apply, robust, reproducible)

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.stattools import coint

# ---------- Logging ----------
logger = logging.getLogger("Tier1StrategyOptimized")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel("INFO")

# ---------- Defaults ----------
DEFAULT_COINT_ALPHA: float = 0.05
DEFAULT_PREFILTER_MIN_OBS: int = 30
DEFAULT_HALF_LIFE_MIN_DAYS: float = 5.0
DEFAULT_HALF_LIFE_MAX_DAYS: float = 60.0
DEFAULT_MAX_HOLD_MULTIPLE: float = 2.0
DEFAULT_MIN_DERIVED_DAYS: int = 5


# ---------- Stats helpers ----------
def safe_coint(x: pd.Series, y: pd.Series, alpha: float = DEFAULT_COINT_ALPHA) -> bool:
    """Engle-Granger Cointegration; False bei Fehlern/zu wenig Daten."""
    xx = pd.Series(pd.to_numeric(x, errors="coerce"), index=x.index).dropna()
    yy = pd.Series(pd.to_numeric(y, errors="coerce"), index=y.index).dropna()
    if xx.empty or yy.empty:
        return False
    try:
        return bool(coint(xx, yy)[1] < alpha)
    except Exception as e:
        logger.debug("safe_coint failed: %s", e)
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
    except Exception as e:
        logger.debug("evaluate_pair_cointegration: EG test failed: %s", e)
        out["reject_reason"] = "eg_error"
        return out

    if not np.isfinite(float(out["eg_pvalue"])) or float(out["eg_pvalue"]) >= float(
        coint_alpha
    ):
        return out

    if half_life_cfg is None:
        out["passed"] = True
        out["reject_reason"] = None
        return out

    hl_cfg = resolve_half_life_cfg(half_life_cfg)

    try:
        X = add_constant(x.to_numpy(dtype=float, copy=False))
        res = OLS(y.to_numpy(dtype=float, copy=False), X).fit()
        beta_hat = float(res.params[-1])
    except Exception as e:
        logger.debug("evaluate_pair_cointegration: beta estimation failed: %s", e)
        out["reject_reason"] = "beta_estimation_failed"
        return out

    if not np.isfinite(beta_hat):
        out["reject_reason"] = "beta_estimation_failed"
        return out
    out["beta"] = float(beta_hat)

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


def compute_spread_zscore(
    y: pd.Series,
    x: pd.Series,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Klassische Spread-/Z-Score-Berechnung, konsistent mit der Baseline-Strategie:

      - statisches OLS-Hedge-Ratio β (y_t = α + β x_t + ε_t)
      - Spread S_t = y_t − β x_t
      - Rolling-Mean/Std-Z-Score mit Fenster z_window

    """
    cfg = dict(cfg or {})
    z_window = int(cfg.get("z_window", 30))
    z_min_periods = int(cfg.get("z_min_periods", max(z_window // 2, 1)))

    # 1) Alignment und Numerik-Säuberung
    yy = pd.Series(pd.to_numeric(y, errors="coerce"), index=y.index)
    xx = pd.Series(pd.to_numeric(x, errors="coerce"), index=x.index)
    idx = yy.index.intersection(xx.index)
    yy = yy.reindex(idx).ffill().bfill()
    xx = xx.reindex(idx).ffill().bfill()

    # 2) OLS β-Schätzung (y ~ α + β x) auf gemeinsamen finite Daten
    mask = yy.notna() & xx.notna()
    y_reg = yy[mask]
    x_reg = xx[mask]

    if len(y_reg) < 2:
        # Fallback: β = 1.0, Spread = y − x
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

    try:
        X = add_constant(x_reg.values)
        res = OLS(y_reg.values, X).fit()
        beta_hat = float(res.params[-1])
    except Exception as e:
        logger.debug("OLS hedge ratio failed in compute_spread_zscore: %s", e)
        beta_hat = 1.0

    # 3) Spread + Z-Score
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


# ---------- Pair prefilter (nur Engle-Granger) ----------
def pair_prefilter(
    prices: pd.DataFrame,
    *,
    coint_alpha: float = DEFAULT_COINT_ALPHA,
    min_obs: int = DEFAULT_PREFILTER_MIN_OBS,
    half_life_cfg: Mapping[str, Any] | None = None,
) -> bool:
    """
    Grobfilter für Paare: minimaler QC + Engle-Granger-Cointegrationstest.

    - erwartet DataFrame mit mind. zwei Spalten (erste = y, zweite = x)
    - numerische Säuberung + Alignment
    - Mindestlänge
    - Engle-Granger via safe_coint
    """
    result = evaluate_pair_cointegration(
        prices,
        coint_alpha=float(coint_alpha),
        min_obs=int(min_obs),
        half_life_cfg=half_life_cfg,
    )
    return bool(result.get("passed", False))
