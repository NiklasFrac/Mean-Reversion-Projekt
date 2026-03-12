# src/backtest/research/overfit.py
"""
Overfit analysis (config-independent)

Features:
- Pure, deterministic functions without global state
- Robust numerics (NaN/Inf handling, protection against division by zero)
- Mypy-/Ruff-clean typing and API
- CPCV-PBO, (deflated) Sharpe, Memmel test statistic
- Defensive, tolerant I/O (CSV/Parquet) without depending on config.yaml
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Optional SciPy (preferred for accurate distribution CDF/PPF). If unavailable,
# fall back to lightweight approximations to keep the backtest runnable.
try:  # pragma: no cover
    from scipy.stats import norm as _norm  # type: ignore[import-untyped]
    from scipy.stats import t as _student_t

    def _norm_cdf(z: float) -> float:
        return float(_norm.cdf(z))

    def _norm_ppf(p: float) -> float:
        return float(_norm.ppf(p))

    def _student_t_cdf(x: float, *, df: int) -> float:
        return float(_student_t.cdf(x, df=max(1, int(df))))

except Exception:  # pragma: no cover

    def _norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))

    # Acklam inverse normal CDF approximation (good accuracy for p in (0,1))
    def _norm_ppf(p: float) -> float:
        p = float(p)
        if not math.isfinite(p):
            return float("nan")
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            return float(num / den)
        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
            den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
            return float(-num / den)
        q = p - 0.5
        r = q * q
        num = ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        return float(num * q / den)

    def _student_t_cdf(x: float, *, df: int) -> float:
        # Normal approximation (coarse but monotonic); keeps metrics usable without SciPy.
        return _norm_cdf(float(x))


__all__ = [
    "OverfitSummary",
    "summarize_overfit_from_equity",
    "pbo_cpcv",
    "write_overfit_report",
    "analyze_bo_trials",
]

# ---- Typaliases ----------------------------------------------------------------

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


# ---- Small, robust utils -------------------------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Robustly converts to float and returns default on errors/non-finite values."""
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _finite_or_default(a: FloatArray, default: float = 0.0) -> FloatArray:
    """Replaces non-finite values with default."""
    out = np.array(a, dtype=float, copy=True)
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = default
    return out


def _to_float_array(seq: Iterable[float] | FloatArray) -> FloatArray:
    """Converts to FloatArray; non-finite values -> 0.0."""
    arr = np.asarray(list(seq) if not isinstance(seq, np.ndarray) else seq, dtype=float)
    return _finite_or_default(arr, default=0.0)


# ---- Metrics / tests -----------------------------------------------------------


def _returns_from_equity(equity: pd.Series) -> pd.Series:
    """Robust returns (pct_change) from equity."""
    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r.astype(float)


def _sharpe_ratio_from_returns(returns: pd.Series, *, trading_days: int = 252) -> float:
    """
    Annualized Sharpe ratio from a returns series.
    Non-finite values are removed; sd==0 -> 0.0.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    mu = r.mean() * trading_days
    sd = r.std(ddof=1) * math.sqrt(trading_days)
    if not math.isfinite(sd) or sd == 0.0:
        return 0.0
    return _safe_float(mu / sd)


def _memmel_t_stat(sr: float, n: int, *, skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Memmel-adjusted t-statistic for the Sharpe ratio.
    Optionally includes skewness and kurtosis of returns.
    """
    if n <= 1 or not math.isfinite(sr):
        return 0.0
    denom_sq = max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr * sr))
    return _safe_float(sr * math.sqrt(n) / math.sqrt(denom_sq))


def _p_one_sided_from_sr(
    sr: float, n: int, *, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """
    One-sided p-value for SR>0 under a t-distribution with Memmel correction.
    Clipped in [0, 1].
    """
    t_val = _memmel_t_stat(sr, n, skew=skew, kurt=kurt)
    df = max(1, n - 1)
    p_one = 1.0 - float(_student_t_cdf(float(t_val), df=int(df)))
    return float(np.clip(p_one, 0.0, 1.0))


def _sample_skew_kurtosis(returns: pd.Series) -> tuple[float, float]:
    """Skewness/kurtosis (Pearson, not excess) from returns."""
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(r.size)
    if n < 3:
        return 0.0, 3.0
    mu = float(np.mean(r))
    diff = r - mu
    m2 = float(np.mean(diff**2))
    if not math.isfinite(m2) or m2 <= 0.0:
        return 0.0, 3.0
    m3 = float(np.mean(diff**3))
    m4 = float(np.mean(diff**4))
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2)
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    return float(skew), float(kurt)


def _sigma_sr(sr_hat: float, n: int, *, skew: float, kurt: float) -> float:
    if n <= 1:
        return 0.0
    denom_sq = max(
        1e-12, 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat * sr_hat)
    )
    return float(math.sqrt(denom_sq / max(1, n - 1)))


def _psr(
    sr_hat: float, n: int, *, skew: float, kurt: float, sr_ref: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio (PSR) per Lopez de Prado."""
    if not math.isfinite(sr_hat) or n <= 1:
        return 0.0
    sigma_sr = _sigma_sr(sr_hat, n, skew=skew, kurt=kurt)
    if sigma_sr <= 0.0 or not math.isfinite(sigma_sr):
        return 0.0 if sr_hat <= sr_ref else 1.0
    z = (sr_hat - sr_ref) / sigma_sr
    return float(np.clip(_norm_cdf(float(z)), 0.0, 1.0))


def _expected_max_sr(n_trials: int, *, sigma_sr: float) -> float:
    """Expected max SR among N trials (EVT approximation)."""
    n = int(max(1, n_trials))
    if n <= 1 or sigma_sr <= 0.0 or not math.isfinite(sigma_sr):
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni
    p1 = 1.0 - 1.0 / float(n)
    p2 = 1.0 - 1.0 / float(n * math.e)
    z1 = _norm_ppf(p1)
    z2 = _norm_ppf(p2)
    return float(sigma_sr * ((1.0 - gamma) * z1 + gamma * z2))


def _deflated_sharpe_ratio(
    sr_hat: float,
    *,
    n: int,
    n_trials: int,
    skew: float,
    kurt: float,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) per Lopez de Prado.
    Uses expected max SR among N trials as the reference threshold.
    """
    if not math.isfinite(sr_hat) or n <= 1:
        return 0.0
    sigma_sr = _sigma_sr(sr_hat, n, skew=skew, kurt=kurt)
    if sigma_sr <= 0.0 or not math.isfinite(sigma_sr):
        return 0.0
    sr_max = _expected_max_sr(int(n_trials), sigma_sr=sigma_sr)
    z = (sr_hat - sr_max) / sigma_sr
    return float(np.clip(_norm_cdf(float(z)), 0.0, 1.0))


# ---- Public API ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OverfitSummary:
    """
    Compact overfit analysis report.
    - dsr:    Deflated Sharpe (0..1, higher = more credible)
    - memmel_p_one_sided: one-sided p(SR<=0) with Memmel correction
    - n_test_obs: number of OOS PnL observations
    - candidates: number of tested candidates (for deflation)
    - pbo:    Probability of Backtest Overfitting from CPCV
    - n_folds: number of CPCV folds
    """

    dsr: float
    memmel_p_one_sided: float
    n_test_obs: int
    candidates: int
    pbo: float = 0.0
    n_folds: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "dsr": float(self.dsr),
            "memmel_p_one_sided": float(self.memmel_p_one_sided),
            "n_test_obs": int(self.n_test_obs),
            "n_candidates": int(self.candidates),
            "pbo": float(self.pbo),
            "n_folds": int(self.n_folds),
        }
        if self.meta:
            data["meta"] = self.meta
        return data


def summarize_overfit_from_equity(
    equity_curve_test_only: pd.Series,
    *,
    candidate_sharpes: Sequence[float] | None = None,
    trading_days: int = 252,
) -> OverfitSummary:
    """
    Computes SR/DSR from a TEST/OOS equity curve and optional candidate SRs.
    """
    equity = pd.to_numeric(equity_curve_test_only, errors="coerce").dropna()
    rets = _returns_from_equity(equity)
    n = int(rets.shape[0])
    cand_vals: list[float] = []
    for x in candidate_sharpes or []:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isfinite(v):
            cand_vals.append(v)
    n_trials = max(1, int(len(set(cand_vals))) if cand_vals else 1)

    if n == 0:
        return OverfitSummary(
            dsr=0.0,
            memmel_p_one_sided=0.0,
            n_test_obs=0,
            candidates=len(candidate_sharpes or []),
            meta={"warnings": ["empty_equity"], "n_trials": int(n_trials)},
        )

    sr = _sharpe_ratio_from_returns(rets, trading_days=trading_days)
    skew, kurt = _sample_skew_kurtosis(rets)
    p_one = _p_one_sided_from_sr(sr, n, skew=skew, kurt=kurt)
    dsr = _deflated_sharpe_ratio(sr, n=n, n_trials=n_trials, skew=skew, kurt=kurt)
    psr = _psr(sr, n, skew=skew, kurt=kurt, sr_ref=0.0)

    return OverfitSummary(
        dsr=float(np.clip(dsr, 0.0, 1.0)),
        memmel_p_one_sided=float(np.clip(p_one, 0.0, 1.0)),
        n_test_obs=n,
        candidates=len(candidate_sharpes or []),
        meta={
            "sr_hat": float(sr),
            "psr": float(np.clip(psr, 0.0, 1.0)),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "n_trials": int(n_trials),
        },
    )


# ---- CPCV / PBO ----------------------------------------------------------------


def pbo_cpcv(is_scores: FloatArray, oos_scores: FloatArray) -> float:
    """
    Probability of Backtest Overfitting (CPCV) per Bailey et al.:
      - Select best model by IS within each fold.
      - Compute its OOS rank percentile w in (0,1).
      - Lambda = logit(w). PBO = P(lambda < 0) across folds.

    Expects arrays of shape (n_folds, n_models).
    NaNs/Inf are conservatively mapped to -inf (i.e. never the "best" choice).
    """
    if is_scores.shape != oos_scores.shape or is_scores.size == 0:
        return 0.0

    is_arr = _finite_or_default(np.asarray(is_scores, dtype=float), default=-np.inf)
    oos_arr = _finite_or_default(np.asarray(oos_scores, dtype=float), default=-np.inf)

    n_folds, n_models = is_arr.shape
    if n_models <= 1:
        return 0.0

    is_best: IntArray = np.argmax(is_arr, axis=1)

    # ranks: 0=worst ... n_models-1=best
    order = np.argsort(oos_arr, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n_folds).reshape(-1, 1)
    ranks[rows, order] = np.arange(n_models)

    # convert to 1=best ... n_models=worst
    rank_best = (n_models - ranks[np.arange(n_folds), is_best]).astype(float)
    w = (rank_best - 0.5) / float(n_models)
    w = np.clip(w, 1e-9, 1.0 - 1e-9)
    lamb = np.log(w / (1.0 - w))
    return float(np.mean(lamb < 0.0)) if lamb.size else 0.0


# ---- I/O -----------------------------------------------------------------------


def write_overfit_report(path: Path, summary: OverfitSummary) -> Path:
    """
    Writes a compact JSON report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ---- BO-Adapter / Batch-Analyse ------------------------------------------------


def _read_frame(p: Path) -> pd.DataFrame:
    """
    Tolerantly reads CSV/Parquet. Missing or invalid files -> empty DF.
    """
    try:
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() in {".parq", ".parquet"}:
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    """Case-insensitive Column-Map (lower->original)."""
    return {c.lower(): c for c in df.columns}


def analyze_bo_trials(
    trials_csv_or_list: str | Path | Sequence[str | Path],
    *,
    aggregate: Literal["median", "mean", "max"] = "median",
    equity_curve: pd.Series | None = None,
    out_path: str | Path = "runs/results/overfit/overfit_summary.json",
    trading_days: int = 252,
    component_filter: str | Sequence[str] | None = None,
    metric_filter: str | Sequence[str] | None = None,
    **_: Any,
) -> Path:
    """
    - Accepts a path OR a list of paths (CSV/Parquet)
    - Candidates: columns 'score' or 'sharpe'
    - Optional: CPCV fields -> PBO ('fold', 'is_sharpe/oos_sharpe' or 'is_score/oos_score')
               + 'model_id' | 'candidate' | 'name' for pivoting/grouping
    - Optional: component/metric filters to avoid mixing stages
    - Writes a JSON report (compatible with the runner)
    """
    if isinstance(trials_csv_or_list, (str, Path)):
        paths = [Path(trials_csv_or_list)]
    else:
        paths = [Path(x) for x in trials_csv_or_list]

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = _read_frame(p)
        if not df.empty:
            frames.append(df)

    cand_sr: list[float] = []
    pbo = 0.0
    n_folds = 0
    meta_warnings: list[str] = []
    meta_component: str | None = None
    meta_metric: str | None = None
    n_trials = 0

    if frames:
        df_all = pd.concat(frames, axis=0, ignore_index=True)
        if not df_all.empty:
            cols = _normalize_columns(df_all)

            # Optional component/metric filtering to avoid mixing stages.
            comp_col = cols.get("component")
            metric_col = cols.get("metric")
            df_use = df_all
            if comp_col and comp_col in df_all:
                comps = [
                    c for c in df_all[comp_col].dropna().astype(str).unique().tolist()
                ]
                comp_filter: list[str] | None = None
                if component_filter is not None:
                    if isinstance(component_filter, (list, tuple)):
                        comp_filter = [str(x) for x in component_filter]
                    else:
                        comp_filter = [str(component_filter)]
                if comp_filter:
                    df_use = df_use[df_use[comp_col].astype(str).isin(comp_filter)]
                    meta_component = ",".join(comp_filter)
                elif len(comps) > 1:
                    counts = df_all[comp_col].astype(str).value_counts()
                    if not counts.empty:
                        meta_component = str(counts.index[0])
                        df_use = df_use[df_use[comp_col].astype(str) == meta_component]
                        meta_warnings.append("component_ambiguous_defaulted")
            if metric_col and metric_col in df_use:
                mets = [
                    m for m in df_use[metric_col].dropna().astype(str).unique().tolist()
                ]
                met_filter: list[str] | None = None
                if metric_filter is not None:
                    if isinstance(metric_filter, (list, tuple)):
                        met_filter = [str(x) for x in metric_filter]
                    else:
                        met_filter = [str(metric_filter)]
                if met_filter:
                    df_use = df_use[df_use[metric_col].astype(str).isin(met_filter)]
                    meta_metric = ",".join(met_filter)
                elif len(mets) > 1:
                    counts = df_use[metric_col].astype(str).value_counts()
                    if not counts.empty:
                        meta_metric = str(counts.index[0])
                        df_use = df_use[df_use[metric_col].astype(str) == meta_metric]
                        meta_warnings.append("metric_ambiguous_defaulted")

            # Candidates (score/Sharpe)
            score_col = cols.get("score") or cols.get("sharpe")
            model_col = (
                cols.get("model_id") or cols.get("candidate") or cols.get("name")
            )

            if score_col is not None and score_col in df_use:
                df_scores = pd.DataFrame(
                    {
                        "score": pd.to_numeric(df_use[score_col], errors="coerce"),
                        "model": df_use[model_col] if model_col in df_use else None,
                    }
                ).dropna(subset=["score"])

                if not df_scores.empty:
                    if model_col in df_use:
                        if aggregate == "median":
                            grouped = df_scores.groupby("model", dropna=False)[
                                "score"
                            ].median()
                        elif aggregate == "mean":
                            grouped = df_scores.groupby("model", dropna=False)[
                                "score"
                            ].mean()
                        else:
                            grouped = df_scores.groupby("model", dropna=False)[
                                "score"
                            ].max()
                        cand_sr = [float(x) for x in grouped.to_numpy(dtype=float)]
                        n_trials = int(grouped.shape[0])
                    else:
                        cand_sr = [
                            float(x) for x in df_scores["score"].to_numpy(dtype=float)
                        ]
                        n_trials = int(df_scores.shape[0])

            # CPCV-PBO
            fold_col = cols.get("fold") or cols.get("fold_id") or cols.get("cv_fold")
            is_col = cols.get("is_sharpe") or cols.get("is_score")
            oos_col = cols.get("oos_sharpe") or cols.get("oos_score")

            if fold_col and is_col and oos_col and model_col and fold_col in df_use:
                try:
                    df_pbo = df_use.dropna(
                        subset=[fold_col, model_col, is_col, oos_col]
                    )
                    piv_is = (
                        df_pbo.pivot_table(
                            index=fold_col,
                            columns=model_col,
                            values=is_col,
                            aggfunc="mean",
                        )
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                        .to_numpy(dtype=float)
                    )
                    piv_oos = (
                        df_pbo.pivot_table(
                            index=fold_col,
                            columns=model_col,
                            values=oos_col,
                            aggfunc="mean",
                        )
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                        .to_numpy(dtype=float)
                    )
                    pbo = pbo_cpcv(piv_is, piv_oos)
                    n_folds = int(piv_is.shape[0]) if piv_is.ndim == 2 else 0
                except Exception:
                    pbo = 0.0
                    n_folds = 0

    # --- DSR/summary berechnen
    if equity_curve is not None and len(equity_curve) > 3:
        eq = pd.to_numeric(equity_curve, errors="coerce").dropna()
        rets = _returns_from_equity(eq)
        n = int(rets.shape[0])
        sr = _sharpe_ratio_from_returns(rets, trading_days=trading_days)
        skew, kurt = _sample_skew_kurtosis(rets)
        p_one = _p_one_sided_from_sr(sr, n, skew=skew, kurt=kurt)
        n_trials_eff = int(n_trials) if n_trials > 0 else max(1, len(cand_sr))
        dsr = _deflated_sharpe_ratio(
            sr, n=n, n_trials=n_trials_eff, skew=skew, kurt=kurt
        )
        psr = _psr(sr, n, skew=skew, kurt=kurt, sr_ref=0.0)
        summary = OverfitSummary(
            dsr=float(np.clip(dsr, 0.0, 1.0)),
            memmel_p_one_sided=float(np.clip(p_one, 0.0, 1.0)),
            n_test_obs=n,
            candidates=len(cand_sr),
            pbo=float(pbo),
            n_folds=int(n_folds),
            meta={
                "sr_hat": float(sr),
                "psr": float(np.clip(psr, 0.0, 1.0)),
                "skew": float(skew),
                "kurtosis": float(kurt),
                "n_trials": int(n_trials_eff),
                "component": meta_component,
                "metric": meta_metric,
                "warnings": meta_warnings,
            },
        )
    else:
        summary = OverfitSummary(
            dsr=0.0,
            memmel_p_one_sided=0.0,
            n_test_obs=0,
            candidates=len(cand_sr),
            pbo=float(pbo),
            n_folds=int(n_folds),
            meta={
                "component": meta_component,
                "metric": meta_metric,
                "warnings": (meta_warnings + ["missing_equity"])
                if meta_warnings
                else ["missing_equity"],
            },
        )

    out = Path(out_path)
    write_overfit_report(out, summary)
    return out
