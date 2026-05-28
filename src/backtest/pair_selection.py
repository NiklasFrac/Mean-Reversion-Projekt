from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backtest.config import PairSelectionConfig
from backtest.walkforward import Window


@dataclass(frozen=True)
class PairMeta:
    y: str
    x: str
    alpha: float
    beta: float
    half_life: float


def select_pairs(
    prices: pd.DataFrame,
    window: Window,
    cfg: PairSelectionConfig,
    sectors: pd.Series | None = None,
) -> tuple[dict[str, PairMeta], pd.DataFrame]:
    train = prices.loc[window.train_start : window.train_end].dropna(axis=1)
    train = train.loc[:, (train > 0).all(axis=0)]
    if sectors is not None:
        sector_map = sectors.reindex(train.columns).dropna()
        train = train.loc[:, sector_map.index]
    if train.shape[1] < 2:
        return {}, pd.DataFrame()

    rows = []
    groups = (
        [(None, list(map(str, train.columns)))]
        if sectors is None
        else [
            (sector, list(cols))
            for sector, cols in sector_map.groupby(sector_map).groups.items()
        ]
    )
    for sector, cols in groups:
        if len(cols) < 2:
            continue
        returns = train[cols].pct_change(fill_method=None).dropna(how="any")
        corr = returns.corr()
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                c = corr.at[a, b]
                if pd.notna(c) and cfg.min_corr <= float(c) <= cfg.max_corr:
                    row = _test_pair(
                        np.log(train[a]), np.log(train[b]), a, b, float(c), cfg
                    )
                    if row:
                        if sector is not None:
                            row["sector"] = sector
                        rows.append(row)

    if not rows:
        return {}, pd.DataFrame()
    df = pd.DataFrame(rows)
    df["eg_qvalue"] = _bh_qvalues(df["eg_pvalue"]) if cfg.fdr_enabled else np.nan
    df = df[df.apply(lambda row: _passes(row, cfg), axis=1)]
    if df.empty:
        return {}, pd.DataFrame()
    df = df.sort_values(
        ["eg_qvalue", "eg_pvalue", "hurst", "half_life", "corr"],
        ascending=[True, True, True, True, False],
        na_position="last",
    )
    df = df.head(int(cfg.max_pairs))
    pairs = {
        r.pair: PairMeta(r.y, r.x, float(r.alpha), float(r.beta), float(r.half_life))
        for r in df.itertuples(index=False)
    }
    return pairs, df.reset_index(drop=True)


def _test_pair(
    y: pd.Series,
    x: pd.Series,
    y_name: str,
    x_name: str,
    corr: float,
    cfg: PairSelectionConfig,
):
    candidates = []
    for left, right, lname, rname in ((y, x, y_name, x_name), (x, y, x_name, y_name)):
        try:
            pvalue = float(coint(left, right)[1])
        except Exception:
            continue
        spread, alpha, beta = _spread(left, right)
        candidates.append((pvalue, spread, alpha, beta, lname, rname))
    if not candidates:
        return None
    pvalue, spread, alpha, beta, y_name, x_name = min(candidates, key=lambda v: v[0])
    half_life = _half_life(spread)
    return {
        "pair": f"{y_name}-{x_name}",
        "y": y_name,
        "x": x_name,
        "corr": corr,
        "eg_pvalue": pvalue,
        "alpha": alpha,
        "beta": beta,
        "half_life": half_life,
        "hurst": _hurst(spread) if cfg.use_hurst_filter else np.nan,
    }


def _spread(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float, float]:
    mat = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    alpha, beta = np.linalg.lstsq(mat, y.to_numpy(float), rcond=None)[0]
    return y - (alpha + beta * x), float(alpha), float(beta)


def _half_life(spread: pd.Series) -> float:
    df = pd.DataFrame({"lag": spread.shift(1), "delta": spread.diff()}).dropna()
    if len(df) < 3:
        return np.nan
    mat = np.column_stack([np.ones(len(df)), df["lag"].to_numpy(float)])
    lam = float(np.linalg.lstsq(mat, df["delta"].to_numpy(float), rcond=None)[0][1])
    return float(-np.log(2) / lam) if np.isfinite(lam) and lam < 0 else np.nan


def _hurst(spread: pd.Series) -> float:
    s = spread.dropna().to_numpy(float)
    max_lag = min(20, len(s) // 4)
    if max_lag < 3:
        return np.nan
    lags = np.arange(2, max_lag + 1)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    ok = tau > 0
    return (
        float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])
        if ok.sum() >= 2
        else np.nan
    )


def _passes(row: dict, cfg: PairSelectionConfig) -> bool:
    fdr_ok = (
        not cfg.fdr_enabled
        or (np.isfinite(row["eg_qvalue"]) and row["eg_qvalue"] <= cfg.fdr_alpha)
    )
    half_life_ok = not cfg.use_half_life_filter or (
        np.isfinite(row["half_life"])
        and cfg.min_half_life <= row["half_life"] <= cfg.max_half_life
    )
    hurst_ok = not cfg.use_hurst_filter or (
        np.isfinite(row["hurst"]) and row["hurst"] <= cfg.max_hurst
    )
    return (
        np.isfinite(row["eg_pvalue"])
        and row["eg_pvalue"] <= cfg.max_eg_pvalue
        and fdr_ok
        and np.isfinite(row["alpha"])
        and np.isfinite(row["beta"])
        and row["beta"] > 0
        and half_life_ok
        and hurst_ok
    )


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    q = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return pd.Series(q, index=pvalues.index)

    order = np.argsort(p[ok])
    ranked = p[ok][order]
    n = float(len(ranked))
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, len(ranked) + 1))[::-1])[
        ::-1
    ]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    ok_idx = np.flatnonzero(ok)
    q[ok_idx[order]] = adjusted
    return pd.Series(q, index=pvalues.index)
