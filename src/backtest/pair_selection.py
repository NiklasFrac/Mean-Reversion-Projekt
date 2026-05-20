from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from backtest.config import PairSelectionConfig
from backtest.walkforward import Window


def select_pairs(
    prices: pd.DataFrame, window: Window, cfg: PairSelectionConfig
) -> tuple[dict[str, tuple[str, str]], pd.DataFrame]:
    train = prices.loc[window.train_start : window.train_end].dropna(axis=1)
    train = train.loc[:, (train > 0).all(axis=0)]
    if len(train) < cfg.min_obs or train.shape[1] < 2:
        return {}, pd.DataFrame()

    returns = train.pct_change(fill_method=None).dropna(how="any")
    corr = returns.corr()
    rows = []
    cols = list(map(str, train.columns))
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = corr.at[a, b]
            if pd.notna(c) and float(c) >= cfg.min_corr:
                row = _test_pair(np.log(train[a]), np.log(train[b]), a, b, float(c))
                if row and _passes(row, cfg):
                    rows.append(row)

    if not rows:
        return {}, pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["eg_pvalue", "hurst", "half_life", "corr"],
        ascending=[True, True, True, False],
    )
    df = df.head(int(cfg.max_pairs))
    pairs = {r.pair: (r.y, r.x) for r in df.itertuples(index=False)}
    return pairs, df.reset_index(drop=True)


def _test_pair(y: pd.Series, x: pd.Series, y_name: str, x_name: str, corr: float):
    candidates = []
    for left, right, lname, rname in ((y, x, y_name, x_name), (x, y, x_name, y_name)):
        try:
            pvalue = float(coint(left, right)[1])
        except Exception:
            continue
        spread, beta = _spread(left, right)
        candidates.append((pvalue, spread, beta, lname, rname))
    if not candidates:
        return None
    pvalue, spread, beta, y_name, x_name = min(candidates, key=lambda v: v[0])
    return {
        "pair": f"{y_name}-{x_name}",
        "y": y_name,
        "x": x_name,
        "corr": corr,
        "eg_pvalue": pvalue,
        "beta": beta,
        "half_life": _half_life(spread),
        "hurst": _hurst(spread),
    }


def _spread(y: pd.Series, x: pd.Series) -> tuple[pd.Series, float]:
    mat = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    alpha, beta = np.linalg.lstsq(mat, y.to_numpy(float), rcond=None)[0]
    return y - (alpha + beta * x), float(beta)


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
    return (
        np.isfinite(row["eg_pvalue"])
        and row["eg_pvalue"] <= cfg.max_eg_pvalue
        and np.isfinite(row["beta"])
        and row["beta"] > 0
        and np.isfinite(row["half_life"])
        and cfg.min_half_life <= row["half_life"] <= cfg.max_half_life
        and np.isfinite(row["hurst"])
        and row["hurst"] <= cfg.max_hurst
    )
