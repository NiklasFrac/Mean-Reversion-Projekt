from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.config import CostsConfig, RiskConfig


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, float]


def run_engine(
    prices: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    betas: pd.DataFrame,
    positions: pd.DataFrame,
    zscores: pd.DataFrame,
    *,
    initial_capital: float,
    costs: CostsConfig,
    risk: RiskConfig,
) -> BacktestResult:
    idx = positions.index
    weights = positions.astype(float) * float(risk.max_pair_weight)
    pair_return_series: dict[str, pd.Series] = {}
    for pair, (y_name, x_name) in pairs.items():
        if pair not in positions or pair not in betas:
            continue
        y = prices[y_name].reindex(idx).pct_change(fill_method=None)
        x = prices[x_name].reindex(idx).pct_change(fill_method=None)
        beta = betas[pair].reindex(idx).shift(1)
        hedge_ret = (y - beta * x) / (1.0 + beta.abs())
        pair_return_series[pair] = weights[pair].shift(1).fillna(0.0) * hedge_ret
    pair_returns = (
        pd.concat(pair_return_series, axis=1)
        if pair_return_series
        else pd.DataFrame(index=idx)
    )

    ret = pair_returns.sum(axis=1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    ret -= turnover * (float(costs.fee_bps) + float(costs.slippage_bps)) / 10_000.0
    daily = pd.DataFrame({"return": ret, "turnover": turnover}, index=idx)
    daily["equity"] = float(initial_capital) * (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    trade_rows = []
    for pair in positions.columns:
        changes = positions[pair].diff().fillna(positions[pair]).ne(0)
        for ts in positions.index[changes]:
            trade_rows.append(
                {
                    "date": ts,
                    "pair": pair,
                    "position": int(positions.at[ts, pair]),
                    "z": float(zscores.at[ts, pair]) if pair in zscores else np.nan,
                }
            )
    trades = pd.DataFrame(trade_rows)
    if daily.empty:
        summary = {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "trades": 0, "winrate": 0.0}
        return BacktestResult(daily, positions, weights, trades, summary)

    ret = daily["return"].fillna(0.0)
    active_ret = ret[ret.ne(0.0)]
    vol = ret.std(ddof=0)
    sharpe = float(np.sqrt(252) * ret.mean() / vol) if vol > 0 else 0.0
    summary = {
        "total_return": float(daily["equity"].iloc[-1] / daily["equity"].iloc[0] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(daily["drawdown"].min()),
        "trades": int(len(trades)),
        "winrate": float(active_ret.gt(0.0).mean()) if not active_ret.empty else 0.0,
    }
    return BacktestResult(daily, positions, weights, trades, summary)
