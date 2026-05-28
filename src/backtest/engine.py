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
    exit_reasons: pd.DataFrame | None = None,
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
    ret -= turnover * float(costs.cost_bps) / 10_000.0
    pair_costs = weights.diff().abs().fillna(0.0) * float(costs.cost_bps) / 10_000.0
    trade_returns = pair_returns.sub(pair_costs, fill_value=0.0)
    daily = pd.DataFrame({"return": ret, "turnover": turnover}, index=idx)
    daily["equity"] = float(initial_capital) * (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    trades = _closed_trades(positions, trade_returns, zscores, exit_reasons)
    if daily.empty:
        summary = _summary(daily, weights, trades)
        return BacktestResult(daily, positions, weights, trades, summary)

    ret = daily["return"].fillna(0.0)
    vol = ret.std(ddof=0)
    sharpe = float(np.sqrt(252) * ret.mean() / vol) if vol > 0 else 0.0
    summary = _summary(daily, weights, trades) | {"sharpe": sharpe}
    return BacktestResult(daily, positions, weights, trades, summary)


def _closed_trades(
    positions: pd.DataFrame,
    pair_returns: pd.DataFrame,
    zscores: pd.DataFrame,
    exit_reasons: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    for pair in positions.columns:
        entry_date, side, entry_z = None, 0, np.nan
        for ts, pos in positions[pair].astype(int).items():
            if side == 0 and pos != 0:
                entry_date, side = ts, int(pos)
                entry_z = float(zscores.at[ts, pair]) if pair in zscores else np.nan
            elif side != 0 and pos == 0 and entry_date is not None:
                returns = pair_returns.get(pair, pd.Series(0.0, index=positions.index))
                trade_return = float((1.0 + returns.loc[entry_date:ts]).prod() - 1.0)
                exit_reason = None
                if exit_reasons is not None and pair in exit_reasons:
                    raw_reason = exit_reasons.at[ts, pair]
                    if pd.notna(raw_reason):
                        exit_reason = str(raw_reason)
                rows.append(
                    {
                        "pair": pair,
                        "entry_date": entry_date,
                        "exit_date": ts,
                        "side": side,
                        "entry_z": entry_z,
                        "exit_z": float(zscores.at[ts, pair])
                        if pair in zscores
                        else np.nan,
                        "return": trade_return,
                        "holding_days": int(
                            positions.loc[entry_date:ts, pair].iloc[:-1].ne(0).sum()
                        ),
                        "exit_reason": exit_reason,
                    }
                )
                entry_date, side, entry_z = None, 0, np.nan
    return pd.DataFrame(rows)


def _summary(
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, float]:
    trade_ret = trades["return"] if "return" in trades else pd.Series(dtype=float)
    wins = trade_ret[trade_ret.gt(0.0)]
    losses = trade_ret[trade_ret.lt(0.0)]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    return {
        "total_return": float(daily["equity"].iloc[-1] / daily["equity"].iloc[0] - 1.0)
        if not daily.empty
        else 0.0,
        "sharpe": 0.0,
        "max_drawdown": float(daily["drawdown"].min()) if not daily.empty else 0.0,
        "n_trades": int(len(trades)),
        "trade_winrate": float(wins.size / len(trades)) if len(trades) else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else np.inf,
        "avg_holding_days": float(trades["holding_days"].mean())
        if "holding_days" in trades and not trades.empty
        else 0.0,
        "avg_exposure": float(weights.abs().sum(axis=1).mean())
        if not weights.empty
        else 0.0,
    }
