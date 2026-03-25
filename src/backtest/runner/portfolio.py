from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from backtest.utils.io import write_json

__all__ = [
    "collect_portfolio_intents",
    "collect_portfolio_trades",
    "write_pnl_concentration_report",
]


def _gini_from_abs(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").abs().dropna().to_numpy(dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    gini = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(max(0.0, min(1.0, gini)))


def write_pnl_concentration_report(out_dir: Path, trades: pd.DataFrame) -> None:
    if (
        trades is None
        or trades.empty
        or "pair" not in trades.columns
        or "net_pnl" not in trades.columns
    ):
        return
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce")
    by_pair = pnl.groupby(trades["pair"]).sum().sort_values(ascending=False)
    if by_pair.empty:
        return
    total = float(by_pair.sum())
    abs_total = float(by_pair.abs().sum())
    top5 = by_pair.head(5)
    payload = {
        "n_pairs": int(by_pair.shape[0]),
        "total_net_pnl": float(total),
        "top5_net_pnl_sum": float(top5.sum()),
        "top5_share": float(top5.sum() / total) if total != 0.0 else 0.0,
        "top5_abs_share": float(top5.abs().sum() / abs_total)
        if abs_total > 0.0
        else 0.0,
        "gini_abs": _gini_from_abs(by_pair),
        "top5_pairs": top5.to_dict(),
    }
    write_json(out_dir / "pnl_concentration.json", payload)


def collect_portfolio_trades(portfolio: Mapping[str, Any] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, meta in (portfolio or {}).items():
        if not isinstance(meta, Mapping):
            continue
        trades = meta.get("trades")
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            frames.append(trades)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_portfolio_intents(portfolio: Mapping[str, Any] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, meta in (portfolio or {}).items():
        if not isinstance(meta, Mapping):
            continue
        intents = meta.get("intents")
        if isinstance(intents, pd.DataFrame) and not intents.empty:
            frames.append(intents.copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
