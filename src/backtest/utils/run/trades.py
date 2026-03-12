from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.utils.reporting import equity_from_stats as _equity_from_stats

__all__ = [
    "_apply_global_positions_ledger",
    "_collect_portfolio_trades",
    "_collect_portfolio_intents",
    "_equity_from_stats",
    "_portfolio_from_trades",
    "_write_pnl_concentration_report",
]


def _apply_global_positions_ledger(
    trades_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if trades_df is None or trades_df.empty:
        return trades_df, pd.DataFrame(), {"kept": 0, "blocked": 0}
    df = trades_df.copy()
    if "entry_date" not in df.columns or "exit_date" not in df.columns:
        return (
            df,
            pd.DataFrame(),
            {"kept": int(len(df)), "blocked": 0, "warning": "missing entry/exit"},
        )
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"])
    if df.empty:
        return df, pd.DataFrame(), {"kept": 0, "blocked": 0}

    def _pair_key(row: pd.Series) -> str:
        p = row.get("pair")
        if p is not None and str(p).strip():
            return str(p)
        y = row.get("y_symbol") or row.get("t1_symbol") or row.get("leg1_symbol")
        x = row.get("x_symbol") or row.get("t2_symbol") or row.get("leg2_symbol")
        if y and x:
            return f"{str(y).upper()}-{str(x).upper()}"
        return "PAIR"

    df["_ledger_pair"] = df.apply(_pair_key, axis=1)
    df = df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
    open_until: dict[str, pd.Timestamp] = {}
    keep_mask = []
    blocked_rows: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        pair = str(row.get("_ledger_pair", "PAIR"))
        entry = pd.Timestamp(row["entry_date"])
        exit_ts = pd.Timestamp(row["exit_date"])
        last_exit = open_until.get(pair)
        if last_exit is not None and entry <= last_exit:
            keep_mask.append(False)
            blocked_rows.append(pos)
            continue
        keep_mask.append(True)
        open_until[pair] = exit_ts

    kept = df.loc[keep_mask].drop(columns=["_ledger_pair"])
    blocked = (
        df.loc[blocked_rows].drop(columns=["_ledger_pair"])
        if blocked_rows
        else pd.DataFrame()
    )
    report = {"kept": int(len(kept)), "blocked": int(len(blocked))}
    return kept, blocked, report


def _gini_from_abs(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").abs().dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    g = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(max(0.0, min(1.0, g)))


def _write_pnl_concentration_report(out_dir: Path, trades: pd.DataFrame) -> None:
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
    (out_dir / "pnl_concentration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _collect_portfolio_trades(portfolio: Mapping[str, Any] | None) -> pd.DataFrame:
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


def _collect_portfolio_intents(portfolio: Mapping[str, Any] | None) -> pd.DataFrame:
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


def _portfolio_from_trades(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades is None or trades.empty:
        return {}
    df = trades.copy()
    if "pair" not in df.columns:
        return {}
    portfolio: dict[str, dict[str, Any]] = {}
    for pair, grp in df.groupby("pair"):
        key = str(pair)
        if not key or key.lower() == "nan":
            continue
        portfolio[key] = {"trades": grp.copy()}
    return portfolio
