from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from backtest.config import Config
from backtest.engine import BacktestResult

EXIT_REASONS = ("normal", "stop", "timeout", "forced_window_end")


def build_wf_debug(
    result: BacktestResult,
    windows: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    bo_trials: pd.DataFrame,
    bo_best: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    best = {int(x["window"]): x for x in bo_best if "window" in x}
    rows = []
    for _, row in windows.iterrows():
        window = int(row["window"])
        test_start, test_end = pd.Timestamp(row.test_start), pd.Timestamp(row.test_end)
        selected = _window_rows(selected_pairs, window)
        trials = _window_rows(bo_trials, window)
        signals = _signals(result.positions, test_start, test_end)
        perf = _performance(result, test_start, test_end, cfg)
        quality = _trade_quality(result.trades, test_start, test_end)
        pair_count = len(selected) if not selected.empty else int(row.n_pairs)
        max_pairs = _num(row.get("max_pairs"), int)
        max_hit = max_pairs is not None and pair_count >= max_pairs

        rows.append(
            _clean(
                {
                    "window": {
                        "id": window,
                        "train_start": str(pd.Timestamp(row.train_start).date()),
                        "train_end": str(pd.Timestamp(row.train_end).date()),
                        "test_start": str(test_start.date()),
                        "test_end": str(test_end.date()),
                        "train_days": _num(row.get("train_days"), int),
                        "test_days": _num(row.get("test_days"), int),
                    },
                    "universe": {
                        "assets_before_filter": _num(
                            row.get("assets_before_filter"), int
                        ),
                        "assets_missing_sector": _num(
                            row.get("assets_missing_sector"), int
                        ),
                        "assets_after_sector": _num(row.get("assets_after_sector"), int),
                        "eligible_sector_groups": _num(
                            row.get("eligible_sector_groups"), int
                        ),
                    },
                    "selection": {
                        "selected_pair_count": int(pair_count),
                        "max_pairs": max_pairs,
                        "max_pairs_hit": max_hit,
                        "top_pairs": _top_pairs(row, selected),
                        "metrics": {
                            col: _min_median(selected, col)
                            for col in ("corr", "eg_pvalue", "half_life", "hurst")
                        },
                    },
                    "optimization": {
                        "method": "gridsearch"
                        if not trials.empty and cfg.gridsearch.enabled
                        else "bo"
                        if not trials.empty and cfg.bo.enabled
                        else "none",
                        "trials": int(len(trials)),
                        "best_score": _num(best.get(window, {}).get("score")),
                        "final_params": {
                            "entry_z": _num(row.entry_z),
                            "exit_z": _num(row.exit_z),
                            "stop_z": _num(row.stop_z),
                        },
                    },
                    "signals": signals,
                    "performance": perf,
                    "trade_quality": quality,
                    "flags": _flags(max_hit, signals, perf, quality),
                }
            )
        )
    return rows


def _window_rows(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df[df["window"].astype(int).eq(window)] if "window" in df else df.iloc[:0]


def _top_pairs(row: pd.Series, selected: pd.DataFrame) -> list[str]:
    if "pair" in selected:
        return [str(x) for x in selected["pair"].head(10)]
    return [x for x in str(row.get("pairs", "")).split(";") if x][:10]


def _min_median(df: pd.DataFrame, col: str) -> dict[str, float | None]:
    values = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else []
    return {
        "min": _num(min(values)) if len(values) else None,
        "median": _num(pd.Series(values).median()) if len(values) else None,
    }


def _signals(
    positions: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict[str, int | float]:
    pos = positions.loc[test_start:test_end]
    active = pos.ne(0).sum(axis=1)
    prev = positions.shift(1).reindex(pos.index).fillna(0)
    return {
        "active_days": int(active.gt(0).sum()),
        "active_day_ratio": float(active.gt(0).mean()),
        "entry_count": int((pos.ne(0) & prev.eq(0)).sum().sum()),
        "exit_count": int((pos.eq(0) & prev.ne(0)).sum().sum()),
        "open_positions_at_start": int(active.iloc[0]),
        "open_positions_at_end": int(active.iloc[-1]),
        "avg_concurrent_pairs": float(active.mean()),
        "max_concurrent_pairs": int(active.max()),
    }


def _performance(
    result: BacktestResult,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    cfg: Config,
) -> dict[str, float | None]:
    daily = result.daily.loc[test_start:test_end]
    returns = daily["return"].fillna(0.0)
    growth = (1.0 + returns).cumprod()
    equity = pd.concat([pd.Series([1.0]), growth], ignore_index=True)
    turnover = daily["turnover"].fillna(0.0)
    exposure = result.weights.loc[test_start:test_end].abs().sum(axis=1)
    vol = float(returns.std(ddof=0))
    return {
        "window_return": _num(growth.iloc[-1] - 1.0),
        "window_sharpe": np.sqrt(252) * float(returns.mean()) / vol if vol > 0 else 0.0,
        "local_max_drawdown": _num((equity / equity.cummax() - 1.0).min()),
        "avg_exposure": _num(exposure.mean()),
        "turnover_sum": _num(turnover.sum()),
        "turnover_avg": _num(turnover.mean()),
        "estimated_cost_impact": _num(
            turnover.sum() * float(cfg.costs.cost_bps) / 10_000.0
        ),
    }


def _trade_quality(
    trades: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict[str, int | float | None]:
    dates = pd.to_datetime(trades["exit_date"], errors="coerce") if "exit_date" in trades else pd.Series(dtype="datetime64[ns]")
    tr = trades[dates.ge(test_start) & dates.le(test_end)]
    ret = pd.to_numeric(tr["return"], errors="coerce").dropna() if "return" in tr else pd.Series(dtype=float)
    wins, losses = ret[ret > 0], ret[ret < 0]
    gross_profit, gross_loss = float(wins.sum()), float(losses.sum())
    counts = tr["exit_reason"].fillna("").astype(str).value_counts() if "exit_reason" in tr else {}
    return {
        "closed_trade_count": int(len(tr)),
        "winrate": float(len(wins) / len(tr)) if len(tr) else None,
        "avg_win": _num(wins.mean()) if len(wins) else None,
        "avg_loss": _num(losses.mean()) if len(losses) else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else None,
        "largest_win": _num(wins.max()) if len(wins) else None,
        "largest_loss": _num(losses.min()) if len(losses) else None,
        "worst_3_losses_share_of_gross_loss": _worst_loss_share(losses),
        "avg_holding_days": _num(tr["holding_days"].mean()) if "holding_days" in tr and len(tr) else None,
        **{f"{reason}_exit_count": int(counts.get(reason, 0)) for reason in EXIT_REASONS},
    }


def _worst_loss_share(losses: pd.Series) -> float | None:
    gross_loss = float(losses.sum())
    return abs(float(losses.nsmallest(3).sum())) / abs(gross_loss) if gross_loss < 0 else None


def _flags(
    max_pairs_hit: bool,
    signals: dict[str, int | float],
    perf: dict[str, float | None],
    quality: dict[str, int | float | None],
) -> list[str]:
    flags = []
    if max_pairs_hit:
        flags.append("max_pairs_hit")
    if signals["entry_count"] == 0:
        flags.append("no_entries")
    if quality["closed_trade_count"] == 0:
        flags.append("no_closed_trades")
    if signals["open_positions_at_end"] > 0:
        flags.append("open_positions_at_end")
    if quality["gross_loss"] >= 0:
        flags.append("no_losses")
    if perf["window_return"] is not None and perf["window_return"] < 0:
        flags.append("negative_window_return")
    return flags


def _num(value: Any, cast=float) -> Any:
    try:
        value = cast(value)
    except (TypeError, ValueError):
        return None
    return value if not isinstance(value, float) or math.isfinite(value) else None


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value
