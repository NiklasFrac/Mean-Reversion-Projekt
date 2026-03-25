from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtest.config.cfg import AppConfig, config_to_dict
from backtest.reporting.tearsheet import summarize_stats
from backtest.reporting.common import equity_from_stats
from backtest.strat.registry import build_strategy_from_cfg
from backtest.utils.io import write_json
from backtest.utils.prices import as_price_map


@dataclass(frozen=True)
class TrainRefitArtifacts:
    train_start: str
    train_end: str
    n_pairs: int
    wf_i: int | None
    stats: pd.DataFrame
    trades: pd.DataFrame
    entry_intents: pd.DataFrame
    state_transitions: pd.DataFrame
    equity: pd.Series
    summary: dict[str, Any]


def _read_train_only_equity(train_dir: Path) -> pd.Series | None:
    for name in ("train_equity.csv", "equity_curve_train.csv", "stats.csv"):
        path = Path(train_dir) / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "date" not in df.columns or "equity" not in df.columns:
            continue
        try:
            idx = pd.to_datetime(df["date"], errors="coerce", utc=True)
        except Exception:
            idx = pd.to_datetime(df["date"], errors="coerce", utc=True)
        out = pd.Series(
            pd.to_numeric(df["equity"], errors="coerce").to_numpy(dtype=float),
            index=idx,
            name="equity",
        )
        out = out.loc[~out.index.isna()]
        if isinstance(out.index, pd.DatetimeIndex):
            out = out[~out.index.duplicated(keep="last")].sort_index()
        if not out.empty:
            return out
    return None


def _train_only_cfg_for_strategy(
    cfg_eff: AppConfig | Mapping[str, Any],
) -> dict[str, Any] | None:
    if isinstance(cfg_eff, AppConfig):
        cfg_eff = config_to_dict(cfg_eff)

    bt = (
        cfg_eff.get("backtest", {})
        if isinstance(cfg_eff.get("backtest"), Mapping)
        else {}
    )
    splits_raw = bt.get("splits")
    splits = splits_raw if isinstance(splits_raw, Mapping) else {}
    train_raw = splits.get("train")
    train = train_raw if isinstance(train_raw, Mapping) else None
    if not isinstance(train, Mapping) or not train.get("start") or not train.get("end"):
        return None
    train_start = str(train.get("start"))
    train_end = str(train.get("end"))
    cfg_train = dict(cfg_eff)
    bt2 = dict(cfg_train.get("backtest") or {})
    splits2 = dict(bt2.get("splits") or {})
    splits2["train"] = {"start": train_start, "end": train_end}
    splits2["test"] = {
        "start": train_start,
        "end": train_end,
        "entry_end": train_end,
        "exit_end": train_end,
    }
    bt2["splits"] = splits2
    cfg_train["backtest"] = bt2
    return cfg_train


def _train_only_cfg_for_engine(cfg_train: Mapping[str, Any]) -> dict[str, Any] | None:
    bt = (
        cfg_train.get("backtest", {})
        if isinstance(cfg_train.get("backtest"), dict)
        else {}
    )
    splits_raw = bt.get("splits")
    splits = splits_raw if isinstance(splits_raw, Mapping) else {}
    test_raw = splits.get("test")
    test = test_raw if isinstance(test_raw, Mapping) else None
    if not isinstance(test, Mapping) or not test.get("start") or not test.get("end"):
        return None

    test_start = str(test.get("start"))
    test_end = str(test.get("end"))
    dummy_start = "1900-01-01"
    dummy_end = "1900-01-01"
    try:
        t0 = pd.to_datetime(test_start, errors="coerce")
        if pd.notna(t0):
            t_dummy = pd.Timestamp(t0) - pd.Timedelta(days=1)
            dummy_start = t_dummy.isoformat()
            dummy_end = t_dummy.isoformat()
    except Exception:
        pass

    cfg_eval = dict(cfg_train)
    bt2 = dict(cfg_eval.get("backtest") or {})
    bt2["splits"] = {
        "train": {"start": dummy_start, "end": dummy_end},
        "test": {
            "start": test_start,
            "end": test_end,
            "entry_end": test_end,
            "exit_end": test_end,
        },
    }
    cfg_eval["backtest"] = bt2
    return cfg_eval


def run_train_refit(
    *,
    cfg_eff: AppConfig | Mapping[str, Any],
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs_data: Mapping[str, Any],
    adv_map: Mapping[str, float] | None,
    borrow_ctx: Any | None,
    wf_i: int | None = None,
) -> TrainRefitArtifacts | None:
    cfg_train = _train_only_cfg_for_strategy(cfg_eff)
    if cfg_train is None:
        return None
    cfg_eval = _train_only_cfg_for_engine(cfg_train)
    if cfg_eval is None:
        return None

    strat = build_strategy_from_cfg(dict(cfg_train))
    portfolio = strat(pairs_data)

    from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg

    result = backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=as_price_map(prices),
        market_data_panel=prices_panel,
        adv_map=adv_map,
        yaml_cfg=cfg_eval,
        borrow_ctx=borrow_ctx,
    )
    stats = result.stats
    trades = result.trades
    eq = equity_from_stats(stats)
    summary_df = summarize_stats(eq, trades_df=trades)
    summary = summary_df.to_dict(orient="records")[0] if not summary_df.empty else {}

    bt = (
        cfg_train.get("backtest", {})
        if isinstance(cfg_train.get("backtest"), dict)
        else {}
    )
    splits = bt.get("splits", {}) if isinstance(bt.get("splits"), dict) else {}
    train = splits.get("train", {}) if isinstance(splits.get("train"), dict) else {}

    return TrainRefitArtifacts(
        train_start=str(train.get("start", "")),
        train_end=str(train.get("end", "")),
        n_pairs=int(len(pairs_data or {})),
        wf_i=wf_i,
        stats=stats,
        trades=trades,
        entry_intents=result.entry_intents,
        state_transitions=result.state_transitions,
        equity=eq,
        summary=summary,
    )


def write_train_refit_debug(out_dir: str | Path, art: TrainRefitArtifacts) -> None:
    debug_dir = Path(out_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    art.stats.to_csv(debug_dir / "train_stats.csv", index=False)
    art.trades.to_csv(debug_dir / "train_trades.csv", index=False)
    if isinstance(art.entry_intents, pd.DataFrame) and not art.entry_intents.empty:
        art.entry_intents.to_csv(debug_dir / "train_entry_intents.csv", index=False)
    if (
        isinstance(art.state_transitions, pd.DataFrame)
        and not art.state_transitions.empty
    ):
        art.state_transitions.to_csv(
            debug_dir / "train_state_transitions.csv", index=False
        )
    if isinstance(art.equity, pd.Series) and not art.equity.empty:
        pd.DataFrame(
            {"date": art.equity.index, "equity": art.equity.to_numpy(dtype=float)}
        ).to_csv(
            debug_dir / "train_equity.csv",
            index=False,
        )
    write_json(
        debug_dir / "train_summary.json",
        {
            "wf_i": art.wf_i,
            "train_start": art.train_start,
            "train_end": art.train_end,
            "n_pairs": art.n_pairs,
            "summary": art.summary,
        },
    )
