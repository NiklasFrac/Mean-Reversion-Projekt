from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtest.config.cfg import AppConfig, config_to_dict
from backtest.optimize.runner import BORunResult, run_bo_if_enabled
from backtest.reporting.common import equity_from_stats
from backtest.reporting.tearsheet import summarize_stats
from backtest.runner.train_refit import (
    TrainRefitArtifacts,
    run_train_refit,
    write_train_refit_debug,
)
from backtest.runner.window_run import (
    execute_window_backtest,
    prepare_pairs_data_for_cfg,
)
from backtest.utils.io import write_json

logger = logging.getLogger("backtest.runner.single")


@dataclass(frozen=True)
class SingleRunArtifacts:
    cfg_eff: AppConfig
    bo_run: BORunResult
    pairs_data: dict[str, Any]
    borrow_ctx: Any
    stats: pd.DataFrame
    trades: pd.DataFrame
    raw_trades: pd.DataFrame
    orders: pd.DataFrame
    test_equity: pd.Series
    test_summary: dict[str, Any]
    train_refit: TrainRefitArtifacts | None
    portfolio: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs_data))

    @property
    def n_trades(self) -> int:
        return int(len(self.trades)) if isinstance(self.trades, pd.DataFrame) else 0


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_debug_window(
    debug_out: Path,
    *,
    art: SingleRunArtifacts,
    splits: Mapping[str, Any] | None = None,
) -> None:
    debug_out.mkdir(parents=True, exist_ok=True)
    write_json(debug_out / "config_effective.json", art.cfg_eff)
    art.stats.to_csv(debug_out / "stats.csv", index=False)
    art.trades.to_csv(debug_out / "trades.csv", index=False)
    entry_intents = art.stats.attrs.get("entry_intents_df")
    if not isinstance(entry_intents, pd.DataFrame) or entry_intents.empty:
        if (
            isinstance(art.raw_trades, pd.DataFrame)
            and not art.raw_trades.empty
            and "signal_date" in art.raw_trades.columns
        ):
            entry_intents = art.raw_trades
    if isinstance(entry_intents, pd.DataFrame) and not entry_intents.empty:
        entry_intents.to_csv(debug_out / "entry_intents.csv", index=False)
    state_transitions = art.stats.attrs.get("state_transitions_df")
    if isinstance(state_transitions, pd.DataFrame) and not state_transitions.empty:
        state_transitions.to_csv(debug_out / "state_transitions.csv", index=False)
    if isinstance(art.orders, pd.DataFrame) and not art.orders.empty:
        art.orders.to_csv(debug_out / "orders.csv", index=False)
    if splits is not None:
        write_json(debug_out / "walkforward_window.json", {"splits": dict(splits)})
    if art.bo_run.bo_res is not None:
        bo_debug = debug_out / "bo"
        bo_debug.mkdir(parents=True, exist_ok=True)
        write_json(bo_debug / "bo_best.json", art.bo_run.bo_res)
        if (
            art.bo_run.selected_cv_scores is not None
            and not art.bo_run.selected_cv_scores.empty
        ):
            art.bo_run.selected_cv_scores.to_csv(
                bo_debug / "selected_cv_scores.csv", index=False
            )
        if art.bo_run.bo_out is not None:
            _copy_if_exists(
                art.bo_run.bo_out / "bo_trials.csv", bo_debug / "bo_trials.csv"
            )
            _copy_if_exists(
                art.bo_run.bo_out / "bo_best.json", bo_debug / "bo_best_source.json"
            )
    if art.train_refit is not None:
        write_train_refit_debug(debug_out / "train_refit", art.train_refit)
    try:
        from backtest.reporting.pnl_breakdown import generate_pnl_breakdown

        generate_pnl_breakdown({}, debug_out / "performance")
    except Exception:
        logger.warning("Debug pnl breakdown failed.", exc_info=True)


def run_single_backtest(
    *,
    cfg_eff: AppConfig,
    out_dir: Path,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    quick: bool = False,
    return_raw_trades: bool = False,
    debug_out: Path | None = None,
) -> SingleRunArtifacts:
    pairs_data = prepare_pairs_data_for_cfg(
        prices=prices,
        pairs=pairs,
        cfg=cfg_eff,
        adv_map=adv_map,
    )
    bo_run = run_bo_if_enabled(
        cfg_eff=cfg_eff,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        out_dir=out_dir,
        quick=quick,
        pairs_data=pairs_data,
        persist_quick_budget=True,
    )
    cfg_eff = bo_run.cfg_eff
    pairs_data = bo_run.pairs_data or pairs_data

    window_run = execute_window_backtest(
        cfg=cfg_eff,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        pairs_data=pairs_data,
    )
    stats = window_run.stats
    trades = window_run.trades
    raw_trades = (
        window_run.raw_trades
        if return_raw_trades or debug_out is not None
        else pd.DataFrame()
    )
    eq = equity_from_stats(stats)
    test_summary_df = summarize_stats(eq, trades_df=trades)
    test_summary = (
        test_summary_df.to_dict(orient="records")[0]
        if not test_summary_df.empty
        else {}
    )

    train_refit = None
    try:
        train_refit = run_train_refit(
            cfg_eff=cfg_eff,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=window_run.pairs_data,
            adv_map=adv_map,
            borrow_ctx=window_run.borrow_ctx,
        )
    except Exception:
        logger.warning("Train refit failed.", exc_info=True)

    art = SingleRunArtifacts(
        cfg_eff=cfg_eff,
        bo_run=bo_run,
        pairs_data=window_run.pairs_data,
        borrow_ctx=window_run.borrow_ctx,
        stats=stats,
        trades=trades,
        raw_trades=raw_trades,
        orders=window_run.orders,
        test_equity=eq,
        test_summary=test_summary,
        train_refit=train_refit,
        portfolio=getattr(window_run, "portfolio", {}) or {},
    )

    if debug_out is not None:
        write_debug_window(debug_out, art=art)
    return art
