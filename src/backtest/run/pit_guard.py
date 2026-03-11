from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backtest.run.window_execution import execute_window_backtest

logger = logging.getLogger("backtest.runner")


def _pit_guard_window_runner(
    start_idx: int,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    cfg: dict[str, Any],
    adv_map: dict[str, float] | None,
    borrow_ctx: Any | None,
    availability_long: Any | None,
    availability_scope: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Minimal window runner for PIT-Guard: no side-effects, no reports/BO.
    """
    window_run = execute_window_backtest(
        cfg=dict(cfg),
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
    )
    info = window_run.info
    artifacts = {"trades_te": window_run.trades}
    return window_run.stats, info, window_run.stats, info, artifacts
