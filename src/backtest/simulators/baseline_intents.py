from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .contracts import ExecutionContext
from .execution_backends import make_execution_backend
from .intent_scheduler import portfolio_has_intents, simulate_intent_portfolio

__all__ = [
    "portfolio_has_intents",
    "simulate_baseline_intent_portfolio",
]


def simulate_baseline_intent_portfolio(
    *,
    portfolio: Mapping[str, Mapping[str, Any]],
    price_data: Mapping[str, pd.Series],
    cfg_obj: Any,
    market_data_panel: pd.DataFrame | None,
    adv_map: Mapping[str, float] | None,
    calendar: pd.DatetimeIndex,
    initial_capital: float,
) -> pd.DataFrame:
    ctx = ExecutionContext(
        cfg_obj=cfg_obj,
        price_data=price_data,
        market_data_panel=market_data_panel,
        adv_map=adv_map,
        calendar=calendar,
    )
    result = simulate_intent_portfolio(
        portfolio=portfolio,
        ctx=ctx,
        backend=make_execution_backend(cfg_obj),
        initial_capital=float(initial_capital),
    )
    out = result.trades.copy()
    out.attrs["entry_intents_df"] = result.entry_intents
    out.attrs["state_transitions_df"] = result.state_transitions
    for key, value in dict(result.debug_artifacts).items():
        out.attrs[key] = value
    return out
