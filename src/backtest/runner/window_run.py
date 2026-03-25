from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from backtest.borrow.context import build_borrow_context
from backtest.config.cfg import AppConfig, config_to_dict
from backtest.config.validation import validate_runtime_config
from backtest.runner.loader import prepare_pairs_data
from backtest.runner.portfolio import (
    collect_portfolio_intents,
    collect_portfolio_trades,
)
from backtest.runner.runtime import pair_prefilter_cfg, pair_prefilter_inputs
from backtest.strat.registry import build_strategy_from_cfg
from backtest.utils.prices import as_price_map
from backtest.utils.portfolio import _df_trades_to_orders_df
from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg


@dataclass(frozen=True)
class WindowPortfolioArtifacts:
    cfg: AppConfig
    borrow_ctx: Any
    pairs_data: dict[str, dict[str, Any]]
    portfolio: dict[str, Any]
    raw_trades: pd.DataFrame
    orders: pd.DataFrame

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs_data))


@dataclass(frozen=True)
class WindowExecutionArtifacts(WindowPortfolioArtifacts):
    stats: pd.DataFrame
    trades: pd.DataFrame
    entry_intents: pd.DataFrame
    state_transitions: pd.DataFrame

    @property
    def info(self) -> dict[str, int]:
        return {
            "n_pairs": int(self.n_pairs),
            "n_trades": int(len(self.trades))
            if isinstance(self.trades, pd.DataFrame)
            else 0,
        }


def _bind_borrow_availability(
    borrow_ctx: Any | None,
    *,
    availability_long: Any | None,
) -> Any | None:
    if borrow_ctx is not None and availability_long is not None:
        try:
            setattr(borrow_ctx, "availability_long", availability_long)
        except Exception:
            pass
    return borrow_ctx


def resolve_window_borrow_context(
    cfg: AppConfig,
    *,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
) -> Any | None:
    borrow_ctx_local = (
        borrow_ctx if borrow_ctx is not None else build_borrow_context(cfg.borrow)
    )
    return _bind_borrow_availability(
        borrow_ctx_local, availability_long=availability_long
    )


def prepare_pairs_data_for_cfg(
    *,
    prices: pd.DataFrame,
    pairs: dict[str, Any],
    cfg: AppConfig,
    adv_map: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    disable_prefilter, prefilter_range = pair_prefilter_inputs(cfg)
    return prepare_pairs_data(
        prices,
        pairs,
        adv_map=adv_map,
        disable_prefilter=disable_prefilter,
        prefilter_range=prefilter_range,
        pair_prefilter_cfg=pair_prefilter_cfg(cfg),
    )


def _orders_from_portfolio(portfolio: Mapping[str, Any] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pair, meta in (portfolio or {}).items():
        if not isinstance(meta, Mapping):
            continue
        orders = meta.get("orders")
        if isinstance(orders, pd.DataFrame) and not orders.empty:
            frame = orders.copy()
            frame["pair"] = str(pair)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare_window_portfolio(
    *,
    cfg: AppConfig,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
    pairs_data: dict[str, dict[str, Any]] | None = None,
) -> WindowPortfolioArtifacts:
    validate_runtime_config(cfg)
    cfg_eff = cfg

    borrow_ctx_local = resolve_window_borrow_context(
        cfg_eff,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
    )
    pairs_data_local = pairs_data or prepare_pairs_data_for_cfg(
        prices=prices,
        pairs=pairs,
        cfg=cfg_eff,
        adv_map=adv_map,
    )

    strat = build_strategy_from_cfg(cfg_eff)
    portfolio = strat(pairs_data_local)
    raw_trades = collect_portfolio_trades(portfolio)
    if raw_trades.empty:
        raw_trades = collect_portfolio_intents(portfolio)
    orders = _orders_from_portfolio(portfolio)

    return WindowPortfolioArtifacts(
        cfg=cfg_eff,
        borrow_ctx=borrow_ctx_local,
        pairs_data=pairs_data_local,
        portfolio=portfolio,
        raw_trades=raw_trades,
        orders=orders,
    )


def execute_window_backtest(
    *,
    cfg: AppConfig,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
    pairs_data: dict[str, dict[str, Any]] | None = None,
) -> WindowExecutionArtifacts:
    prepared = prepare_window_portfolio(
        cfg=cfg,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
        pairs_data=pairs_data,
    )

    result = backtest_portfolio_with_yaml_cfg(
        portfolio=prepared.portfolio,
        price_data=as_price_map(prices),
        market_data_panel=prices_panel,
        adv_map=adv_map,
        yaml_cfg=config_to_dict(prepared.cfg),
        borrow_ctx=prepared.borrow_ctx,
    )
    stats = result.stats
    trades = result.trades
    entry_intents = result.entry_intents
    state_transitions = result.state_transitions
    try:
        stats.attrs["n_pairs"] = int(prepared.n_pairs)
        stats.attrs["n_trades"] = (
            int(len(trades)) if isinstance(trades, pd.DataFrame) else 0
        )
    except Exception:
        pass

    orders = prepared.orders
    if (
        (not isinstance(orders, pd.DataFrame) or orders.empty)
        and isinstance(trades, pd.DataFrame)
        and not trades.empty
    ):
        orders = _df_trades_to_orders_df(trades)

    return WindowExecutionArtifacts(
        cfg=prepared.cfg,
        borrow_ctx=prepared.borrow_ctx,
        pairs_data=prepared.pairs_data,
        portfolio=prepared.portfolio,
        raw_trades=prepared.raw_trades,
        orders=orders,
        stats=stats,
        trades=trades,
        entry_intents=entry_intents,
        state_transitions=state_transitions,
    )
