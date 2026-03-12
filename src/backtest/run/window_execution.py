from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from backtest.borrow.context import build_borrow_context
from backtest.utils.run.data import (
    _as_price_map,
    _pair_prefilter_inputs,
    _prepare_pairs_data,
)
from backtest.utils.portfolio import _df_trades_to_orders_df
from backtest.utils.run.strategy import _build_strategy, _validate_cfg_strict
from backtest.utils.run.trades import _collect_portfolio_intents, _collect_portfolio_trades
from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg


@dataclass(frozen=True)
class WindowPortfolioArtifacts:
    cfg: dict[str, Any]
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
    cfg: Mapping[str, Any],
    *,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
) -> Any | None:
    borrow_ctx_local = (
        borrow_ctx if borrow_ctx is not None else build_borrow_context(dict(cfg))
    )
    return _bind_borrow_availability(
        borrow_ctx_local, availability_long=availability_long
    )


def prepare_pairs_data_for_cfg(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    cfg: Mapping[str, Any],
    adv_map: dict[str, float] | None,
) -> dict[str, dict[str, Any]]:
    disable_prefilter, prefilter_range = _pair_prefilter_inputs(cfg)
    return _prepare_pairs_data(
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        cfg=dict(cfg),
        adv_map=adv_map,
        disable_prefilter=disable_prefilter,
        prefilter_range=prefilter_range,
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
    cfg: dict[str, Any],
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
    pairs_data: dict[str, dict[str, Any]] | None = None,
) -> WindowPortfolioArtifacts:
    _validate_cfg_strict(cfg)
    cfg_eff = dict(cfg)

    borrow_ctx_local = resolve_window_borrow_context(
        cfg_eff,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
    )
    pairs_data_local = pairs_data or prepare_pairs_data_for_cfg(
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        cfg=cfg_eff,
        adv_map=adv_map,
    )

    strat = _build_strategy(cfg_eff, borrow_ctx=borrow_ctx_local)
    portfolio = strat(pairs_data_local)
    raw_trades = _collect_portfolio_trades(portfolio)
    if raw_trades.empty:
        raw_trades = _collect_portfolio_intents(portfolio)
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
    cfg: dict[str, Any],
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

    stats, trades = backtest_portfolio_with_yaml_cfg(
        portfolio=prepared.portfolio,
        price_data=_as_price_map(prices),
        market_data_panel=prices_panel,
        adv_map=adv_map,
        yaml_cfg=prepared.cfg,
        borrow_ctx=prepared.borrow_ctx,
    )
    try:
        stats.attrs["n_pairs"] = int(prepared.n_pairs)
        stats.attrs["n_trades"] = (
            int(len(trades)) if isinstance(trades, pd.DataFrame) else 0
        )
    except Exception:
        pass

    orders = prepared.orders
    if (not isinstance(orders, pd.DataFrame) or orders.empty) and isinstance(
        trades, pd.DataFrame
    ) and not trades.empty:
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
    )
