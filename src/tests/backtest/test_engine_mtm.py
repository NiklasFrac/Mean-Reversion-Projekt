from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.simulators.engine_mtm import _map_trades_to_daily_pnl


def _cfg(
    *,
    strict_only: bool = False,
    calendar_mapping: str = "prior",
    initial_capital: float = 1_000.0,
    annualization_factor: int = 252,
) -> SimpleNamespace:
    return SimpleNamespace(
        calendar_mapping=calendar_mapping,
        raw_yaml={"backtest": {"calendar": {"strict_only": bool(strict_only)}}},
        initial_capital=float(initial_capital),
        annualization_factor=int(annualization_factor),
    )


def test_map_trades_to_daily_pnl_basic_pair() -> None:
    calendar = pd.date_range("2024-01-02", periods=3, freq="B", tz="America/New_York")
    trades = pd.DataFrame(
        {
            "entry_date": [calendar[0]],
            "exit_date": [calendar[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "y_units": [1.0],
            "x_units": [-1.0],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
        }
    )
    price_data = {
        "AAA": pd.Series([100.0, 105.0, 110.0], index=calendar),
        "BBB": pd.Series([50.0, 49.0, 48.0], index=calendar),
    }

    daily_pnl, daily_gross, mapped, dropped = _map_trades_to_daily_pnl(
        trades,
        calendar=calendar,
        cfg=_cfg(),
        price_data=price_data,
        borrow_ctx=None,
    )

    assert mapped == 1
    assert dropped == 0
    assert daily_pnl.index.equals(calendar)
    assert np.isclose(float(daily_pnl.sum()), 12.0)
    assert float(daily_gross.max()) > 0.0


def test_map_trades_to_daily_pnl_drops_missing_price_series() -> None:
    calendar = pd.date_range("2024-01-02", periods=2, freq="B", tz="America/New_York")
    trades = pd.DataFrame(
        {
            "entry_date": [calendar[0]],
            "exit_date": [calendar[1]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "y_units": [1.0],
            "x_units": [-1.0],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
        }
    )
    price_data = {
        "AAA": pd.Series([100.0, 101.0], index=calendar),
        # BBB intentionally missing
    }

    daily_pnl, _daily_gross, mapped, dropped = _map_trades_to_daily_pnl(
        trades,
        calendar=calendar,
        cfg=_cfg(),
        price_data=price_data,
        borrow_ctx=None,
    )

    assert mapped == 0
    assert dropped == 1
    assert np.isclose(float(daily_pnl.sum()), 0.0)


def test_map_trades_to_daily_pnl_applies_borrow_daily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = pd.date_range("2024-01-02", periods=2, freq="B", tz="America/New_York")
    trades = pd.DataFrame(
        {
            "entry_date": [calendar[0]],
            "exit_date": [calendar[1]],
            "y_symbol": ["AAA"],
            "y_units": [1.0],
            "entry_price_y": [100.0],
        }
    )
    price_data = {"AAA": pd.Series([100.0, 100.0], index=calendar)}

    def _fake_borrow_daily(*_args, **_kwargs) -> pd.Series:
        return pd.Series([-1.0, -2.0], index=calendar)

    monkeypatch.setattr(
        "backtest.simulators.engine_mtm.compute_borrow_daily_costs_for_trade_row",
        _fake_borrow_daily,
    )

    class BorrowCtx:
        enabled = True

    daily_pnl, _daily_gross, mapped, dropped = _map_trades_to_daily_pnl(
        trades,
        calendar=calendar,
        cfg=_cfg(),
        price_data=price_data,
        borrow_ctx=BorrowCtx(),
    )

    assert mapped == 1
    assert dropped == 0
    assert np.isclose(float(daily_pnl.sum()), -3.0)


def test_map_trades_to_daily_pnl_falls_back_to_borrow_total() -> None:
    calendar = pd.date_range("2024-01-02", periods=2, freq="B", tz="America/New_York")
    trades = pd.DataFrame(
        {
            "entry_date": [calendar[0]],
            "exit_date": [calendar[1]],
            "y_symbol": ["AAA"],
            "y_units": [1.0],
            "entry_price_y": [100.0],
            "borrow_cost": [-2.0],
        }
    )
    price_data = {"AAA": pd.Series([100.0, 100.0], index=calendar)}

    class BorrowCtx:
        enabled = False

    daily_pnl, _daily_gross, mapped, dropped = _map_trades_to_daily_pnl(
        trades,
        calendar=calendar,
        cfg=_cfg(),
        price_data=price_data,
        borrow_ctx=BorrowCtx(),
    )

    assert mapped == 1
    assert dropped == 0
    assert np.isclose(float(daily_pnl.sum()), -2.0)
