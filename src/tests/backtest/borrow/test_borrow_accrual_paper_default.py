import pandas as pd

from backtest.borrow.accrual import (
    compute_borrow_cost_for_trade_row,
    compute_borrow_daily_costs_for_trade_row,
    compute_borrow_meta_for_trade_row,
)
from backtest.borrow.context import BorrowContext


def _px(dates: list[str], vals: list[float]) -> pd.Series:
    s = pd.Series(vals, index=pd.to_datetime(pd.Series(dates), errors="coerce"))
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def test_borrow_mtm_daily_calendar_days_includes_exit_day() -> None:
    borrow_ctx = BorrowContext(
        enabled=True,
        default_rate_annual=0.10,
        day_basis=360,
        accrual_mode="mtm_daily",
        day_count="calendar_days",
        include_exit_day=True,
        min_days=1,
    )
    cal = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    price_data = {
        "X": _px(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 110.0, 90.0])
    }

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2020-01-01"),
            "exit_date": pd.Timestamp("2020-01-03"),
            "y_symbol": "Y",
            "x_symbol": "X",
            "y_units": 10,
            "x_units": -10,  # short X
            "signal": 1,
            "notional_x": -1000.0,
        }
    )

    cost = compute_borrow_cost_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    expected = -(0.10 / 360.0) * (10.0 * 100.0 + 10.0 * 110.0 + 10.0 * 90.0)
    assert abs(cost - expected) < 1e-9

    meta = compute_borrow_meta_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    assert meta["short_symbol"] == "X"
    assert meta["n_days"] == 3
    assert meta["mtm_price_used"] is True


def test_borrow_mtm_calendar_days_weekend_uses_asof_last_close() -> None:
    borrow_ctx = BorrowContext(
        enabled=True,
        default_rate_annual=0.12,
        day_basis=360,
        accrual_mode="mtm_daily",
        day_count="calendar_days",
        include_exit_day=True,
        min_days=1,
    )
    cal = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    # Only Fri + Mon closes exist; Sat/Sun should use Fri close via asof.
    price_data = {"X": _px(["2020-01-03", "2020-01-06"], [100.0, 110.0])}

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2020-01-03"),  # Fri
            "exit_date": pd.Timestamp("2020-01-06"),  # Mon
            "y_symbol": "Y",
            "x_symbol": "X",
            "y_units": 5,
            "x_units": -5,  # short X
        }
    )

    cost = compute_borrow_cost_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    # Fri+Sat+Sun use 100; Mon uses 110
    expected = -(0.12 / 360.0) * (5.0 * (100.0 + 100.0 + 100.0 + 110.0))
    assert abs(cost - expected) < 1e-9


def test_borrow_mtm_sessions_excludes_exit_when_configured() -> None:
    borrow_ctx = BorrowContext(
        enabled=True,
        default_rate_annual=0.20,
        day_basis=360,
        accrual_mode="mtm_daily",
        day_count="sessions",
        include_exit_day=False,
        min_days=1,
    )
    # Session calendar (trading days)
    cal = pd.DatetimeIndex(pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]))
    price_data = {
        "X": _px(["2020-01-06", "2020-01-07", "2020-01-08"], [10.0, 10.0, 10.0])
    }

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2020-01-06"),
            "exit_date": pd.Timestamp("2020-01-08"),
            "x_symbol": "X",
            "x_units": -10,  # short X
        }
    )

    # include_exit_day=False -> sessions: 2020-01-06, 2020-01-07 only
    cost = compute_borrow_cost_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    expected = -(0.20 / 360.0) * (10.0 * 10.0 + 10.0 * 10.0)
    assert abs(cost - expected) < 1e-9


def test_borrow_mtm_falls_back_to_entry_notional_if_missing_prices() -> None:
    borrow_ctx = BorrowContext(
        enabled=True,
        default_rate_annual=0.10,
        day_basis=360,
        accrual_mode="mtm_daily",
        day_count="calendar_days",
        include_exit_day=True,
        min_days=1,
    )
    cal = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    price_data: dict[str, pd.Series] = {}  # missing symbol prices

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2020-01-01"),
            "exit_date": pd.Timestamp("2020-01-03"),
            "y_symbol": "Y",
            "x_symbol": "X",
            "signal": 1,
            # entry notional available (used as fallback)
            "notional_y": 1000.0,
            "notional_x": -1200.0,
        }
    )

    cost = compute_borrow_cost_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    expected = -(0.10 / 360.0) * (3.0 * 1200.0)  # 3 calendar days incl exit
    assert abs(cost - expected) < 1e-9


def test_borrow_daily_series_matches_total() -> None:
    borrow_ctx = BorrowContext(
        enabled=True,
        default_rate_annual=0.08,
        day_basis=360,
        accrual_mode="mtm_daily",
        day_count="calendar_days",
        include_exit_day=True,
        min_days=1,
    )
    cal = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    price_data = {
        "X": _px(["2020-01-01", "2020-01-02", "2020-01-03"], [100.0, 102.0, 101.0])
    }

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2020-01-01"),
            "exit_date": pd.Timestamp("2020-01-03"),
            "y_symbol": "Y",
            "x_symbol": "X",
            "x_units": -5,  # short X
        }
    )

    total = compute_borrow_cost_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    daily = compute_borrow_daily_costs_for_trade_row(
        row, calendar=cal, price_data=price_data, borrow_ctx=borrow_ctx
    )
    assert abs(float(daily.sum()) - float(total)) < 1e-9
    assert len(daily) == 3
