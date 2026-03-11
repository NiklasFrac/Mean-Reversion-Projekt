import pandas as pd

from backtest.borrow import accrual


def test_borrow_accrual_helper_counts() -> None:
    assert accrual._to_ts("not-a-date") is None

    a = pd.Timestamp("2024-01-01", tz="UTC")
    b = pd.Timestamp("2024-01-03", tz="UTC")
    assert accrual._busdays_between(a, b) >= 1
    assert accrual._calendar_days_between(a, b, include_end=True) == 3
    assert accrual._calendar_days_between(a, b, include_end=False) == 2

    row = pd.Series({"a": "x", "b": 2.0})
    assert accrual._get_first_float(row, ("a", "b")) == 2.0


def test_infer_short_leg_variants() -> None:
    row = pd.Series(
        {
            "signal": 1,
            "y_symbol": "AAA",
            "x_symbol": "BBB",
            "notional_y": 100.0,
            "notional_x": -200.0,
        }
    )
    assert accrual._infer_short_leg(row) == ("BBB", 200.0)

    row2 = pd.Series(
        {"signal": -1, "y_symbol": "AAA", "x_symbol": "BBB", "gross_notional": 300.0}
    )
    assert accrual._infer_short_leg(row2) == ("AAA", 150.0)

    row3 = pd.Series({"symbol": "ZZZ", "notional": 50.0})
    assert accrual._infer_short_leg(row3) == (None, 25.0)

    row4 = pd.Series({"symbol": "YYY", "size": 2, "price": 10.0})
    assert accrual._infer_short_leg(row4) == ("YYY", 20.0)


def test_resolve_rate_and_cfg_helpers() -> None:
    class DummyBorrow:
        default_rate_annual = 0.1

        def resolve_borrow_rate(self, _symbol, _day):
            raise RuntimeError("boom")

    assert (
        accrual._resolve_rate(DummyBorrow(), "AAA", pd.Timestamp("2024-01-01")) == 0.0
    )

    class DummyCfg:
        accrual_mode = "mtm_daily"
        day_count = "sessions"
        include_exit_day = True
        min_days = -1
        day_basis = 360

    mode, day_count, include_exit, min_days = accrual._get_borrow_cfg(DummyCfg())
    assert mode == "mtm_daily"
    assert day_count == "sessions"
    assert include_exit is True
    assert min_days == 0


def test_asof_price_and_short_symbol_units() -> None:
    px = pd.Series([10.0, 11.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert accrual._asof_price(px, pd.Timestamp("2024-01-01")) is None

    row = pd.Series({"y_symbol": "AAA", "x_symbol": "BBB", "y_units": -5, "x_units": 5})
    assert accrual._infer_short_symbol_and_units(row) == ("AAA", 5)

    row2 = pd.Series({"symbol": "ZZZ"})
    assert accrual._infer_short_symbol_and_units(row2) == ("ZZZ", None)


def test_build_day_schedule_variants() -> None:
    entry = pd.Timestamp("2024-01-02")
    exit_ = pd.Timestamp("2024-01-05")
    cal = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))

    sched = accrual._build_day_schedule(
        entry_day=entry,
        exit_day=exit_,
        calendar=cal,
        day_count="calendar_days",
        include_exit_day=True,
        min_days=1,
    )
    assert len(sched) >= 1

    sched_sessions = accrual._build_day_schedule(
        entry_day=entry,
        exit_day=exit_,
        calendar=cal,
        day_count="sessions",
        include_exit_day=False,
        min_days=1,
    )
    assert len(sched_sessions) >= 1

    sched_bus = accrual._build_day_schedule(
        entry_day=entry,
        exit_day=exit_,
        calendar=cal,
        day_count="busdays",
        include_exit_day=False,
        min_days=1,
    )
    assert len(sched_bus) >= 1

    sched_fallback = accrual._build_day_schedule(
        entry_day=entry,
        exit_day=entry,
        calendar=pd.DatetimeIndex([]),
        day_count="sessions",
        include_exit_day=False,
        min_days=1,
    )
    assert len(sched_fallback) == 1


def test_accrual_cfg_error_paths() -> None:
    class BadCfg:
        @property
        def accrual_mode(self):
            raise ValueError("boom")

        @property
        def day_count(self):
            raise ValueError("boom")

        @property
        def include_exit_day(self):
            raise ValueError("boom")

        @property
        def min_days(self):
            raise ValueError("boom")

        @property
        def day_basis(self):
            raise ValueError("boom")

    mode, day_count, include_exit, min_days = accrual._get_borrow_cfg(BadCfg())
    assert mode == "entry_notional"
    assert day_count == "busdays"
    assert include_exit is False
    assert min_days == 1
    assert accrual._day_basis(BadCfg()) == 252


def test_borrow_cost_mtm_daily_empty_schedule() -> None:
    class Ctx:
        enabled = True
        accrual_mode = "mtm_daily"
        day_count = "sessions"
        include_exit_day = False
        min_days = 0
        day_basis = 0

        def resolve_borrow_rate(self, _sym, _day):
            return 0.1

    row = pd.Series(
        {
            "entry_date": pd.Timestamp("2024-01-02"),
            "exit_date": pd.Timestamp("2024-01-03"),
            "y_symbol": "AAA",
            "x_symbol": "BBB",
            "notional_y": 100.0,
            "notional_x": -100.0,
        }
    )
    cost = accrual.compute_borrow_cost_for_trade_row(
        row,
        calendar=pd.DatetimeIndex([]),
        price_data={},
        borrow_ctx=Ctx(),
    )
    assert cost == 0.0


def test_borrow_meta_missing_dates_and_trade_df_empty() -> None:
    class Ctx:
        enabled = True

    row = pd.Series({"entry_date": pd.NaT, "exit_date": pd.NaT})
    meta = accrual.compute_borrow_meta_for_trade_row(
        row,
        calendar=pd.DatetimeIndex([]),
        price_data={},
        borrow_ctx=Ctx(),
    )
    assert meta["short_symbol"] is None

    out = accrual.compute_borrow_cost_for_trades_df(
        pd.DataFrame(),
        calendar=pd.DatetimeIndex([]),
        price_data={},
        borrow_ctx=None,
    )
    assert out.empty or out.sum() == 0.0
