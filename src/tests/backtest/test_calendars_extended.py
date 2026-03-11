from __future__ import annotations

import pandas as pd

from backtest import calendars


def test_map_to_calendar_nearest_and_unknown_policy() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    ts = pd.Timestamp("2024-01-04 12:00")
    nearest = calendars.map_to_calendar(ts, idx, policy="nearest")
    assert nearest == pd.Timestamp("2024-01-04")

    unknown = calendars.map_to_calendar(ts, idx, policy="weird")
    assert unknown == pd.Timestamp("2024-01-04")


def test_build_trading_calendar_daily_fallback() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    price_data = {"AAA": pd.Series(range(5), index=idx)}
    cal = calendars.build_trading_calendar(price_data, calendar_name="INVALID")
    assert len(cal) == len(idx)


def test_apply_settlement_lag_maps_missing() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    ts = pd.Timestamp("2024-01-01")
    out = calendars.apply_settlement_lag(ts, idx, lag_bars=1)
    assert out == pd.Timestamp("2024-01-03")


def test_map_to_calendar_prior_next_and_empty() -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    ts = pd.Timestamp("2024-01-02 12:00")
    prior = calendars.map_to_calendar(ts, idx, policy="prior")
    nxt = calendars.map_to_calendar(ts, idx, policy="next")
    assert prior == pd.Timestamp("2024-01-02", tz="UTC")
    assert nxt == pd.Timestamp("2024-01-03", tz="UTC")

    assert calendars.map_to_calendar(ts, pd.DatetimeIndex([]), policy="prior") is None


def test_build_trading_calendar_empty_and_apply_lag_noop() -> None:
    empty = calendars.build_trading_calendar({})
    assert empty.empty

    idx = pd.bdate_range("2024-01-02", periods=2)
    ts = pd.Timestamp("2024-01-02")
    out = calendars.apply_settlement_lag(ts, idx, lag_bars=0)
    assert out == ts


def test_is_intraday_empty_and_error() -> None:
    assert calendars._is_intraday(pd.DatetimeIndex([])) is False
    assert calendars._is_intraday(pd.Index([1, 2])) is False


def test_build_trading_calendar_tz_fallback() -> None:
    idx = pd.date_range("2024-01-02", periods=3, tz="UTC")
    price_data = {"AAA": pd.Series([1.0, 2.0, 3.0], index=idx)}
    cal = calendars.build_trading_calendar(price_data, calendar_name="INVALID")
    assert str(cal.tz) == "UTC"


def test_map_to_calendar_nearest_between_and_nat() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    ts = pd.Timestamp("2024-01-02 12:00")
    nearest = calendars.map_to_calendar(ts, idx, policy="nearest")
    assert nearest == pd.Timestamp("2024-01-02")

    assert calendars.map_to_calendar(pd.NaT, idx, policy="prior") is None


def test_apply_settlement_lag_empty_and_nat() -> None:
    ts = pd.Timestamp("2024-01-02")
    assert calendars.apply_settlement_lag(ts, pd.DatetimeIndex([]), lag_bars=1) == ts

    idx = pd.bdate_range("2024-01-02", periods=2)
    out = calendars.apply_settlement_lag(pd.NaT, idx, lag_bars=1)
    assert pd.isna(out)
