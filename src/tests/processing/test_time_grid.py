from __future__ import annotations

import sys
import types

import pandas as pd

from processing.timebase import build_tradable_mask
from processing.timebase import pick_time_grid


def _make_raw():
    idx_a = pd.date_range("2020-01-01", periods=5, tz="UTC")
    idx_b = pd.date_range("2020-01-02", periods=4, tz="UTC")
    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3, 4, 5], index=idx_a, dtype="float64"),
            "B": pd.Series([1, 2, 3, 4], index=idx_b, dtype="float64"),
        }
    )
    return df


def test_pick_time_grid_leader():
    raw = _make_raw()
    grid = pick_time_grid(raw, mode="leader")
    # A has 5 non-NaN -> leader is A
    assert len(grid) == 5 and isinstance(grid, pd.DatetimeIndex)


def test_pick_time_grid_intersection_and_union():
    raw = _make_raw()
    inter = pick_time_grid(raw, mode="intersection")
    union = pick_time_grid(raw, mode="union")
    assert len(inter) < len(union)
    assert set(inter).issubset(set(union))


def test_pick_time_grid_calendar_fallback(monkeypatch):
    # Stub for pandas_market_calendars that raises to trigger fallback.
    fake = types.ModuleType("pandas_market_calendars")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    fake.get_calendar = lambda name: types.SimpleNamespace(schedule=_boom)
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    raw = _make_raw()
    grid = pick_time_grid(raw, mode="calendar", calendar_code="XNYS")
    assert isinstance(grid, pd.DatetimeIndex)
    exp = pd.DatetimeIndex(raw.index.normalize().unique()).sort_values()
    assert grid.equals(exp)


def test_build_tradable_mask_marks_missing_sessions_false(monkeypatch):
    import types as _types

    import pandas as _pd

    fake = _types.ModuleType("pandas_market_calendars")

    def _get_calendar(name):
        class _Cal:
            def schedule(self, start_date, end_date):
                idx = _pd.DatetimeIndex(["2020-01-03"])
                return _pd.DataFrame(
                    {
                        "market_open": _pd.to_datetime(idx)
                        .tz_localize("America/New_York")
                        .normalize()
                        + _pd.Timedelta(hours=9, minutes=30),
                        "market_close": _pd.to_datetime(idx)
                        .tz_localize("America/New_York")
                        .normalize()
                        + _pd.Timedelta(hours=16),
                    },
                    index=idx,
                )

        return _Cal()

    fake.get_calendar = _get_calendar
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-03 12:00", tz="America/New_York"),  # Friday
            pd.Timestamp("2020-01-04 12:00", tz="America/New_York"),  # Saturday
        ]
    )

    mask = build_tradable_mask(idx, calendar_code="XNYS", rth_only=True)

    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False
