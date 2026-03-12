from __future__ import annotations

import sys
import types

import pandas as pd

from processing.timebase import pick_time_grid


def test_pick_time_grid_calendar_normal_path(monkeypatch):
    """
    Covers the normal path via pandas_market_calendars.schedule.
    """

    class FakeCal:
        def schedule(self, start_date, end_date):
            # Return business days in the range
            idx = pd.bdate_range(start=start_date, end=end_date)
            return pd.DataFrame(index=idx)

    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda name: FakeCal()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    # Raw data with min/max
    idx = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    raw = pd.DataFrame({"A": range(6)}, index=idx, dtype="float64")

    grid = pick_time_grid(raw, mode="calendar", calendar_code="XNYS")
    assert str(getattr(grid, "tz", None)) == "America/New_York"
    # Business days between 2020-01-01 and 2020-01-06 are 4 (weekend removed)
    assert len(grid) >= 4
