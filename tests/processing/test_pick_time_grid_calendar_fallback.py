from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from processing.timebase import pick_time_grid


def test_pick_time_grid_calendar_fallback_without_mcal(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": range(5)}, index=idx)

    # Two paths:
    # 1) If pandas_market_calendars is installed, force get_calendar to raise.
    # 2) If not installed, import itself raises and fallback path is used.
    try:
        import pandas_market_calendars as mcal  # noqa: F401

        stub = types.SimpleNamespace(
            get_calendar=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setitem(sys.modules, "pandas_market_calendars", stub)
    except Exception:
        pass

    out = pick_time_grid(df, mode="calendar", calendar_code="XNYS")
    # Fallback keeps observed sessions from raw input (no synthetic holidays).
    exp = pd.DatetimeIndex(df.index.normalize().unique()).sort_values()
    assert out.equals(exp)


def test_pick_time_grid_calendar_rejects_intraday():
    idx = pd.date_range("2020-01-01 09:30", periods=3, freq="h")
    df = pd.DataFrame({"A": range(3)}, index=idx)

    with pytest.raises(ValueError):
        pick_time_grid(df, mode="calendar", calendar_code="XNYS")


def test_pick_time_grid_calendar_accepts_shifted_daily_like_index():
    idx = (
        pd.bdate_range("2020-01-01", periods=5, tz="America/New_York")
        .normalize()
        .shift(16, freq="h")
    )
    df = pd.DataFrame({"A": range(5)}, index=idx)
    out = pick_time_grid(df, mode="calendar", calendar_code="XNYS")
    assert len(out) > 0
    assert set(pd.DatetimeIndex(out).hour) == {16}
