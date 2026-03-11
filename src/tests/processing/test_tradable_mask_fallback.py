from __future__ import annotations

import sys
import types

import pandas as pd

from processing.timebase import build_tradable_mask


def test_build_tradable_mask_business_day_fallback_handles_non_midnight_daily(
    monkeypatch,
):
    # Force calendar path into fallback branch.
    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    idx = (
        pd.bdate_range("2020-01-01", periods=5, tz="America/New_York")
        .normalize()
        .shift(19, freq="h")
    )
    mask = build_tradable_mask(idx, calendar_code="XNYS", rth_only=True)

    assert mask.index.equals(idx)
    assert bool(mask.all())


def test_build_tradable_mask_business_day_fallback_handles_eod_daily_time(monkeypatch):
    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    idx = (
        pd.bdate_range("2020-01-01", periods=5, tz="America/New_York")
        .normalize()
        .shift(16, freq="h")
    )
    mask = build_tradable_mask(idx, calendar_code="XNYS", rth_only=True)

    assert mask.index.equals(idx)
    assert bool(mask.all())
