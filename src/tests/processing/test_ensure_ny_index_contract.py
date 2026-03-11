from __future__ import annotations

import pandas as pd

from processing.timebase import ensure_ny_index


def test_ensure_ny_index_preserves_daily_date_labels_for_naive_utc() -> None:
    idx = pd.DatetimeIndex(
        [
            "2025-01-02 00:00:00",
            "2025-01-03 00:00:00",
            "2025-01-06 00:00:00",
        ]
    )
    df = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)

    out = ensure_ny_index(df, vendor_tz="UTC")

    assert str(out.index.tz) == "America/New_York"
    assert [ts.hour for ts in out.index] == [0, 0, 0]
    assert list(out.index.tz_localize(None).normalize()) == list(idx.normalize())


def test_ensure_ny_index_converts_intraday_naive_from_vendor_tz() -> None:
    idx = pd.DatetimeIndex(
        [
            "2025-01-02 14:30:00",  # UTC
            "2025-01-02 15:30:00",  # UTC
        ]
    )
    df = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)

    out = ensure_ny_index(df, vendor_tz="UTC")

    assert str(out.index.tz) == "America/New_York"
    assert [ts.hour for ts in out.index] == [9, 10]
    assert [ts.minute for ts in out.index] == [30, 30]
