from __future__ import annotations

import pandas as pd

from universe.datetime_utils import coerce_date_like_value, coerce_utc_naive_timestamp


def test_coerce_utc_naive_timestamp_handles_tz_and_normalize():
    ts = pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")
    out = coerce_utc_naive_timestamp(ts, normalize=True)
    assert out == pd.Timestamp("2024-01-02")


def test_coerce_utc_naive_timestamp_handles_datetimeindex_input():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 15:00:00", tz="UTC")])
    out = coerce_utc_naive_timestamp(idx, normalize=False)
    assert out == pd.Timestamp("2024-01-02 15:00:00")


def test_coerce_date_like_value_preserves_strings_and_unknown_values():
    marker = object()
    assert coerce_date_like_value("2024-01-02") == "2024-01-02"
    assert coerce_date_like_value(marker) is marker
    assert coerce_date_like_value(pd.Timestamp("2024-01-02", tz="UTC")) == "2024-01-02"
