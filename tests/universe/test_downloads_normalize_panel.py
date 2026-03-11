from __future__ import annotations

import pandas as pd

from universe.downloads import derive_date_range, normalize_panel_for_universe
from universe.panel_utils import (
    coerce_utc_naive_index,
    collapse_duplicate_index_rows,
    merge_duplicate_columns_prefer_non_null,
)


def test_normalize_panel_for_universe_normalizes_daily_to_midnight():
    df = pd.DataFrame(
        {"AAA_close": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-02 05:00:00", "2024-06-03 04:00:00"]),
    )

    out = normalize_panel_for_universe(df, "1d")

    assert out.index.tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-06-03"),
    ]


def test_normalize_panel_for_universe_preserves_intraday_times():
    df = pd.DataFrame(
        {"AAA_close": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-02 09:30:00", "2024-01-02 10:30:00"]),
    )

    out = normalize_panel_for_universe(df, "1h")

    assert out.index.tolist() == df.index.tolist()


def test_normalize_panel_for_universe_merges_duplicate_days_without_dropping_data():
    df = pd.DataFrame(
        {
            "AAA_close": [1.0, None],
            "BBB_close": [None, 2.0],
        },
        index=pd.DatetimeIndex(["2024-01-02 05:00:00", "2024-01-02 00:00:00"]),
    )

    out = normalize_panel_for_universe(df, "1d")

    assert out.index.tolist() == [pd.Timestamp("2024-01-02")]
    assert float(out.loc[pd.Timestamp("2024-01-02"), "AAA_close"]) == 1.0
    assert float(out.loc[pd.Timestamp("2024-01-02"), "BBB_close"]) == 2.0


def test_normalize_panel_for_universe_merges_duplicate_columns_preferring_non_null():
    idx = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        [[None, 10.0], [None, 11.0]],
        index=idx,
        columns=["AAA_close", "AAA_close"],
    )

    out = normalize_panel_for_universe(df, "1d")

    assert list(out.columns) == ["AAA_close"]
    assert out["AAA_close"].notna().all()
    assert out["AAA_close"].iloc[0] == 10.0
    assert out["AAA_close"].iloc[1] == 11.0


def test_merge_duplicate_columns_prefer_non_null_shared_helper():
    idx = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        [[None, 1.0], [2.0, None], [None, 3.0]],
        index=idx,
        columns=["AAA_close", "AAA_close"],
    )

    out = merge_duplicate_columns_prefer_non_null(df)

    assert list(out.columns) == ["AAA_close"]
    assert out["AAA_close"].tolist() == [1.0, 2.0, 3.0]


def test_coerce_utc_naive_index_shared_helper():
    idx = pd.DatetimeIndex(["2024-01-02 09:30:00-05:00", "2024-01-03 16:00:00-05:00"])
    out = coerce_utc_naive_index(idx, normalize=True)
    assert out.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_collapse_duplicate_index_rows_keeps_last_non_null_per_column():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02"])
    df = pd.DataFrame({"AAA_close": [1.0, None], "BBB_close": [None, 2.0]}, index=idx)
    out = collapse_duplicate_index_rows(df)
    assert out.shape == (1, 2)
    assert float(out.iloc[0]["AAA_close"]) == 1.0
    assert float(out.iloc[0]["BBB_close"]) == 2.0


def test_derive_date_range_ytd_uses_calendar_year_start():
    start, end = derive_date_range(
        {"download_period": "ytd", "download_end_date": "2026-02-23"}
    )
    assert isinstance(start, str)
    assert start == "2026-01-01"
    assert end == "2026-02-23"


def test_derive_date_range_coerces_date_like_inputs_to_iso_strings():
    start, end = derive_date_range(
        {
            "download_start_date": pd.Timestamp("2024-01-02", tz="UTC"),
            "download_end_date": pd.Timestamp("2024-03-01"),
        }
    )
    assert start == "2024-01-02"
    assert end == "2024-03-01"
