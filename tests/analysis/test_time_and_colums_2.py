import numpy as np
import pandas as pd
import pytest

from analysis.preprocess import ensure_utc_index, select_price_columns


@pytest.mark.unit
def test_ensure_utc_index_sorts_and_dedups():
    ts = pd.date_range("2024-01-01", periods=5, tz="UTC")
    # unsorted + duplicate last
    idx = pd.Index([ts[2], ts[0], ts[1], ts[3], ts[3]])
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5]}, index=idx.tz_localize(None))
    out = ensure_utc_index(df)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    # duplicate was dropped (keep='last')
    assert out.shape[0] == 4


@pytest.mark.unit
def test_select_price_columns_wide_numeric():
    df = pd.DataFrame(
        {
            "A": [1.0, 1.1, 1.2],
            "B": [2.0, 2.1, 2.2],
            "volume": [10, 11, 12],  # soll raus
        },
        index=pd.date_range("2023-01-01", periods=3, tz="UTC"),
    )
    out = select_price_columns(df)
    assert list(out.columns) == ["A", "B"]


@pytest.mark.unit
def test_select_price_columns_multiindex_last_level_matches():
    arrays = [
        ["AAA", "AAA", "BBB", "BBB"],
        ["close", "volume", "close", "turnover"],
    ]
    cols = pd.MultiIndex.from_arrays(arrays, names=("id", "field"))
    df = pd.DataFrame(
        np.arange(12, dtype=float).reshape(3, 4),
        columns=cols,
        index=pd.date_range("2020-01-01", periods=3, tz="UTC"),
    )
    out = select_price_columns(df)
    assert list(out.columns) == ["AAA", "BBB"]
    assert out.iloc[0].tolist() == [0.0, 2.0]


@pytest.mark.unit
def test_select_price_columns_long_pivot():
    # long → pivot via ("asset_id", "<pricecol>")
    idx = pd.date_range("2024-02-01", periods=3, tz="UTC")
    long = pd.DataFrame(
        {
            "asset_id": ["X", "Y", "X"],
            "price": [10.0, 20.0, 11.0],
        },
        index=idx,
    )
    out = select_price_columns(long)
    assert set(out.columns) == {"X", "Y"}
