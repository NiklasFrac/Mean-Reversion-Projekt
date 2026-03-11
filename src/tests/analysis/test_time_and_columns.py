# src/tests/analysis/test_time_and_columns.py
import logging

import pandas as pd

from analysis.preprocess import ensure_utc_index, select_price_columns


def test_ensure_utc_index_duplicates_and_sort(caplog):
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"A": [1, 2, 3]}, index=idx)  # dupe "2024-01-02"
    out = ensure_utc_index(df)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    # Keep='last' -> Wert 3 bleibt für 2024-01-02
    assert out.iloc[-1]["A"] == 3


def test_select_price_columns_multiindex():
    arrays = [["A", "B"], ["open", "close"]]
    cols = pd.MultiIndex.from_product(arrays, names=["ticker", "field"])
    # 2 Zeilen, 4 Spalten
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=cols)

    prices = select_price_columns(df)
    assert list(prices.columns) == ["A", "B"]
    assert prices.iloc[0].tolist() == [2, 4]  # close-Werte


def test_select_price_columns_long_format():
    idx = pd.date_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {"asset_id": ["A", "B", "A"], "price": [10.0, 20.0, 11.0]}, index=idx
    )
    wide = select_price_columns(df)
    assert set(wide.columns) == {"A", "B"}


def test_select_price_columns_preferred_field_open():
    arrays = [["A", "B"], ["open", "close"]]
    cols = pd.MultiIndex.from_product(arrays, names=["ticker", "field"])
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=cols)

    prices = select_price_columns(df, preferred_field="open")
    assert list(prices.columns) == ["A", "B"]
    assert prices.iloc[0].tolist() == [1, 3]


def test_select_price_columns_preferred_missing_falls_back(caplog):
    arrays = [["A", "B"], ["open", "close"]]
    cols = pd.MultiIndex.from_product(arrays, names=["ticker", "field"])
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=cols)

    with caplog.at_level(logging.WARNING):
        prices = select_price_columns(df, preferred_field="adj_close")
    assert "falling back" in caplog.text
    assert list(prices.columns) == ["A", "B"]
