import pandas as pd
import pytest

import analysis.data_analysis as da


def test_select_price_columns_long_pivot():
    idx = pd.date_range("2022-01-03", periods=3, tz="UTC")
    df_long = pd.DataFrame(
        {"asset_id": ["A", "B", "A"], "close": [100, 200, 101], "volume": [10, 20, 10]},
        index=idx,
    )
    wide = da.select_price_columns(df_long)
    assert set(wide.columns) == {"A", "B"}


def test_select_price_columns_error_no_numeric():
    idx = pd.date_range("2022-01-03", periods=2, tz="UTC")
    df = pd.DataFrame({"foo": ["x", "y"], "bar": ["u", "v"]}, index=idx)
    with pytest.raises(ValueError):
        da.select_price_columns(df)
