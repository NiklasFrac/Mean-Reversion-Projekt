from __future__ import annotations

import numpy as np
import pandas as pd

from processing.liquidity import (
    build_adv_map_from_price_and_volume,
    compute_adv,
    compute_adv_series,
)


def test_compute_adv_series_and_compute_adv_basic_and_empty():
    # empty
    vol_empty = pd.DataFrame()
    assert compute_adv_series(vol_empty).empty
    assert compute_adv(vol_empty).empty

    # basic roll-mean
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    vol = pd.DataFrame({"AAA": [10, 10, 10, 10, 10], "BBB": [1, 2, 3, 4, 5]}, index=idx)
    adv_series = compute_adv_series(vol, window=3, min_periods=2)
    assert list(adv_series.columns) == ["AAA", "BBB"]
    assert adv_series.loc[idx[-1], "AAA"] == 10  # konstant
    assert adv_series.loc[idx[-1], "BBB"] == np.mean([3, 4, 5])

    adv_last = compute_adv(vol, window=3, min_periods=2)
    assert adv_last["AAA"] == 10
    assert adv_last["BBB"] == np.mean([3, 4, 5])


def test_build_adv_map_from_price_and_volume():
    idx = pd.date_range("2020-01-01", periods=3, freq="B", tz="UTC")
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0], "BBB": [5.0, 6.0, np.nan]}, index=idx
    )
    volume = pd.DataFrame(
        {"AAA": [10.0, 10.0, 10.0], "BBB": [100.0, 100.0, 100.0]}, index=idx
    )

    result = build_adv_map_from_price_and_volume(prices, volume, window=2)
    # keys pro price-spalte
    assert set(result.keys()) == {"AAA", "BBB"}
    assert {"adv", "last_price"} <= set(result["AAA"].keys())

    # Werte sind float-castbar und NaN bei fehlendem last_price
    assert isinstance(result["AAA"]["adv"], float)
    assert result["AAA"]["last_price"] == 3.0
    assert np.isnan(result["BBB"]["last_price"])

    # empty input → {}
    assert build_adv_map_from_price_and_volume(pd.DataFrame(), pd.DataFrame()) == {}


def test_build_adv_map_respects_missing_volume_no_unbounded_fill():
    idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, np.nan, np.nan, np.nan]}, index=idx)

    result = build_adv_map_from_price_and_volume(prices, volume, window=2)
    assert np.isnan(result["AAA"]["adv"])
