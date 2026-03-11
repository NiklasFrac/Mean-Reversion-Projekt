from __future__ import annotations

import pandas as pd

from universe.ticker_sets import price_tickers, volume_tickers


def test_price_tickers_supports_close_and_bare_columns():
    idx = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {
            "AAA_close": [10.0, 11.0],
            "BBB_close": [float("nan"), float("nan")],
            "CCC": [1.0, 2.0],
            "DDD_open": [1.0, 2.0],
        },
        index=idx,
    )

    out = price_tickers(df, require_data=True, include_bare_columns=True)

    assert out == {"AAA", "CCC"}


def test_volume_tickers_honors_require_data():
    idx = pd.date_range("2024-01-01", periods=2)
    vols = pd.DataFrame(
        {"AAA": [100.0, 120.0], "BBB": [float("nan"), float("nan")]}, index=idx
    )

    assert volume_tickers(vols, require_data=False) == {"AAA", "BBB"}
    assert volume_tickers(vols, require_data=True) == {"AAA"}
