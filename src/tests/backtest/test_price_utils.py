from __future__ import annotations

import pandas as pd

from backtest.utils.common import prices as price_utils


def _series_naive() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.Series([1.0, 2.0, 3.0], index=idx)


def test_as_price_map_coerce_timezone_exchange() -> None:
    out = price_utils.as_price_map({"AAA": _series_naive()}, coerce_timezone="exchange")
    idx = out["AAA"].index
    assert idx.tz is not None
    assert str(idx.tz) in {"America/New_York", "US/Eastern"}


def test_as_price_map_coerce_timezone_utc() -> None:
    out = price_utils.as_price_map({"AAA": _series_naive()}, coerce_timezone="utc")
    assert str(out["AAA"].index.tz) == "UTC"


def test_as_price_map_coerce_timezone_local_naive() -> None:
    out = price_utils.as_price_map(
        {"AAA": _series_naive()}, coerce_timezone="local_naive"
    )
    assert out["AAA"].index.tz is None


def test_as_price_map_unknown_timezone_policy_keeps_index() -> None:
    src = _series_naive()
    out = price_utils.as_price_map({"AAA": src}, coerce_timezone="does_not_exist")
    assert out["AAA"].index.equals(src.index)


def test_series_price_at_requires_symbol_for_mapping() -> None:
    prices = {"AAA": _series_naive()}
    assert price_utils.series_price_at(prices, None, pd.Timestamp("2024-01-02")) is None
