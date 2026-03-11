import numpy as np
import pandas as pd

from backtest.utils import be


def test_be_safe_helpers_and_ticks() -> None:
    assert be._safe_float("bad", 1.0) == 1.0
    assert be._safe_float(float("nan"), 2.0) == 2.0
    assert be._safe_int("bad", 3) == 3
    assert be._nan_to_zero(float("nan")) == 0.0
    assert be._ticks_from_prices(10.0, 9.0, 0.0) == 0.0

    assert be._as_timestamp("not-a-date") is None
    assert be._nan_to_zero(object()) == 0.0


def test_be_fees_slippage_and_time_to_fill_edges() -> None:
    fills = [{"qty": 1, "price": 10.0, "liquidity": "M"}]
    fees, m_cnt, t_cnt = be._calc_fees_from_fills(
        fills, maker_bps=1.0, taker_bps=2.0, min_fee=0.5
    )
    assert fees == -0.5
    assert m_cnt == 1 and t_cnt == 0

    bad_fees, _, _ = be._calc_fees_from_fills(
        [{"qty": 0, "price": 10.0}], maker_bps=1.0, taker_bps=2.0
    )
    assert bad_fees == 0.0

    slip = be._calc_slippage_from_fills(fills, side="sell", reference_px=9.5, tick=None)
    assert slip["slippage_ccy"] == 0.0
    assert slip["slippage_ticks_avg"] == 0.0

    slip_skip = be._calc_slippage_from_fills(
        [{"qty": 0, "price": 10.0}, {"qty": 1, "price": 10.0}],
        side="buy",
        reference_px=9.0,
        tick=0.01,
    )
    assert slip_skip["slippage_ccy"] >= 0.0

    slip_empty = be._calc_slippage_from_fills(
        [], side="buy", reference_px=float("nan"), tick=0.01
    )
    assert slip_empty["vwap"] != slip_empty["vwap"]

    assert np.isnan(be._time_to_fill_ms(None, fills))
    assert np.isnan(be._time_to_fill_ms(pd.Timestamp("2024-01-02"), [{"qty": 1}]))
    assert np.isnan(be._time_to_fill_ms(pd.Timestamp("2024-01-02"), [{"ts": "bad"}]))


def test_be_realized_spread_and_series_price_at_edges() -> None:
    assert np.isnan(be._realized_spread_bps(vwap=10.0, reference_mid=0.0, side="buy"))

    idx_tz = pd.date_range("2024-01-02", periods=2, tz="UTC")
    df = pd.DataFrame({"close": [10.0, 10.5]}, index=idx_tz)
    px = be._series_price_at({"AAA": df}, "AAA", pd.Timestamp("2024-01-03"))
    assert px == 10.5

    idx = pd.date_range("2024-01-02", periods=2)
    s = pd.Series([1.0, 1.1], index=idx)
    px2 = be._series_price_at({"BBB": s}, "BBB", pd.Timestamp("2024-01-03", tz="UTC"))
    assert px2 == 1.1

    assert be._series_price_at({"BBB": s}, None, pd.Timestamp("2024-01-03")) is None
    assert be._series_price_at({"CCC": 123}, "CCC", pd.Timestamp("2024-01-03")) is None
    assert (
        be._series_price_at(
            {"AAA": pd.Series([1.0, 2.0], index=[0, 1])},
            "AAA",
            pd.Timestamp("2024-01-03"),
        )
        is None
    )
    assert (
        be._series_price_at(
            {"DDD": pd.Series([np.nan, np.nan], index=idx)},
            "DDD",
            pd.Timestamp("2024-01-03"),
        )
        is None
    )

    assert be._series_price_at({"CCC": s}, "CCC", pd.Timestamp("2024-01-01")) is None
    bad_df = pd.DataFrame({"text": ["a", "b"]}, index=idx)
    assert (
        be._series_price_at({"EEE": bad_df}, "EEE", pd.Timestamp("2024-01-03")) is None
    )

    df_alt = pd.DataFrame({"open": [1.0, 2.0]}, index=idx)
    assert (
        be._series_price_at({"FFF": df_alt}, "FFF", pd.Timestamp("2024-01-03")) == 2.0
    )


def test_be_infer_exit_column_with_datetime_fallback() -> None:
    df = pd.DataFrame(
        {
            "entry_date": pd.date_range("2024-01-01", periods=2),
            "alt_exit": pd.date_range("2024-01-02", periods=2),
        }
    )
    assert be._infer_exit_column(df) == "alt_exit"

    assert be._infer_exit_column(pd.DataFrame()) is None
    df_bad = pd.DataFrame({"exit_bad": ["x", "y"]})
    assert be._infer_exit_column(df_bad) is None


def test_be_pair_parsing_and_annualization() -> None:
    assert be._parse_pair_symbols_generic(None) == (None, None)
    assert be._parse_pair_symbols_generic("") == (None, None)
    assert be._parse_pair_symbols_generic("AAA") == ("AAA", None)
    assert be._parse_pair_symbols("BBB-CCC") == ("BBB", "CCC")

    assert (
        be._infer_annualization_factor(pd.date_range("2024-01-01", periods=3, freq="W"))
        == 52
    )
    assert (
        be._infer_annualization_factor(
            pd.date_range("2024-01-01", periods=3, freq="ME")
        )
        == 12
    )
    assert (
        be._infer_annualization_factor(
            pd.date_range("2024-01-01", periods=3, freq="QE")
        )
        == 4
    )
    assert (
        be._infer_annualization_factor(
            pd.date_range("2024-01-01", periods=3, freq="YE")
        )
        == 1
    )
    assert (
        be._infer_annualization_factor(pd.date_range("2024-01-01", periods=3, freq="h"))
        == 252
    )
    assert (
        be._infer_annualization_factor(pd.DatetimeIndex([pd.Timestamp("2024-01-01")]))
        == 252
    )
