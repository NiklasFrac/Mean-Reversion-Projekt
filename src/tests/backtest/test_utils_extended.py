import pandas as pd
import pytest

from backtest.utils import alpha, be, tz


def test_alpha_extra_metrics() -> None:
    assert alpha.DEFAULT_COINT_ALPHA == pytest.approx(0.05)
    assert alpha.DEFAULT_PREFILTER_MIN_OBS >= 2


def test_tz_helpers_more() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D").tz_localize("UTC")
    df = pd.DataFrame({"v": [1.0, 2.0]}, index=idx)

    assert tz._tz_to_str(df.index.tz) == "UTC"
    assert tz._extract_tz_from_index_like(df.index) == "UTC"
    assert tz._normalize_tz_name("US/Eastern") == tz.NY_TZ

    out = tz.ensure_index_tz(df, tz.NY_TZ)
    assert str(out.index.tz) == tz.NY_TZ

    ts = tz.align_ts_to_series(pd.Timestamp("2024-01-02"), out)
    tz.same_tz_or_raise(out.index, ts, allow_naive_pair=False)

    naive = tz.to_naive_local(out)
    assert naive.index.tz is None

    cfg = {"backtest": {"timezone": "US/Eastern"}}
    assert tz.get_ex_tz(cfg, prices=None) == tz.NY_TZ


def test_be_more_helpers() -> None:
    assert be._safe_float("bad", default=1.0) == 1.0
    assert be._safe_int("bad", default=2) == 2
    assert be._nan_to_zero(float("nan")) == 0.0

    t0 = be._as_timestamp("2024-01-02")
    assert t0 is not None

    ticks = be._ticks_from_prices(10.0, 10.2, 0.01)
    assert ticks == pytest.approx(-20.0)

    fills = [{"qty": 1, "price": 10.0}, {"qty": 2, "price": 10.5}]
    vwap, notional, qty = be._compute_vwap_and_totals(fills)
    assert qty == 3
    assert vwap == pytest.approx(notional / qty)

    fee, _, _ = be._calc_fees_from_fills(
        fills, maker_bps=-1.0, taker_bps=2.0, min_fee=0.0
    )
    assert fee < 0.0

    df = pd.DataFrame({"exit_ts": ["2024-01-02"], "entry_ts": ["2024-01-01"]})
    assert be._infer_exit_column(df) == "exit_ts"

    prices = {
        "AAA": pd.Series([1.0, 1.1], index=pd.date_range("2024-01-01", periods=2))
    }
    assert be._series_price_at(prices, "AAA", pd.Timestamp("2024-01-02")) == 1.1
