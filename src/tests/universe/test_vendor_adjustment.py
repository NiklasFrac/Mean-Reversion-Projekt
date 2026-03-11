from __future__ import annotations

import time

import pandas as pd

from universe import vendor


def test_fetch_price_volume_data_respects_chunking_and_flags(monkeypatch):
    calls = []

    def fake_download_chunk(
        chunk,
        interval,
        pause,
        retries,
        backoff,
        *,
        auto_adjust,
        start_date,
        end_date,
        junk_filter,
        **_ignored,
    ):
        calls.append((list(chunk), auto_adjust, start_date, end_date))
        idx = pd.date_range("2024-01-01", periods=1)
        prices = pd.DataFrame({f"{sym}_close": [1.0] for sym in chunk}, index=idx)
        vols = pd.DataFrame({sym: [10.0] for sym in chunk}, index=idx)
        return prices, vols

    monkeypatch.setattr(vendor, "_download_chunk", fake_download_chunk)

    prices, vols = vendor.fetch_price_volume_data(
        ["AAA", "BBB", "$JUNK"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        batch_size=2,
        pause=0.1,
        retries=1,
        backoff=2.0,
        use_threads=False,
        auto_adjust=False,
        junk_filter=lambda s: s.startswith("$"),
    )

    assert len(calls) == 1
    chunk, auto_adj, start, end = calls[0]
    assert chunk == ["AAA", "BBB"]
    assert auto_adj is False
    assert start == "2024-01-01"
    assert end == "2024-01-31"
    assert set(prices.columns) == {"AAA_close", "BBB_close"}
    assert set(vols.columns) == {"AAA", "BBB"}


def test_fetch_price_volume_data_parallel_preserves_chunk_order(monkeypatch):
    def fake_download_chunk(
        chunk,
        interval,
        pause,
        retries,
        backoff,
        *,
        auto_adjust,
        start_date,
        end_date,
        request_timeout=None,
        junk_filter=None,
        **_ignored,
    ):
        # Force reversed completion order: first chunk sleeps longer.
        if chunk and chunk[0] == "AAA":
            time.sleep(0.05)
        idx = pd.date_range("2024-01-01", periods=1)
        prices = pd.DataFrame({f"{sym}_close": [1.0] for sym in chunk}, index=idx)
        vols = pd.DataFrame({sym: [10.0] for sym in chunk}, index=idx)
        return prices, vols

    monkeypatch.setattr(vendor, "_download_chunk", fake_download_chunk)

    prices, vols = vendor.fetch_price_volume_data(
        ["AAA", "BBB", "CCC", "DDD"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        batch_size=2,
        pause=0.0,
        retries=0,
        backoff=1.0,
        use_threads=True,
        max_workers=2,
        auto_adjust=True,
        show_progress=False,
    )

    assert list(prices.columns) == ["AAA_close", "BBB_close", "CCC_close", "DDD_close"]
    assert list(vols.columns) == ["AAA", "BBB", "CCC", "DDD"]


def test_fetch_price_volume_data_forwards_request_timeout(monkeypatch):
    seen: list[float | None] = []

    def fake_download_chunk(
        chunk,
        interval,
        pause,
        retries,
        backoff,
        *,
        auto_adjust,
        start_date,
        end_date,
        request_timeout=None,
        junk_filter=None,
        **_ignored,
    ):
        seen.append(request_timeout)
        idx = pd.date_range("2024-01-01", periods=1)
        prices = pd.DataFrame({f"{sym}_close": [1.0] for sym in chunk}, index=idx)
        vols = pd.DataFrame({sym: [10.0] for sym in chunk}, index=idx)
        return prices, vols

    monkeypatch.setattr(vendor, "_download_chunk", fake_download_chunk)

    vendor.fetch_price_volume_data(
        ["AAA", "BBB"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        batch_size=2,
        pause=0.0,
        retries=0,
        backoff=1.0,
        use_threads=False,
        auto_adjust=True,
        show_progress=False,
        request_timeout=12.5,
    )

    assert seen == [12.5]


def test_retry_missing_history_fetches_missing_symbols(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=2)
    prices_df = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
    vols_df = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    seen = {}

    def fake_fetch(
        tickers,
        start_date,
        end_date,
        interval,
        batch_size,
        pause,
        retries,
        backoff,
        use_threads,
        **kwargs,
    ):
        seen["tickers"] = list(tickers)
        assert batch_size == 1  # sequential retry
        return (
            pd.DataFrame({"BBB_close": [2.0, 2.1]}, index=idx),
            pd.DataFrame({"BBB": [20.0, 21.0]}, index=idx),
        )

    monkeypatch.setattr(vendor, "fetch_price_volume_data", fake_fetch)

    new_prices, new_vols, missing = vendor._retry_missing_history(
        ["AAA", "BBB"],
        prices_df,
        vols_df,
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=True,
    )

    assert seen["tickers"] == ["BBB"]
    assert "BBB_close" in new_prices.columns
    assert "BBB" in new_vols.columns
    assert missing == []


def test_retry_missing_history_forwards_request_timeout(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=2)
    prices_df = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
    vols_df = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    seen: dict[str, object] = {}

    def fake_fetch(
        tickers,
        start_date,
        end_date,
        interval,
        batch_size,
        pause,
        retries,
        backoff,
        use_threads,
        **kwargs,
    ):
        seen["request_timeout"] = kwargs.get("request_timeout")
        return (
            pd.DataFrame({"BBB_close": [2.0, 2.1]}, index=idx),
            pd.DataFrame({"BBB": [20.0, 21.0]}, index=idx),
        )

    monkeypatch.setattr(vendor, "fetch_price_volume_data", fake_fetch)

    vendor._retry_missing_history(
        ["AAA", "BBB"],
        prices_df,
        vols_df,
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=True,
        request_timeout=7.0,
    )

    assert seen["request_timeout"] == 7.0


def test_retry_missing_history_prefers_non_null_retry_values_for_duplicate_columns(
    monkeypatch,
):
    idx = pd.date_range("2024-01-01", periods=2)
    prices_df = pd.DataFrame({"AAA_close": [float("nan"), float("nan")]}, index=idx)
    vols_df = pd.DataFrame({"AAA": [float("nan"), float("nan")]}, index=idx)

    def fake_fetch(
        tickers,
        start_date,
        end_date,
        interval,
        batch_size,
        pause,
        retries,
        backoff,
        use_threads,
        **kwargs,
    ):
        assert list(tickers) == ["AAA"]
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0]}, index=idx),
            pd.DataFrame({"AAA": [100.0, 110.0]}, index=idx),
        )

    monkeypatch.setattr(vendor, "fetch_price_volume_data", fake_fetch)

    new_prices, new_vols, missing = vendor._retry_missing_history(
        ["AAA"],
        prices_df,
        vols_df,
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=True,
    )

    assert missing == []
    assert new_prices["AAA_close"].notna().all()
    assert new_vols["AAA"].notna().all()
    assert new_prices["AAA_close"].iloc[-1] == 11.0
    assert new_vols["AAA"].iloc[-1] == 110.0


def test_download_chunk_prefers_close_when_unadjusted(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=2)
    cols = pd.MultiIndex.from_tuples(
        [
            ("Adj Close", "AAA"),
            ("Close", "AAA"),
            ("Volume", "AAA"),
            ("Adj Close", "BBB"),
            ("Close", "BBB"),
            ("Volume", "BBB"),
        ]
    )
    data = [
        [100.0, 10.0, 1000.0, 200.0, 20.0, 2000.0],
        [110.0, 11.0, 1100.0, 210.0, 21.0, 2100.0],
    ]
    panel = pd.DataFrame(data, index=idx, columns=cols)

    def fake_download(chunk, **kwargs):
        assert set(chunk) == {"AAA", "BBB"}
        assert "start" in kwargs and "end" in kwargs
        return panel

    monkeypatch.setattr(vendor.yf, "download", fake_download)

    prices_adj, vols_adj = vendor._download_chunk(
        ["AAA", "BBB"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=True,
        start_date="2024-01-01",
        end_date="2024-01-31",
        junk_filter=None,
    )
    prices_unadj, vols_unadj = vendor._download_chunk(
        ["AAA", "BBB"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=False,
        start_date="2024-01-01",
        end_date="2024-01-31",
        junk_filter=None,
    )

    assert prices_adj.loc[idx[0], "AAA_close"] == 100.0
    assert prices_unadj.loc[idx[0], "AAA_close"] == 10.0
    assert prices_adj.loc[idx[0], "BBB_close"] == 200.0
    assert prices_unadj.loc[idx[0], "BBB_close"] == 20.0
    assert vols_unadj["AAA"].iloc[0] == 1000.0
    assert vols_unadj["BBB"].iloc[0] == 2000.0
    # Ensure volumes unchanged by adjustment choice
    pd.testing.assert_frame_equal(vols_adj, vols_unadj)


def test_download_chunk_single_symbol_uses_close_when_unadjusted(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {"Adj Close": [50.0, 51.0], "Close": [5.0, 6.0], "Volume": [500.0, 600.0]},
        index=idx,
    )

    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, interval, auto_adjust, start=None, end=None, period=None):
            assert auto_adjust is False
            assert start is not None and end is not None
            return df

    monkeypatch.setattr(vendor.yf, "Ticker", lambda sym: DummyTicker(sym))

    def fake_download(*args, **kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr(vendor.yf, "download", fake_download)

    prices, vols = vendor._download_chunk(
        ["AAA"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=False,
        start_date="2024-01-01",
        end_date="2024-01-31",
        junk_filter=None,
    )

    assert prices["AAA_close"].iloc[-1] == 6.0  # from Close, not Adj Close
    assert vols["AAA"].iloc[-1] == 600.0
