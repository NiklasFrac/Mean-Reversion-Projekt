from __future__ import annotations

import pandas as pd

from universe import vendor


def test_period_to_offset_parses_units():
    assert vendor._period_to_offset("5d").days == 5
    assert vendor._period_to_offset("2w") == pd.Timedelta(weeks=2)
    assert vendor._period_to_offset("3mo").months == 3  # type: ignore[attr-defined]
    assert vendor._period_to_offset("max").years == 50  # type: ignore[attr-defined]


def test_period_start_from_end_handles_ytd():
    end_ts = pd.Timestamp("2026-02-23")
    start = vendor.period_start_from_end(end_ts=end_ts, period="ytd")
    assert start == pd.Timestamp("2026-01-01")


def test_vendor_config_from_mapping_uses_shared_coercion():
    cfg = vendor.VendorConfig.from_mapping(
        {
            "rate_limit_per_sec": "bad",
            "max_retries": -3,
            "base_backoff": "bad",
            "backoff_factor": 0,
            "cooldown_after_rate_limit": -1,
            "use_internal_rate_limiter": "false",
        }
    )

    assert cfg.rate_limit_per_sec == 0.5
    assert cfg.max_retries == 0
    assert cfg.base_backoff == 0.75
    assert cfg.backoff_factor == 2.0
    assert cfg.cooldown_after_rate_limit == 0.0
    assert cfg.use_internal_rate_limiter is False


def test_clean_panel_normalizes_timezone_and_deduplicates_columns():
    idx = pd.date_range("2024-01-01", periods=2, tz="America/New_York")
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], columns=["AAA_close", "AAA_close"], index=idx
    )
    cleaned = vendor._clean_panel(df)
    assert cleaned.index.tz is None
    # Duplicate columns get dropped
    assert list(cleaned.columns) == ["AAA_close"]


def test_fetch_price_volume_data_handles_mixed_tz_chunk_indexes(monkeypatch):
    calls = {"n": 0}

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
    ):
        calls["n"] += 1
        idx = (
            pd.date_range("2024-01-01", periods=2, tz="America/New_York")
            if calls["n"] == 1
            else pd.date_range("2024-01-01", periods=2)
        )
        sym = chunk[0]
        prices = pd.DataFrame({f"{sym}_close": [1.0, 1.1]}, index=idx)
        vols = pd.DataFrame({sym: [100.0, 110.0]}, index=idx)
        return prices, vols

    monkeypatch.setattr(vendor, "_download_chunk", fake_download_chunk)

    prices, vols = vendor.fetch_price_volume_data(
        ["AAA", "BBB"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d",
        batch_size=1,
        pause=0.0,
        retries=0,
        backoff=1.0,
        use_threads=False,
        auto_adjust=False,
        show_progress=False,
    )

    assert prices.index.tz is None
    assert vols.index.tz is None
    assert set(prices.columns) == {"AAA_close", "BBB_close"}
    assert set(vols.columns) == {"AAA", "BBB"}


def test_retry_missing_history_reports_remaining(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=1)
    prices_df = pd.DataFrame({"AAA_close": [1.0]}, index=idx)
    vols_df = pd.DataFrame({"AAA": [1.0]}, index=idx)

    def fake_fetch(*args, **kwargs):
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(vendor, "fetch_price_volume_data", fake_fetch)

    _, _, missing = vendor._retry_missing_history(
        ["AAA", "BBB"],
        prices_df,
        vols_df,
        start_date="2024-01-01",
        end_date="2024-01-02",
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=True,
    )

    assert missing == ["BBB"]


def test_download_chunk_respects_end_date(monkeypatch):
    calls: list[dict] = []

    def fake_download(tickers, **kwargs):
        calls.append(kwargs)
        idx = pd.date_range("2024-01-01", periods=2)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
        df = pd.DataFrame(
            [[1.0, 100.0, 2.0, 200.0], [1.1, 110.0, 2.1, 210.0]],
            index=idx,
            columns=cols,
        )
        return df

    monkeypatch.setattr(vendor.yf, "download", fake_download)

    prices, vols = vendor._download_chunk(
        ["AAA", "BBB"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=False,
        start_date="2019-01-31",
        end_date="2024-01-31",
    )
    assert prices is not None and not prices.empty
    assert vols is not None and not vols.empty
    assert calls, "yfinance.download should be called"
    kwargs = calls[0]
    assert kwargs.get("start") == "2019-01-31"
    assert kwargs.get("end") == "2024-02-01"
    assert "period" not in kwargs


def test_download_chunk_passes_timeout_to_yf_download(monkeypatch):
    calls: list[dict] = []

    def fake_download(tickers, **kwargs):
        calls.append(kwargs)
        idx = pd.date_range("2024-01-01", periods=1)
        cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
        return pd.DataFrame([[1.0, 100.0, 2.0, 200.0]], index=idx, columns=cols)

    monkeypatch.setattr(vendor.yf, "download", fake_download)

    prices, vols = vendor._download_chunk(
        ["AAA", "BBB"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=False,
        start_date="2024-01-01",
        end_date="2024-01-31",
        request_timeout=9.0,
    )

    assert not prices.empty and not vols.empty
    assert calls
    assert calls[0].get("timeout") == 9.0


def test_download_chunk_passes_timeout_to_single_ticker_history(monkeypatch):
    seen: dict[str, object] = {}

    class DummyTicker:
        def history(self, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            idx = pd.date_range("2024-01-01", periods=1)
            return pd.DataFrame({"Close": [3.0], "Volume": [300.0]}, index=idx)

    monkeypatch.setattr(vendor.yf, "Ticker", lambda _sym: DummyTicker())

    prices, vols = vendor._download_chunk(
        ["AAA"],
        interval="1d",
        pause=0.0,
        retries=0,
        backoff=1.0,
        auto_adjust=False,
        start_date="2024-01-01",
        end_date="2024-01-31",
        request_timeout=11.0,
    )

    assert not prices.empty and not vols.empty
    assert seen["timeout"] == 11.0


def test_yfinance_vendor_uses_rate_limit_cooldown(monkeypatch):
    cfg = vendor.VendorConfig(
        rate_limit_per_sec=1000.0,
        max_retries=1,
        base_backoff=0.1,
        backoff_factor=2.0,
        cooldown_after_rate_limit=3.0,
    )
    client = vendor.YFinanceVendor(cfg)

    sleeps: list[float] = []
    monkeypatch.setattr(vendor.time, "sleep", lambda s: sleeps.append(float(s)))

    attempts = {"n": 0}

    def _flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("HTTP 429 Too Many Requests")
        return "ok"

    out = client._with_retries(_flaky)
    assert out == "ok"
    assert sleeps
    assert sleeps[0] >= 3.0


def test_rate_limiter_supports_sub_one_rates(monkeypatch):
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def _fake_time() -> float:
        return float(clock["t"])

    def _fake_sleep(seconds: float) -> None:
        sleeps.append(float(seconds))
        clock["t"] += float(seconds)
        if len(sleeps) > 10:
            raise RuntimeError("rate limiter did not release token")

    monkeypatch.setattr(vendor.time, "time", _fake_time)
    monkeypatch.setattr(vendor.time, "sleep", _fake_sleep)

    limiter = vendor._RateLimiter(0.5)
    limiter.take()  # initial burst token
    limiter.take()  # should wait ~2s (1 / 0.5)

    assert len(sleeps) == 1
    assert 1.9 <= sleeps[0] <= 2.1


def test_fetch_quote_bundle_retries_info_and_holders(monkeypatch):
    class DummyTicker:
        def __init__(self) -> None:
            self.info_calls = 0
            self.holder_calls = 0
            self.fast_info = {"lastPrice": 10.0}

        def get_info(self):
            self.info_calls += 1
            if self.info_calls < 2:
                raise RuntimeError("temporary info error")
            return {"quoteType": "EQUITY"}

        def get_major_holders(self):
            self.holder_calls += 1
            if self.holder_calls < 2:
                raise RuntimeError("temporary holders error")
            return pd.DataFrame({"v": [1.0], "label": ["dummy"]})

    dummy = DummyTicker()
    monkeypatch.setattr(vendor.yf, "Ticker", lambda _sym: dummy)
    monkeypatch.setattr(vendor.time, "sleep", lambda _s: None)

    client = vendor.YFinanceVendor(
        vendor.VendorConfig(
            rate_limit_per_sec=1_000.0,
            max_retries=2,
            base_backoff=0.0,
            backoff_factor=1.0,
        )
    )
    payload = client.fetch_quote_bundle(
        "AAA", include_major_holders=True, request_timeout=2.0
    )

    assert dummy.info_calls == 2
    assert dummy.holder_calls == 2
    assert payload["info"]["quoteType"] == "EQUITY"
    assert payload["major_holders"] is not None


def test_fetch_quote_bundle_forwards_timeout_to_timeout_wrapper(monkeypatch):
    class DummyTicker:
        def __init__(self) -> None:
            self.fast_info = {"lastPrice": 10.0}

        def get_info(self):
            return {"quoteType": "EQUITY"}

        def get_major_holders(self):
            return None

    dummy = DummyTicker()
    monkeypatch.setattr(vendor.yf, "Ticker", lambda _sym: dummy)

    client = vendor.YFinanceVendor(
        vendor.VendorConfig(rate_limit_per_sec=1_000.0, max_retries=0)
    )
    seen: list[float | None] = []

    def _fake_run_with_timeout(func, timeout):
        seen.append(timeout)
        return func()

    monkeypatch.setattr(client, "_run_with_timeout", _fake_run_with_timeout)
    payload = client.fetch_quote_bundle(
        "AAA", include_major_holders=True, request_timeout=7.5
    )

    assert payload["info"]["quoteType"] == "EQUITY"
    # Calls: fast_info, info, major holders.
    assert seen == [7.5, 7.5, 7.5]
