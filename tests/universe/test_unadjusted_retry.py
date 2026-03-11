import pandas as pd

import universe.runner_universe as ru


def test_unadjusted_retry_increases_coverage(monkeypatch):
    calls: list[int] = []
    call_tickers: list[list[str]] = []

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
        *,
        max_workers=None,
        stop_event=None,
        auto_adjust=True,
        show_progress=False,
        request_timeout=None,
        junk_filter=None,
    ):
        calls.append(1)
        call_tickers.append(list(tickers))
        idx = pd.date_range("2020-01-01", periods=3)
        if len(calls) == 1:
            prices = pd.DataFrame({"AAA_close": [1, 2, 3]}, index=idx)
            vols = pd.DataFrame({"AAA": [1, 1, 1]}, index=idx)
        else:
            prices = pd.DataFrame(
                {"AAA_close": [1, 2, 3], "BBB_close": [4, 5, 6]}, index=idx
            )
            vols = pd.DataFrame({"AAA": [1, 1, 1], "BBB": [2, 2, 2]}, index=idx)
        return prices, vols

    def fake_retry(
        tickers,
        prices_df,
        volumes_df,
        *,
        start_date,
        end_date,
        interval,
        pause,
        retries,
        backoff,
        stop_event=None,
        auto_adjust=True,
        show_progress=False,
        request_timeout=None,
        junk_filter=None,
    ):
        return prices_df, volumes_df, []

    monkeypatch.setattr(ru, "fetch_price_volume_data", fake_fetch)
    monkeypatch.setattr(ru, "_retry_missing_history", fake_retry)

    tickers = ["AAA", "BBB"]
    raw_prices_df, raw_volumes_df, missing_hist, meta = ru._fetch_unadjusted_panels(
        tickers,
        download_start_date=pd.Timestamp("2020-01-01").date(),
        download_end_date=pd.Timestamp("2020-01-03").date(),
        download_interval="1d",
        batch_size=2,
        pause=1.0,
        max_retries=1,
        backoff_factor=1.0,
        use_threads=False,
        max_download_workers=None,
        stop_event=None,
        progress_bar=False,
        request_timeout=20.0,
        junk_filter=None,
        coverage_threshold=0.75,  # force retry because first call returns only 1/2 symbols
    )

    assert len(calls) == 2, "Should retry once when coverage is sparse"
    assert call_tickers[0] == ["AAA", "BBB"]
    assert call_tickers[1] == ["BBB"]
    assert meta["retried"] is True
    assert meta["observed"] == 2
    assert meta["coverage_ratio"] >= 1.0
    assert not missing_hist
    assert {"AAA_close", "BBB_close"} <= set(raw_prices_df.columns)
    assert {"AAA", "BBB"} <= set(raw_volumes_df.columns)
