from __future__ import annotations

import pandas as pd
import pytest

from universe.downloads import fetch_unadjusted_panels, validate_unadjusted_vs_adjusted


def test_fetch_unadjusted_panels_returns_empty_meta_for_no_tickers():
    prices, volumes, missing, meta = fetch_unadjusted_panels(
        [],
        download_start_date="2024-01-01",
        download_end_date="2024-01-31",
        download_interval="1d",
        batch_size=1,
        pause=0.0,
        max_retries=0,
        backoff_factor=1.0,
        use_threads=False,
        max_download_workers=None,
        stop_event=None,
        progress_bar=False,
        junk_filter=None,
        fetch_price_volume_data_fn=lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
        retry_missing_history_fn=lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), []),
    )

    assert prices.empty
    assert volumes.empty
    assert missing == []
    assert meta == {
        "wanted": 0,
        "coverage_threshold": 0.85,
        "retried": False,
        "observed": 0,
        "coverage_ratio": 0.0,
    }


def test_fetch_unadjusted_panels_requires_both_price_and_volume_for_coverage():
    idx = pd.date_range("2024-01-01", periods=3)

    def _fetch(*args, **kwargs):
        _ = args, kwargs
        return pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx), pd.DataFrame(
            index=idx
        )

    def _retry(*args, **kwargs):
        _ = args, kwargs
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx),
            pd.DataFrame(index=idx),
            [],
        )

    prices, volumes, missing, meta = fetch_unadjusted_panels(
        ["AAA"],
        download_start_date="2024-01-01",
        download_end_date="2024-01-31",
        download_interval="1d",
        batch_size=1,
        pause=0.0,
        max_retries=0,
        backoff_factor=1.0,
        use_threads=False,
        max_download_workers=None,
        stop_event=None,
        progress_bar=False,
        junk_filter=None,
        fetch_price_volume_data_fn=_fetch,
        retry_missing_history_fn=_retry,
    )

    assert not prices.empty
    assert volumes.empty
    assert missing == ["AAA"]
    assert meta["retried"] is True
    assert meta["observed"] == 0
    assert meta["coverage_ratio"] == 0.0


def test_fetch_unadjusted_panels_skips_retry_on_full_coverage():
    idx = pd.date_range("2024-01-01", periods=3)
    fetch_calls = {"n": 0}
    retry_calls = {"n": 0}

    def _fetch(*args, **kwargs):
        _ = args, kwargs
        fetch_calls["n"] += 1
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx),
            pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx),
        )

    def _retry(*args, **kwargs):
        _ = args, kwargs
        retry_calls["n"] += 1
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx),
            pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx),
            [],
        )

    prices, volumes, missing, meta = fetch_unadjusted_panels(
        ["AAA"],
        download_start_date="2024-01-01",
        download_end_date="2024-01-31",
        download_interval="1d",
        batch_size=1,
        pause=0.0,
        max_retries=0,
        backoff_factor=1.0,
        use_threads=False,
        max_download_workers=None,
        stop_event=None,
        progress_bar=False,
        junk_filter=None,
        fetch_price_volume_data_fn=_fetch,
        retry_missing_history_fn=_retry,
    )

    assert fetch_calls["n"] == 1
    assert retry_calls["n"] == 1  # only the initial normalization retry pass
    assert not prices.empty
    assert not volumes.empty
    assert missing == []
    assert meta["retried"] is False
    assert meta["observed"] == 1
    assert meta["coverage_ratio"] == 1.0


def test_fetch_unadjusted_panels_clamps_invalid_coverage_threshold():
    idx = pd.date_range("2024-01-01", periods=3)
    fetch_calls = {"n": 0}
    retry_calls = {"n": 0}

    def _fetch(*args, **kwargs):
        _ = args, kwargs
        fetch_calls["n"] += 1
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx),
            pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx),
        )

    def _retry(*args, **kwargs):
        _ = args, kwargs
        retry_calls["n"] += 1
        return (
            pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx),
            pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx),
            [],
        )

    _, _, missing, meta = fetch_unadjusted_panels(
        ["AAA"],
        download_start_date="2024-01-01",
        download_end_date="2024-01-31",
        download_interval="1d",
        batch_size=1,
        pause=0.0,
        max_retries=0,
        backoff_factor=1.0,
        use_threads=False,
        max_download_workers=None,
        stop_event=None,
        progress_bar=False,
        junk_filter=None,
        coverage_threshold=1.5,
        fetch_price_volume_data_fn=_fetch,
        retry_missing_history_fn=_retry,
    )

    assert fetch_calls["n"] == 1
    assert retry_calls["n"] == 1  # initial retry_missing_history pass only
    assert missing == []
    assert meta["coverage_threshold"] == 1.0
    assert meta["retried"] is False


def test_validate_unadjusted_vs_adjusted_warns_by_default(caplog):
    idx = pd.date_range("2024-01-01", periods=6)
    adj = pd.DataFrame({"AAA_close": [10, 11, 12, 13, 14, 15]}, index=idx)
    unadj = adj.copy()

    with caplog.at_level("WARNING", logger="runner_universe"):
        validate_unadjusted_vs_adjusted(
            prices_adjusted=adj,
            prices_unadjusted=unadj,
            interval="1d",
            min_rows=5,
        )

    assert "identical" in caplog.text


def test_validate_unadjusted_vs_adjusted_raises_when_strict():
    idx = pd.date_range("2024-01-01", periods=6)
    adj = pd.DataFrame({"AAA_close": [10, 11, 12, 13, 14, 15]}, index=idx)
    unadj = adj.copy()

    with pytest.raises(RuntimeError):
        validate_unadjusted_vs_adjusted(
            prices_adjusted=adj,
            prices_unadjusted=unadj,
            interval="1d",
            min_rows=5,
            strict=True,
        )
