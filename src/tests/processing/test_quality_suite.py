from __future__ import annotations

import types
from typing import cast

import numpy as np
import pandas as pd
import pytest

from processing.quality import (
    build_trading_index,
    longest_nan_run,
    quality_gates,
    robust_outlier_mask,
    scrub_outliers,
)


def _df_for_grid() -> pd.DataFrame:
    idx = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"], utc=True
    )
    a = pd.Series([1.0, np.nan, 3.0, 4.0], index=idx, name="A")
    b = pd.Series([1.1, 2.2, np.nan, 4.4], index=idx, name="B")
    c = pd.Series([np.nan, 2.0, 3.0, 4.0], index=idx, name="C")
    return pd.concat([a, b, c], axis=1)


def test_build_trading_index_leader_intersection_union():
    raw = _df_for_grid()

    # leader: column with the most non-NaNs -> A here (3 values)
    idx_leader = build_trading_index(raw, mode="leader")
    assert isinstance(idx_leader, pd.DatetimeIndex)
    assert len(idx_leader) == 3
    assert idx_leader.is_monotonic_increasing

    # intersection: timestamps shared without NaN in all columns -> only 2020-01-06
    idx_inter = build_trading_index(raw, mode="intersection")
    assert list(idx_inter) == [pd.Timestamp("2020-01-06T00:00:00Z")]

    # union: all timestamps that are valid somewhere
    idx_union = build_trading_index(raw, mode="union")
    assert list(idx_union) == list(
        pd.DatetimeIndex(sorted(raw.dropna(how="all").index.unique()))
    )


def test_build_trading_index_calendar_fallback(monkeypatch: pytest.MonkeyPatch):
    # Stub for pandas_market_calendars -> schedule raises -> fallback to bdate_range
    fake = types.ModuleType("pandas_market_calendars")

    class _Cal:
        @staticmethod
        def schedule(*_args, **_kwargs):
            raise RuntimeError("boom")

    fake.get_calendar = lambda _name: _Cal()  # type: ignore[assignment]
    monkeypatch.setitem(__import__("sys").modules, "pandas_market_calendars", fake)

    raw = _df_for_grid()
    start = pd.to_datetime(raw.index.min())
    end = pd.to_datetime(raw.index.max())
    idx_cal = build_trading_index(raw, mode="calendar", calendar="XNYS")

    # expectation exactly like bdate_range (fallback)
    exp = pd.bdate_range(start=start, end=end)
    assert isinstance(idx_cal, pd.DatetimeIndex)
    assert list(idx_cal) == list(exp)


def test_robust_outlier_mask_and_scrub_outliers_detects_spike():
    """
    Important: with perfectly constant returns, MAD would be 0 -> mask=False.
    We add deterministic tiny jitter so MAD>0 and the spike
    the spike is flagged reliably (matches realistic data).
    """
    rng = np.random.default_rng(12345)
    pre = 100.0 + rng.normal(0, 0.01, 20)
    post = 100.0 + rng.normal(0, 0.01, 20)
    base = list(pre) + [1_000_000.0] + list(post)

    s = pd.Series(
        base,
        index=pd.date_range("2020-01-01", periods=len(base), freq="B", tz="UTC"),
    )

    m = robust_outlier_mask(s, zscore=3.0, window=9, use_log_returns=True)
    assert m.dtype == bool
    assert m.any(), "Spike in returns should be detected as an outlier"

    s2, n = scrub_outliers(s, zscore=3.0, window=9, use_log_returns=True)
    assert n >= 1
    assert not (cast(pd.Series, s2.dropna()) <= 0).any()


def test_longest_nan_run_and_quality_gates_paths():
    # longest_nan_run
    s0 = pd.Series([], dtype=float)
    assert longest_nan_run(s0) == 0

    s = pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, 0.0])
    assert longest_nan_run(s) == 3

    # 1) insufficient_coverage
    ok, diag = quality_gates(s, ref_len=10, non_na_min_pct=0.5)
    assert not ok and diag["reason"] == "insufficient_coverage"

    # 2) large_gap
    ok, diag = quality_gates(s, ref_len=len(s), non_na_min_pct=0.1, max_gap=2)
    assert not ok and diag["reason"] == "large_gap"

    # 3) excess_leading_na
    ok, diag = quality_gates(
        pd.Series([np.nan, np.nan, 1.0, 2.0]),
        ref_len=4,
        max_start_na=1,
        non_na_min_pct=0.1,
    )
    assert not ok and diag["reason"] == "excess_leading_na"

    # 4) excess_trailing_na
    ok, diag = quality_gates(
        pd.Series([1.0, 2.0, np.nan, np.nan]),
        ref_len=4,
        max_end_na=1,
        non_na_min_pct=0.1,
    )
    assert not ok and diag["reason"] == "excess_trailing_na"

    # 5) nonpositive_prices
    ok, diag = quality_gates(pd.Series([1.0, 2.0, 0.0]), ref_len=3, non_na_min_pct=0.1)
    assert not ok and diag["reason"] == "nonpositive_prices"

    # 6) pass
    ok, diag = quality_gates(pd.Series([1.0, 2.0, 3.0, 4.0]), ref_len=4)
    assert ok and "reason" not in diag
