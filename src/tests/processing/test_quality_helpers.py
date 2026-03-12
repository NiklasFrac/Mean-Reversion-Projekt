# tests/processing/test_quality_helpers.py
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from processing.outliers import robust_outlier_mask_causal, safe_log
from processing.quality import quality_gates
from processing.quality_helpers import validate_prices_wide
from processing.timebase import ensure_utc_index, pick_time_grid


def test_ensure_utc_index_duplicates_median():
    ts = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, tzinfo=UTC),  # duplicate
        datetime(2020, 1, 2, tzinfo=UTC),
    ]
    df = pd.DataFrame({"A": [1.0, 3.0, 2.0]}, index=ts)
    out = ensure_utc_index(df)
    # duplicate timestamps are aggregated via median => (1,3) -> 2 on 2020-01-01
    assert list(out.index.astype("datetime64[ns]")) == list(
        pd.to_datetime(["2020-01-01", "2020-01-02"])
    )
    assert out.loc["2020-01-01", "A"] == 2.0


def test_validate_prices_wide_basic():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"A": [1.0, np.nan, 2.0], "B": [1.0, 1.0, 1.0]}, index=idx)
    checks = validate_prices_wide(df)["checks"]
    assert checks["rows"] == 3 and checks["cols"] == 2
    assert checks["na_total"] == 1
    assert checks["nonpositive_prices"] == 0
    assert checks["monotonic_index"] == 1


def test_validate_prices_wide_handles_malformed_values():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"A": [1.0, "bad", -2.0]}, index=idx)
    checks = validate_prices_wide(df)["checks"]
    assert checks["rows"] == 3 and checks["cols"] == 1
    assert checks["na_total"] == 1
    assert checks["nonpositive_prices"] == 1


def _toy_prices():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "A": [1, 2, np.nan, 4, 5],
            "B": [np.nan, np.nan, 3, np.nan, 5],
        },
        index=idx,
    )
    return df


def test_pick_time_grid_modes():
    df = _toy_prices()
    leader = pick_time_grid(df, mode="leader")
    assert isinstance(leader, pd.DatetimeIndex)
    inter = pick_time_grid(df, mode="intersection")
    union = pick_time_grid(df, mode="union")
    assert len(inter) <= len(leader) <= len(union)


def test_safe_log_series_and_df_shapes():
    s = pd.Series([1.0, 2.0, 0.0, -1.0, np.inf, np.nan])
    out_s = safe_log(s)
    assert isinstance(out_s, pd.Series) and out_s.shape == s.shape
    df = pd.DataFrame({"A": s, "B": s})
    out_df = safe_log(df)
    assert isinstance(out_df, pd.DataFrame) and out_df.shape == df.shape
    assert np.isfinite(out_s.dropna()).all()


def test_robust_outlier_mask_and_quality_gates():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    s = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    s.iloc[15] = 1000.0  # outlier
    mask = robust_outlier_mask_causal(s, zscore=6.0, window=11, use_log_returns=True)
    assert mask.sum() >= 1
    ok, diag = quality_gates(s, ref_len=len(idx), non_na_min_pct=0.9, max_gap=3)
    assert ok is True and "non_na_pct" in diag
