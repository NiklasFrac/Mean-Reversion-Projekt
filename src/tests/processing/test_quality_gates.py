from __future__ import annotations

import numpy as np
import pandas as pd

from processing.quality import (
    longest_nan_run,
    quality_gates,
)
from processing.quality_helpers import validate_prices_wide
from processing.timebase import ensure_utc_index


def test_ensure_utc_index_sorts_and_dedups():
    idx = pd.to_datetime(
        ["2020-01-01 00:00Z", "2020-01-01 00:00Z", "2020-01-02 00:00Z"]
    )
    df = pd.DataFrame({"A": [1.0, 3.0, 2.0]}, index=idx)
    out = ensure_utc_index(df)
    # duplicates are collapsed via median -> (1.0, 3.0) -> 2.0
    assert out.shape == (2, 1)
    assert float(out.iloc[0, 0]) == 2.0
    assert out.index.is_monotonic_increasing


def test_validate_prices_wide_basic_metrics():
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    df = pd.DataFrame({"A": [1.0, 0.0, np.nan], "B": [2.0, 2.0, 2.0]}, index=idx)
    checks = validate_prices_wide(df)["checks"]
    assert checks["rows"] == 3
    assert checks["cols"] == 2
    assert checks["nonpositive_prices"] >= 1
    assert checks["na_total"] >= 1
    assert checks["monotonic_index"] == 1


def test_longest_nan_run_and_quality_gates_reasons():
    # series with leading NAs, a long gap, and nonpositive values at the end
    s = pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, 0.0])
    assert longest_nan_run(s) == 3

    # 1) insufficient_coverage
    ok, diag = quality_gates(s, ref_len=10, non_na_min_pct=0.5)
    assert not ok and diag["reason"] == "insufficient_coverage"

    # 2) large_gap
    ok, diag = quality_gates(s, ref_len=len(s), non_na_min_pct=0.1, max_gap=2)
    assert not ok and diag["reason"] == "large_gap"

    # 3) excess_leading_na – Coverage-Bedingung bewusst deaktivieren
    ok, diag = quality_gates(
        pd.Series([np.nan, np.nan, 1.0, 2.0]),
        ref_len=4,
        max_start_na=1,
        non_na_min_pct=0.0,
    )
    assert not ok and diag["reason"] == "excess_leading_na"

    # 4) excess_trailing_na
    ok, diag = quality_gates(
        pd.Series([1.0, 2.0, np.nan, np.nan]),
        ref_len=4,
        max_end_na=1,
        non_na_min_pct=0.0,
    )
    assert not ok and diag["reason"] == "excess_trailing_na"

    # 5) nonpositive_prices
    ok, diag = quality_gates(
        pd.Series([1.0, 0.0, -1.0, 2.0]), ref_len=4, forbid_nonpositive=True
    )
    assert not ok and diag["reason"] == "nonpositive_prices"
