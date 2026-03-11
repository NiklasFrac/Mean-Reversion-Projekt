# tests/test_data_processing.py

import numpy as np
import pandas as pd
import pytest
from processing.fill import (
    fill_gap_with_context,
    find_nan_gaps,
)
from processing.processing_primitives import (
    process_and_fill_prices,
)


def make_series_with_gaps(n=50, gaps=None, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(scale=0.5, size=n)) + 100.0
    s = pd.Series(base, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    if gaps:
        for start, end in gaps:
            s.iloc[start:end] = np.nan
    return s


def test_find_nan_gaps_simple():
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, 3.0])
    gaps = find_nan_gaps(s)
    assert gaps == [(1, 3), (4, 5)]


def test_fill_gap_with_context_small_gap():
    s = make_series_with_gaps(30, gaps=[(5, 7)])
    vals = fill_gap_with_context(s, 5, 7, max_gap=10)
    assert vals is not None
    assert len(vals) == 2
    assert not np.any(np.isnan(vals))


def test_fill_gap_with_context_medium_gap():
    s = make_series_with_gaps(60, gaps=[(10, 16)])  # medium gap length 6
    vals = fill_gap_with_context(s, 10, 16, max_gap=12)
    assert vals is not None
    assert len(vals) == 6
    assert not np.any(np.isnan(vals))


def test_fill_gaps_safely_and_process_many():
    # build DataFrame with 3 symbols
    s1 = make_series_with_gaps(200, gaps=[(10, 12), (50, 55)])
    s2 = make_series_with_gaps(200, gaps=[(0, 5)])  # has NaNs at start
    s3 = make_series_with_gaps(200, gaps=[(0, 50)])  # large gap -> should be removed
    df = pd.DataFrame({"A": s1, "B": s2, "C": s3})
    filled, removed, diagnostics = process_and_fill_prices(
        df, max_gap=20, keep_pct_threshold=0.7, n_jobs=1
    )
    # C should be removed (large gap)
    assert "C" in removed
    # A and B should be kept
    assert "A" not in removed
    assert "B" not in removed
    # filled is DataFrame and contains A,B
    assert set(filled.columns) >= {"A", "B"}
    # diagnostics keys present
    assert "A" in diagnostics and "C" in diagnostics


if __name__ == "__main__":
    pytest.main([__file__])
