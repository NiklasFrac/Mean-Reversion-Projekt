# src/tests/analysis/test_returns_core.py
import numpy as np
import pandas as pd
import pytest

from analysis.numerics import compute_log_returns


@pytest.mark.unit
def test_log_returns_basic_invariants():
    idx = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    df = pd.DataFrame(
        {"A": np.linspace(100, 110, 10), "B": np.linspace(50, 55, 10)}, index=idx
    )
    r = compute_log_returns(
        df, min_positive_frac=0.9, max_nan_frac_cols=0.0, drop_policy_rows="any"
    )
    # diff() verliert 1 Zeile
    assert r.shape == (9, 2)
    assert not r.isna().any().any()


@pytest.mark.unit
def test_log_returns_filters_nonpositive_and_nan_cols():
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            # column A has a zero value -> violates min_positive_frac=1.0
            "A": [100, 100.5, 0.0, 101.0, 101.5, 102.0],
            # column B is clean
            "B": [50, 50.2, 50.4, 50.6, 50.8, 51.0],
            # column C produces NaNs after log->diff (due to NaN in raw data)
            "C": [10, 10.1, np.nan, 10.3, 10.4, 10.5],
        },
        index=idx,
    )

    r = compute_log_returns(
        df,
        min_positive_frac=1.0,  # A is dropped because of 0.0
        max_nan_frac_cols=0.0,  # C is dropped because of NaNs
        drop_policy_rows="any",
    )
    assert list(r.columns) == ["B"]
    assert r.shape[0] == 5
    assert not r.isna().any().any()
