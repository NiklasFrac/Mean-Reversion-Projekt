from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processing.fill import (
    fill_gap_segment,
    fill_gaps_safely,
    kalman_smoother_level,
)


def test_kalman_smoother_handles_empty_all_nan_and_zero_var():
    assert kalman_smoother_level(np.array([])).size == 0

    all_nan = np.array([np.nan, np.nan])
    out_nan = kalman_smoother_level(all_nan)
    assert np.isnan(out_nan).all()

    zero_var = np.array([2.0, 2.0, 2.0])
    out_zero_var = kalman_smoother_level(zero_var)
    assert np.isfinite(out_zero_var).all()
    # flat input stays flat even after smoothing
    assert out_zero_var[0] == pytest.approx(out_zero_var[-1])


def test_fill_gap_segment_noop_bfill_and_causal_ffill():
    s = pd.Series([1.0, 2.0], dtype=float)
    vals, label = fill_gap_segment(s, start=1, end=1, max_gap=3)  # length <= 0
    assert vals.size == 0 and label == "filled_noop"

    only_right = pd.Series([np.nan, 5.0], dtype=float)
    vals, label = fill_gap_segment(only_right, start=0, end=1, max_gap=3)
    assert label == "filled_bfill"
    assert vals.tolist() == [5.0]

    causal_src = pd.Series([7.0, np.nan, 9.0], dtype=float)
    vals, label = fill_gap_segment(
        causal_src, start=1, end=2, max_gap=3, causal_only=True
    )
    assert label == "filled_ffill"
    assert vals.tolist() == [7.0]


def test_fill_gap_segment_linear_two_step():
    s = pd.Series([1.0, np.nan, np.nan, 4.0], dtype=float)
    vals, label = fill_gap_segment(s, start=1, end=3, max_gap=5)
    assert label == "filled_linear"
    assert np.allclose(vals, [2.0, 3.0])


def test_fill_gaps_safely_skips_nontradable_without_drop(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    series = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan], index=idx, dtype=float)
    tradable = pd.Series([True, False, False, True, True], index=idx)

    filled, removed, diag = fill_gaps_safely(
        series,
        max_gap=1,  # first gap length=2 -> drop branch; trailing gap length=1
        tradable_mask=tradable,
        hard_drop=False,
    )

    assert removed is False
    # first gap (idx 1-2) stays NaN because non-tradable + hard_drop=False
    assert filled.isna().sum() >= 2
    methods = [m["method"] for m in diag["methods"]]
    assert "skip_nontradable" in methods


def test_fill_gaps_safely_long_gap_soft_drop():
    idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
    series = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx, dtype=float)

    filled, removed, diag = fill_gaps_safely(series, max_gap=1, hard_drop=False)

    assert removed is False
    # long gap not filled because longer than max_gap
    assert filled.isna().any()
    assert any(m["method"] in {"drop", "skip_edge_unfilled"} for m in diag["methods"])


def test_fill_gaps_nontradable_no_drop_even_hard():
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    series = pd.Series([np.nan, np.nan, 5.0], index=idx, dtype=float)
    tradable = pd.Series([False, False, True], index=idx)

    filled, removed, diag = fill_gaps_safely(
        series, max_gap=1, tradable_mask=tradable, hard_drop=True
    )

    assert removed is False  # skip_nontradable must not trigger drop anymore
    assert any(m["method"] == "skip_nontradable" for m in diag["methods"])


def test_fill_gaps_keeps_edge_gaps_within_allowance():
    idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
    # Leading and trailing gaps longer than max_gap cannot be interpolated but should not force a drop.
    series = pd.Series([np.nan, np.nan, 10.0, 11.0], index=idx, dtype=float)

    filled, removed, diag = fill_gaps_safely(series, max_gap=1, hard_drop=True)

    assert removed is False
    methods = [m["method"] for m in diag["methods"]]
    # Edge gaps stay unfilled but are explicitly skipped instead of dropping the symbol.
    assert "skip_edge_unfilled" in methods
