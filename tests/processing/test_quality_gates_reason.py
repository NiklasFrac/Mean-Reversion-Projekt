from __future__ import annotations

import numpy as np
import pandas as pd


from processing.quality import quality_gates


def test_quality_gates_insufficient_coverage():
    s = pd.Series([1.0, np.nan, np.nan, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, 4.0])
    ok, diag = quality_gates(s, ref_len=len(s), non_na_min_pct=0.7, max_gap=12)
    assert ok is False and diag.get("reason") == "insufficient_coverage"


def test_quality_gates_large_gap_wins():
    s = pd.Series([1.0, 1.0] + [np.nan] * 6 + [1.0, 1.0])
    ok, diag = quality_gates(s, ref_len=len(s), non_na_min_pct=0.1, max_gap=3)
    assert ok is False and diag.get("reason") == "large_gap"


def test_quality_gates_excess_leading_and_trailing_na():
    # leading
    s1 = pd.Series([np.nan] * 6 + [1.0] * 6)
    ok1, d1 = quality_gates(
        s1, ref_len=len(s1), non_na_min_pct=0.5, max_gap=12, max_start_na=5
    )
    assert ok1 is False and d1.get("reason") == "excess_leading_na"

    # trailing
    s2 = pd.Series([1.0] * 6 + [np.nan] * 4)
    ok2, d2 = quality_gates(
        s2, ref_len=len(s2), non_na_min_pct=0.5, max_gap=12, max_end_na=3
    )
    assert ok2 is False and d2.get("reason") == "excess_trailing_na"


def test_quality_gates_nonpositive_prices_rejected():
    s = pd.Series([1.0, 2.0, 3.0, -1.0, 5.0, 6.0])
    ok, d = quality_gates(
        s, ref_len=len(s), non_na_min_pct=0.9, max_gap=12, forbid_nonpositive=True
    )
    assert ok is False and d.get("reason") == "nonpositive_prices"
