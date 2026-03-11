from __future__ import annotations

import numpy as np
import pandas as pd

from processing.fill import (
    fill_gap_with_context,
    fill_gaps_safely,
    find_nan_gaps,
)


def test_find_nan_gaps_and_fill_variants():
    s = pd.Series([1.0, np.nan, 2.0, np.nan, np.nan, 5.0, np.nan, 8.0])
    gaps = find_nan_gaps(s)
    assert gaps == [(1, 2), (3, 5), (6, 7)]  # (start, end) Slices

    # length=1 -> linear interpolate
    seg1 = fill_gap_with_context(s, 1, 2, max_gap=12)
    assert seg1 is not None and len(seg1) == 1

    # length=2 -> linear (<=2)
    seg2 = fill_gap_with_context(s, 3, 5, max_gap=12)
    assert seg2 is not None and len(seg2) == 2

    # length>max -> None
    long_s = pd.Series([1.0] + [np.nan] * 13 + [2.0])
    assert fill_gap_with_context(long_s, 1, 14, max_gap=12) is None


def test_fill_gaps_safely_drop_flag_on_long_gap():
    s = pd.Series([1.0] + [np.nan] * 20 + [2.0])
    out, removed, diag = fill_gaps_safely(s, max_gap=12)
    assert removed is True
    assert any(g["len"] > 12 for g in diag["gaps"])
