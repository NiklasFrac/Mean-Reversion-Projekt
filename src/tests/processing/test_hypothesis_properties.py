from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from processing.fill import find_nan_gaps
from processing.outliers import safe_log
from processing.quality import longest_nan_run


@settings(max_examples=70, deadline=None)
@given(st.lists(st.booleans(), min_size=0, max_size=120))
def test_find_nan_gaps_reconstructs_mask(bits):
    s = pd.Series([np.nan if b else i for i, b in enumerate(bits)], dtype="float64")
    gaps = find_nan_gaps(s)
    # Rekonstruiere Maske aus Gaps
    mask = np.zeros(len(bits), dtype=bool)
    for a, b in gaps:
        mask[a:b] = True
    assert (mask == pd.isna(s).to_numpy()).all()


@settings(max_examples=70, deadline=None)
@given(st.lists(st.booleans(), min_size=0, max_size=200))
def test_longest_nan_run_matches_truth(bits):
    s = pd.Series([np.nan if b else 0.0 for b in bits], dtype="float64")
    got = longest_nan_run(s)
    # ground truth
    m = 0
    cur = 0
    for b in bits:
        if b:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    assert got == m


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        min_size=1,
        max_size=80,
    )
)
def test_safe_log_properties_series(xs):
    s = pd.Series(xs, dtype="float64")
    out = safe_log(s)
    assert isinstance(out, pd.Series)
    # positive & endlich -> endlich im Log
    pos_mask = (s > 0) & np.isfinite(s.to_numpy())
    assert np.isfinite(out[pos_mask].to_numpy()).all()
    # sonst -> NaN
    neg_or_bad = (~pos_mask).to_numpy()
    if neg_or_bad.any():
        assert pd.isna(out[neg_or_bad]).all()


@settings(max_examples=20, deadline=None)
@given(
    st.lists(
        st.lists(
            st.floats(allow_nan=True, allow_infinity=True, width=64),
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=6,
    )
)
def test_safe_log_properties_frame(xss):
    df = pd.DataFrame(xss, dtype="float64").T  # columns = series
    out = safe_log(df)
    assert isinstance(out, pd.DataFrame) and out.shape == df.shape
    # all positive and finite entries remain finite
    mask = (df > 0) & np.isfinite(df.to_numpy())
    finite_after = np.isfinite(out.to_numpy()[mask.to_numpy()])
    assert finite_after.all() if finite_after.size else True
