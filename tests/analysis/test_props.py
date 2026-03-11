# src/tests/analysis/test_props.py
import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from analysis.bootstrap_fdr import benjamini_hochberg
from analysis.numerics import compute_shrink_corr


@given(st.integers(min_value=2, max_value=15), st.integers(min_value=50, max_value=150))
def test_corr_is_symmetric_and_bounded(cols, rows):
    X = pd.DataFrame(np.random.randn(rows, cols))
    C = compute_shrink_corr(X)
    assert np.allclose(C.values, C.values.T, atol=1e-8)
    assert np.all(C.values <= 1.0 + 1e-9)
    assert np.all(C.values >= -1.0 - 1e-9)


@given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=50))
def test_bh_returns_bools_in_same_length(pvals):
    dec = benjamini_hochberg(pvals, alpha=0.1)
    assert isinstance(dec, list) and len(dec) == len(pvals)
    assert all(isinstance(x, bool) for x in dec)
