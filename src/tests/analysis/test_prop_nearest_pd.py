import hypothesis as h
import hypothesis.strategies as st
import numpy as np

import analysis.data_analysis as da


@h.given(st.integers(min_value=2, max_value=8))
@h.settings(max_examples=50, deadline=700, derandomize=True)
def test_nearest_pd_returns_spd(n):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    M = (A + A.T) / 2.0
    pdM = da._nearest_positive_definite(M, eps=1e-8)
    # Symmetrisch
    assert np.allclose(pdM, pdM.T, atol=1e-10)
    # Eigenwerte >= eps
    eigs = np.linalg.eigvalsh(pdM)
    assert np.all(eigs >= 1e-10)
