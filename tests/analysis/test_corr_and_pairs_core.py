# src/tests/analysis/test_corr_and_pairs_core.py
import numpy as np
import pandas as pd
import pytest

from analysis.numerics import compute_log_returns, compute_shrink_corr
from analysis.pairs import list_high_pairs_vectorized


def _make_prices(seed=42, n=300):
    rng = np.random.default_rng(seed)
    ra = rng.normal(0, 0.01, n)  # Returns von A
    rb = ra + rng.normal(0, 0.002, n)  # Returns von B ~ stark korreliert mit A
    a = 100.0 * np.exp(np.cumsum(ra))
    b = 50.0 * np.exp(np.cumsum(rb))
    c = 75.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))  # unabhängiger Referenz-Ticker
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


@pytest.mark.unit
def test_shrink_corr_invariants_and_bounds():
    px = _make_prices()
    r = compute_log_returns(px)
    C = compute_shrink_corr(r)
    # Diagonale ~1, Symmetrie, Werte im [-1, 1]
    assert np.allclose(np.diag(C.values), 1.0, atol=1e-8)
    assert np.allclose(C.values, C.values.T, atol=1e-12)
    assert (C.values <= 1.0 + 1e-12).all() and (C.values >= -1.0 - 1e-12).all()


@pytest.mark.unit
def test_list_high_pairs_detects_AB():
    px = _make_prices()
    r = compute_log_returns(px)
    C = compute_shrink_corr(r)
    pairs = list_high_pairs_vectorized(C, threshold=0.7)
    names = {f"{i}-{j}" for i, j, _ in pairs} | {f"{j}-{i}" for i, j, _ in pairs}
    assert "A-B" in names or "B-A" in names
