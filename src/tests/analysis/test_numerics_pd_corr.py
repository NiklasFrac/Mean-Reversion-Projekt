import numpy as np
import pandas as pd
import pytest

from analysis.numerics import _nearest_positive_definite, compute_shrink_corr


@pytest.mark.unit
def test_nearest_positive_definite_makes_pd():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # hat Eigenwerte 3 und -1 (nicht PD)
    A_pd = _nearest_positive_definite(A, eps=1e-8)
    vals = np.linalg.eigvalsh(A_pd)
    assert np.all(vals > 0), f"min eig={vals.min()}"


@pytest.mark.unit
def test_compute_shrink_corr_handles_degenerate_returns_via_pd_fix():
    # vollständig konstante Returns → Kovarianz=0 → PD-Fix-Pfad wird genutzt
    r = pd.DataFrame(np.zeros((50, 3)), columns=list("ABC"))
    C = compute_shrink_corr(r)
    assert C.shape == (3, 3)
    assert np.isfinite(C.values).all()
    # symmetrisch und binnen [-1,1]
    assert np.allclose(C, C.T)
    assert (C.values <= 1).all() and (C.values >= -1).all()
