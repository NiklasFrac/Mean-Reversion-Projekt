import numpy as np
import pandas as pd

import analysis.data_analysis as da


def test_compute_shrink_corr_pd_fix_is_used(monkeypatch):
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((60, 3)), columns=list("ABC"))

    # Force negative eigenvalue path to trigger PD-fix:
    monkeypatch.setattr(da.np.linalg, "eigvalsh", lambda _: np.array([-1.0]))

    called = {"fix": 0}

    def fake_near_pd(mat, eps=1e-8):
        called["fix"] += 1
        return np.eye(mat.shape[0])

    monkeypatch.setattr(da, "_nearest_positive_definite", fake_near_pd)

    C = da.compute_shrink_corr(r, fix_pd=True)
    assert called["fix"] == 1
    # Identity corr expected from our fake PD-fix:
    assert np.allclose(np.diag(C.values), 1.0)
    assert np.allclose(C.values, np.eye(C.shape[0]))
