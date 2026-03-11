import numpy as np
import pandas as pd

import analysis.data_analysis as da


class FakeLW:
    def fit(self, X):
        # Liefere absichtlich nicht-PD Kovarianz (eine nicht-positive Eigenvalue)
        C = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)  # eigenvalues: 3, -1
        self.covariance_ = C
        return self


def test_compute_shrink_corr_triggers_pd_fix(monkeypatch):
    # Monkeypatch LedoitWolf mit Fake
    monkeypatch.setattr(da, "LedoitWolf", lambda: FakeLW())
    df = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4], "B": [2.0, 2.2, 2.4, 2.6, 2.8]})
    # Log-Returns unnötig, Funktion erwartet returns, aber Werte sind egal
    corr = da.compute_shrink_corr(df, fix_pd=True)
    assert corr.shape == (2, 2)
    # Korreliationen müssen [-1,1] und symmetrisch sein
    assert np.allclose(corr.values, corr.values.T, atol=1e-8)
    assert (corr.values <= 1.0 + 1e-12).all() and (corr.values >= -1.0 - 1e-12).all()
