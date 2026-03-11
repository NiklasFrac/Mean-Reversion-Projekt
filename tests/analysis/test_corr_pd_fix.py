# src/tests/analysis/test_corr_pd_fix.py
import numpy as np
import pandas as pd

from analysis.numerics import compute_shrink_corr


def test_compute_shrink_corr_triggers_pd_fix(monkeypatch):
    # erzwinge Pfad: eigvalsh -> negatives Minimum, damit Fix greift
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda a: np.array([-1.0]), raising=True)
    # returns
    X = pd.DataFrame({"A": [0.0, 0.1, 0.2], "B": [0.0, 0.1, 0.2]})
    C = compute_shrink_corr(X, fix_pd=True)
    assert C.shape == (2, 2)
    assert np.all(np.isfinite(C.values))
