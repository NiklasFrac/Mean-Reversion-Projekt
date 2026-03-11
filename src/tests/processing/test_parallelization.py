from __future__ import annotations

import numpy as np
import pandas as pd


from processing.processing_primitives import process_and_fill_prices


def test_process_and_fill_prices_parallel_equals_sequential(tmp_path):
    """
    Prüft, dass n_jobs>1 denselben Output liefert wie n_jobs=1.
    (Kleine Datenmenge -> sollte flott sein.)
    """
    idx = pd.date_range("2020-01-01", periods=20, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": pd.Series(np.linspace(100, 119, len(idx)), index=idx, dtype="float64"),
            "B": pd.Series(np.linspace(200, 219, len(idx)), index=idx, dtype="float64"),
            "C": pd.Series(np.linspace(300, 319, len(idx)), index=idx, dtype="float64"),
        }
    )

    out1, rem1, diag1 = process_and_fill_prices(prices, n_jobs=1, grid_mode="leader")
    out2, rem2, diag2 = process_and_fill_prices(prices, n_jobs=2, grid_mode="leader")

    pd.testing.assert_frame_equal(out1, out2)
    assert rem1 == rem2
    assert set(diag1.keys()) == set(diag2.keys())
