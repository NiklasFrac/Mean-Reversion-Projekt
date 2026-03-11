from __future__ import annotations

import numpy as np
import pandas as pd

from processing.outliers import (
    robust_outlier_mask_causal,
    safe_log,
    scrub_outliers_causal,
)


def test_safe_log_series_and_frame():
    s = pd.Series([1.0, np.nan, -1.0, np.inf, 10.0])
    sl = safe_log(s)
    assert np.isfinite(sl.dropna()).all()
    assert sl.iloc[2] != sl.iloc[2]  # -1 -> NaN

    df = pd.DataFrame({"A": [1.0, 2.0, -3.0], "B": [np.inf, 1.0, 1.0]})
    dfl = safe_log(df)
    assert dfl["A"].isna().iloc[2]
    assert dfl["B"].isna().iloc[0]


def test_robust_outlier_mask_small_series_and_spike():
    # <5 Werte -> immer False
    small = pd.Series([1.0, 1.1, 1.2, 1.1])
    m = robust_outlier_mask_causal(small, zscore=4.0, window=3, use_log_returns=True)
    assert not m.any()

    # Deutlicher Spike, aber mit minimaler deterministischer Variation (MAD > 0)
    base = 100.0 + 0.01 * np.arange(21)
    x = pd.Series(base.copy())
    x.iloc[10] = 500.0  # großer Spike
    m2 = robust_outlier_mask_causal(x, zscore=3.0, window=5, use_log_returns=True)
    assert bool(m2.iloc[10])
    assert m2.any()


def test_scrub_outliers_replaces_values():
    base = 100.0 + 0.01 * np.arange(21)
    x = pd.Series(base.copy())
    x.iloc[10] = 500.0
    s2, n = scrub_outliers_causal(x, zscore=3.0, window=5, use_log_returns=True)
    assert n >= 1
    assert pd.isna(s2.iloc[10])
