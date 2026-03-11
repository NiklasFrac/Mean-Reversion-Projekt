import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from processing.processing_primitives import process_and_fill_prices


@st.composite
def random_series(draw):
    n = draw(st.integers(min_value=30, max_value=120))
    base = np.cumsum(np.random.randn(n)) + 100.0
    # inject NaNs
    k = draw(st.integers(min_value=0, max_value=max(0, n // 3)))
    if k > 0:
        idx = np.random.choice(n, size=k, replace=False)
        base[idx] = np.nan
    # nonpositives
    if n > 10:
        base[0] = np.nan
        base[5] = abs(base[5])  # positive
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(base, index=idx, name="X")


@given(random_series())
@settings(deadline=None)
def test_kept_series_respects_gates(s):
    df = pd.DataFrame({"X": s})
    filled, removed, diag = process_and_fill_prices(
        df,
        max_gap=12,
        keep_pct_threshold=0.6,
        n_jobs=1,
        grid_mode="leader",
        max_start_na=10,
        max_end_na=5,
        outlier_cfg={"enabled": True},
    )
    if "X" in removed:
        return
    x = filled["X"]
    assert not (x.dropna() <= 0).any()
