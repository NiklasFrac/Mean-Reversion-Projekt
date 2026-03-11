import numpy as np
import pandas as pd

import analysis.data_analysis as da


def test_rolling_metrics_returns_empty_when_windows_none():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.standard_normal((30, 2)), columns=["A", "B"])
    df, cors = da.rolling_pair_metrics_fast(
        r, [("A", "B")], window=50, step=10, thr1=0.6, thr2=0.7
    )
    assert df.empty and cors == {}
