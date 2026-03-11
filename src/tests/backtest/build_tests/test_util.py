import numpy as np
import pandas as pd
from backtest.utils.alpha import compute_spread_zscore


def test_compute_spread_zscore_basic():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    x = pd.Series(np.linspace(1, 2, 100) + np.random.normal(0, 0.01, 100), index=dates)
    y = 2.0 * x + np.random.normal(0, 0.01, 100)
    spread, z, beta = compute_spread_zscore(y, x, cfg={"z_window": 20})
    assert len(spread) == 100
    assert not beta.isna().all()
