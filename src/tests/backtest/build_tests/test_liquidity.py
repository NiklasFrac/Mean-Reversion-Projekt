# tests/test_liquidity.py
import numpy as np
import pandas as pd
from processing.liquidity import compute_adv


def test_compute_adv():
    dates = pd.date_range("2024-01-01", periods=30)
    vol = pd.DataFrame(
        {"A": np.full(30, 10000), "B": np.arange(30, dtype=float) * 10 + 200},
        index=dates,
    )
    adv = compute_adv(vol, window=5)
    assert adv["A"] == 10000
