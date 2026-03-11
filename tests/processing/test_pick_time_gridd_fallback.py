from __future__ import annotations

import numpy as np
import pandas as pd


from processing.timebase import pick_time_grid


def test_pick_time_grid_fallback_to_leader_on_unknown_mode():
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    df = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3, 4, 5], index=idx, dtype="float64"),
            "B": pd.Series([1, np.nan, 3, np.nan, 5], index=idx, dtype="float64"),
        }
    )
    leader = pick_time_grid(df, mode="leader")
    weird = pick_time_grid(df, mode="weird")
    assert leader.equals(weird)
