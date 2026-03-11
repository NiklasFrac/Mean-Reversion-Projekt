from __future__ import annotations

import numpy as np
import pandas as pd

from processing.processing_primitives import process_and_fill_prices
from processing.timebase import pick_time_grid


def _toy_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=20)
    aaa = pd.Series(np.linspace(100.0, 119.0, len(idx)), index=idx)
    aaa.iloc[8] = np.nan
    bbb = pd.Series(np.nan, index=idx)
    bbb.iloc[0:10] = np.linspace(50.0, 59.0, 10)
    ccc = pd.Series(np.linspace(200.0, 219.0, len(idx)), index=idx)
    ccc.iloc[12] = 2000.0
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc}, index=idx)


def test_process_and_fill_prices_fill_backstop_behavior():
    df = _toy_frame()
    ref = pick_time_grid(df, mode="leader")
    out, removed, _diag = process_and_fill_prices(
        df,
        max_gap=12,
        keep_pct_threshold=0.7,
        n_jobs=1,
        grid_mode="leader",
        calendar_code="XNYS",
        max_start_na=5,
        max_end_na=3,
        outlier_cfg={
            "enabled": True,
            "zscore": 6.0,
            "window": 11,
            "use_log_returns": True,
        },
        hard_drop=False,
    )
    # No coverage or outlier gating in fill-only backstop.
    assert removed == []
    assert list(sorted(out.columns)) == ["AAA", "BBB", "CCC"]
    assert out.index.equals(ref)
    # Outlier remains; stage-level scrub happens upstream now.
    assert float(out["CCC"].max()) >= 1000.0
