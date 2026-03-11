from __future__ import annotations

import numpy as np
import pandas as pd

from processing.processing_primitives import process_and_fill_prices


def test_process_and_fill_prices_handles_duplicate_symbol_columns():
    idx = pd.bdate_range("2024-01-01", periods=6, tz="America/New_York")
    a1 = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0], index=idx)
    a2 = pd.Series([99.0, 100.0, 101.0, 102.0, np.nan, 104.0], index=idx)
    df = pd.concat([a1, a2], axis=1)
    df.columns = ["AAA", "AAA"]

    filled, removed, diagnostics = process_and_fill_prices(
        df,
        max_gap=3,
        keep_pct_threshold=0.5,
        n_jobs=1,
        grid_mode="leader",
        calendar_code="XNYS",
        max_start_na=5,
        max_end_na=5,
        outlier_cfg={"enabled": False},
        causal_only=True,
        hard_drop=True,
    )

    assert list(filled.columns) == ["AAA"]
    assert removed == []
    assert list(diagnostics.keys()) == ["AAA"]
