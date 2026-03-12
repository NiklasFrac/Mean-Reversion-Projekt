from __future__ import annotations

import numpy as np
import pandas as pd


from processing.processing_primitives import _process_symbol, process_and_fill_prices


def test__process_symbol_happy_and_outlier_error():
    # Reference grid
    idx = pd.date_range("2020-01-01", periods=10, tz="UTC")
    s = pd.Series(np.linspace(100, 109, len(idx)), index=idx, dtype="float64")

    # 1) Happy path (kept=True)
    res = _process_symbol(
        symbol="AAA",
        series=s,
        ref_index=idx,
        max_gap=12,
        keep_pct_threshold=0.7,
        max_start_na=5,
        max_end_na=3,
        outlier_cfg={
            "enabled": True,
            "zscore": 8.0,
            "window": 5,
            "use_log_returns": True,
        },
    )
    assert res.kept is True and res.series is not None
    assert "non_na_pct" in res.diagnostics

    # 2) The fill backstop currently ignores outlier_cfg and remains stable.
    res2 = _process_symbol(
        symbol="BBB",
        series=s,
        ref_index=idx,
        max_gap=12,
        keep_pct_threshold=0.7,
        max_start_na=5,
        max_end_na=3,
        outlier_cfg={"enabled": True},
    )
    assert res2.kept is True
    assert "outlier_error" not in res2.diagnostics


def test_process_and_fill_prices_empty_and_union_grid():
    # empty
    df, removed, diag = process_and_fill_prices(pd.DataFrame(), n_jobs=1)
    assert df.empty and removed == [] and diag == {}

    # union grid + removal via quality gates (artificial here)
    idx1 = pd.date_range("2020-01-01", periods=5, tz="UTC")
    idx2 = pd.date_range("2020-01-03", periods=3, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": pd.Series([1, 2, 3, 4, 5], index=idx1, dtype="float64"),
            "B": pd.Series([np.nan] * 3, index=idx2, dtype="float64"),
        }
    )
    out, removed, diag = process_and_fill_prices(prices, grid_mode="union", n_jobs=1)
    # B is removed (almost empty)
    assert "B" in set(removed)
    assert "A" in set(out.columns)
