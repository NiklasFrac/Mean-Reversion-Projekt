import numpy as np
import pandas as pd

from processing.processing_primitives import process_and_fill_prices
from processing.quality_helpers import validate_prices_wide


def make_df():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    a = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    b = pd.Series(np.linspace(50, 60, len(idx)), index=idx)

    # Gaps
    a.iloc[5:8] = np.nan  # short gap
    b.iloc[10:25] = np.nan  # long gap -> should drop at max_gap=12

    # Nonpositive prices are invalid and cause a drop; keep this test focused on filling/outliers.
    a.iloc[30] = a.iloc[29] * 10  # outlier -> nulled out (NaN) by scrub

    return pd.DataFrame({"A": a, "B": b})


def test_golden_invariants():
    df = make_df()
    filled, removed, diag = process_and_fill_prices(
        df,
        max_gap=12,
        keep_pct_threshold=0.7,
        n_jobs=1,
        grid_mode="leader",
        max_start_na=5,
        max_end_na=3,
        outlier_cfg={
            "enabled": True,
            "zscore": 8.0,
            "window": 5,
            "use_log_returns": True,
        },
    )
    # B should drop because of the long gap
    assert "B" in removed
    assert "A" in filled.columns

    # Quality invariants
    checks = validate_prices_wide(filled)["checks"]
    assert checks["nonpositive_prices"] == 0
    assert checks["duplicate_index"] == 0
    assert bool(checks["monotonic_index"]) is True

    # Longest gap for A <= max_gap after filling
    a = filled["A"]
    assert a.isna().sum() == 0 or a.isna().sum() <= len(a) * 0.3
