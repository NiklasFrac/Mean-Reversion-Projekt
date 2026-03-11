import pandas as pd

from analysis.numerics import compute_log_returns


def test_compute_log_returns_no_nan(synthetic_prices_wide):
    logret = compute_log_returns(
        synthetic_prices_wide,
        min_positive_frac=0.95,
        max_nan_frac_cols=0.02,
        drop_policy_rows="any",
    )
    assert isinstance(logret.index, pd.DatetimeIndex)
    assert logret.index.tz is not None
    assert not logret.isna().any().any()
    assert logret.shape[0] > 1000 and logret.shape[1] >= 40
