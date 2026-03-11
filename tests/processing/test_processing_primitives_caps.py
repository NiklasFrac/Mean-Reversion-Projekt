from __future__ import annotations

import numpy as np
import pandas as pd

from processing.processing_primitives import cap_extreme_returns, detect_stale


def test_cap_extreme_returns_respects_exclude_dates():
    idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
    s = pd.Series([100.0, 100.0, 20000.0, 101.0], index=idx)
    excluded = {pd.Timestamp("2020-01-03", tz="UTC")}
    # excluded date should not be blanked out
    out = cap_extreme_returns(s, lower=-0.5, upper=1.0, exclude_dates=excluded)
    assert pd.isna(out.loc[idx[2]]) is False
    # without exclusion we cap the spike
    out2 = cap_extreme_returns(s, lower=-0.5, upper=1.0, exclude_dates=None)
    assert pd.isna(out2.loc[idx[2]])


def test_cap_extreme_returns_flags_after_nan_gap():
    idx = pd.date_range("2020-01-02", periods=4, freq="B", tz="UTC")
    s = pd.Series([100.0, np.nan, 1.0, 1.0], index=idx)
    out = cap_extreme_returns(
        s, lower=-0.5, upper=1.0, exclude_dates=None, max_gap_bars=5
    )
    assert pd.isna(out.loc[idx[2]])


def test_cap_extreme_returns_clip_mode_is_sequential() -> None:
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    s = pd.Series([100.0, 400.0, 800.0], index=idx)
    out = cap_extreme_returns(s, lower=-0.5, upper=1.0, mode="clip")
    assert np.isclose(float(out.iloc[0]), 100.0)
    assert np.isclose(float(out.iloc[1]), 200.0)
    assert np.isclose(float(out.iloc[2]), 400.0)


def test_detect_stale_volume_gate() -> None:
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0], index=idx)
    vol = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    assert detect_stale(s, k=2, min_run=2) is True
    assert detect_stale(s, k=2, min_run=2, vol=vol, vol_thresh=0.0) is False
