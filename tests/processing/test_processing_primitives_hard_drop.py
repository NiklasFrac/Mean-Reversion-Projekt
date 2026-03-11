from __future__ import annotations

import pandas as pd

from processing.processing_primitives import _process_symbol


def test_process_symbol_warns_on_large_gap_when_soft_drop():
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    series = pd.Series(
        [1.0, float("nan"), float("nan"), float("nan"), 2.0], index=idx, dtype="float64"
    )

    res = _process_symbol(
        symbol="AAA",
        series=series,
        ref_index=idx,
        max_gap=1,  # gap length=3 should trigger warn_large_gap
        keep_pct_threshold=0.2,
        max_start_na=5,
        max_end_na=5,
        outlier_cfg={},
        hard_drop=False,
    )

    assert res.kept is True
    assert res.diagnostics.get("warn_large_gap") is None
    assert any(m["method"] == "drop" for m in res.diagnostics["filling"]["methods"])


def test_process_symbol_uses_tradable_mask_for_coverage():
    idx = pd.date_range("2020-01-01 09:30", periods=3, freq="B", tz="UTC")
    # Only last timestamp is tradable for this symbol; price exists there -> should be kept.
    series = pd.Series([float("nan"), float("nan"), 10.0], index=idx, dtype="float64")
    tradable = pd.Series([False, False, True], index=idx)

    res = _process_symbol(
        symbol="IPO",
        series=series,
        ref_index=idx,
        max_gap=5,
        keep_pct_threshold=0.5,
        max_start_na=5,
        max_end_na=5,
        outlier_cfg={},
        tradable_mask=tradable,
    )

    assert res.kept is True
    assert res.series is not None
    assert res.series.loc[idx[-1]] == 10.0


def test_process_symbol_rejects_nonpositive_before_capping(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    series = pd.Series([1.0, 0.0, 2.0], index=idx, dtype="float64")

    def boom(*args, **kwargs):
        raise RuntimeError(
            "cap_extreme_returns should not be invoked for nonpositive rejection"
        )

    # Ensure the cap function would blow up if called; fill-only path should not invoke it.
    monkeypatch.setattr("processing.processing_primitives.cap_extreme_returns", boom)

    res = _process_symbol(
        symbol="ZERO",
        series=series,
        ref_index=idx,
        max_gap=5,
        keep_pct_threshold=0.1,
        max_start_na=5,
        max_end_na=5,
        outlier_cfg={},
    )

    assert res.kept is True
    assert res.diagnostics.get("reason") is None


def test_process_symbol_staleness_uses_tradable_window_only():
    idx = pd.date_range("2025-01-01", periods=10, tz="UTC")
    # Full series has long flat non-tradable blocks that would trigger stale detection,
    # but the tradable sub-series is strictly increasing and should be kept.
    series = pd.Series(
        [100.0, 100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 102.0],
        index=idx,
        dtype="float64",
    )
    tradable = pd.Series(
        [True, False, False, False, False, True, False, False, False, True], index=idx
    )

    res = _process_symbol(
        symbol="AAA",
        series=series,
        ref_index=idx,
        max_gap=10,
        keep_pct_threshold=0.1,
        max_start_na=10,
        max_end_na=10,
        outlier_cfg={},
        tradable_mask=tradable,
        caps_cfg={"enabled": False},
        staleness_cfg={"enabled": True, "k": 3, "min_run": 2, "eps": 1e-8},
        causal_only=True,
        hard_drop=True,
    )

    assert res.kept is True
    assert res.diagnostics.get("reason") is None
