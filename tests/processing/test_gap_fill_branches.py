from __future__ import annotations

import numpy as np
import pandas as pd

import processing.fill as fill_mod
from processing.fill import fill_gap_with_context


def test_fill_gap_with_context_calls_fill_gap_segment(monkeypatch):
    called = {"segment": False}

    def fake_segment(
        series,
        start,
        end,
        *,
        max_gap,
        method_policy=(True, True, True),
        causal_only=False,
    ):
        called["segment"] = True
        return np.full(end - start, 42.0, dtype="float64"), "filled_stub"

    monkeypatch.setattr(fill_mod, "fill_gap_segment", fake_segment, raising=True)

    s = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, 6.0, 7.0], dtype="float64")
    seg = fill_gap_with_context(s, start=2, end=5, max_gap=12)
    assert called["segment"] is True
    assert seg is not None and seg.shape == (3,)


def test_fill_gap_with_context_calls_kalman_branch(monkeypatch):
    called = {"kalman": False}

    def fake_kalman(y, q=1e-4, r=None):
        called["kalman"] = True
        y = y.astype("float64")
        m = np.nanmean(y)
        yy = np.where(np.isfinite(y), y, m)
        return yy

    monkeypatch.setattr(fill_mod, "kalman_smoother_level", fake_kalman, raising=True)

    vals = list(range(20))
    s = pd.Series([float(v) for v in vals], dtype="float64")
    s.iloc[5:15] = np.nan  # gap length 10 -> kalman branch

    seg = fill_gap_with_context(s, start=5, end=15, max_gap=12)
    assert called["kalman"] is True
    assert seg is not None and seg.shape == (10,)
    assert np.isfinite(seg).all()
