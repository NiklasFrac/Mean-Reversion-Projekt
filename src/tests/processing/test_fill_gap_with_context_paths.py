from __future__ import annotations

import numpy as np
import pandas as pd

import processing.fill as fill_mod
from processing.fill import fill_gap_with_context


def _series_with_gap(n=40, start_gap=10, end_gap=12):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    s = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
    s.iloc[start_gap:end_gap] = np.nan
    return s, start_gap, end_gap


def test_fill_gap_linear_path(monkeypatch):
    # length=2 -> linear path in fill_gap_segment
    s, a, b = _series_with_gap(n=20, start_gap=5, end_gap=7)
    vals = fill_gap_with_context(s, a, b, max_gap=12)
    assert isinstance(vals, np.ndarray) and vals.size == (b - a)
    assert np.isfinite(vals).all()


def test_fill_gap_with_context_delegates_to_fill_gap_segment(monkeypatch):
    s, a, b = _series_with_gap(n=25, start_gap=8, end_gap=12)
    sentinel = np.array([111.0, 222.0, 333.0, 444.0], dtype=float)

    def _stub(*args, **kwargs):
        return sentinel, "filled_stub"

    monkeypatch.setattr(fill_mod, "fill_gap_segment", _stub, raising=True)
    vals = fill_gap_with_context(s, a, b, max_gap=12)
    assert np.allclose(vals, sentinel)


def test_fill_gap_kalman_path_stubbed(monkeypatch):
    # length=10 -> kalman path in fill_gap_segment
    s, a, b = _series_with_gap(n=60, start_gap=20, end_gap=30)
    called = {"ok": False}

    def _smooth(y, q=1e-4, r=None):
        called["ok"] = True
        out = np.linspace(
            float(np.nanmean(y[np.isfinite(y)]) or 150.0),
            float(np.nanmean(y[np.isfinite(y)]) or 150.0) + 1.0,
            num=y.shape[0],
        )
        return out

    monkeypatch.setattr(fill_mod, "kalman_smoother_level", _smooth, raising=True)
    vals = fill_gap_with_context(s, a, b, max_gap=12)
    assert called["ok"] is True
    assert (
        isinstance(vals, np.ndarray)
        and vals.size == (b - a)
        and np.isfinite(vals).all()
    )
