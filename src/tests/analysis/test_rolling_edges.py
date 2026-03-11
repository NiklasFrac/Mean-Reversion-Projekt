# src/tests/analysis/test_rolling_edges.py
import numpy as np
import pandas as pd

from analysis.rolling import rolling_pair_metrics_fast


def test_rolling_empty_when_window_too_big():
    idx = pd.date_range("2024-01-01", periods=10)
    df = pd.DataFrame({"A": np.arange(10) + 1, "B": np.arange(10) + 2}, index=idx)
    # returns
    logret = np.log(df).diff().iloc[1:]
    out, cors = rolling_pair_metrics_fast(
        logret, [("A", "B")], window=100, step=10, thr1=0.5, thr2=0.6
    )
    assert out.empty and cors == {}


def test_rolling_filters_unknown_pairs():
    idx = pd.date_range("2024-01-01", periods=40)
    df = pd.DataFrame({"A": np.arange(40) + 1, "B": np.arange(40) + 2}, index=idx)
    logret = np.log(df).diff().iloc[1:]
    out, cors = rolling_pair_metrics_fast(
        logret, [("A", "X")], window=20, step=10, thr1=0.1, thr2=0.2
    )
    assert out.empty and cors == {}
