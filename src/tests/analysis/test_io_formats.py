# src/tests/analysis/test_io_formats.py
from pathlib import Path

import pandas as pd

from analysis.preprocess import load_filled_data


def _make_df():
    idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
    return pd.DataFrame({"A": [1.0, 1.1, 1.2], "B": [2.0, 2.1, 2.2]}, index=idx)


def test_load_csv(tmp_path: Path):
    p = tmp_path / "p.csv"
    _make_df().to_csv(p)
    df = load_filled_data(p)
    assert df.shape == (3, 2)


def test_load_parquet(tmp_path: Path):
    p = tmp_path / "p.parquet"
    _make_df().to_parquet(p)
    df = load_filled_data(p)
    assert df.shape == (3, 2)


def test_load_feather(tmp_path: Path):
    p = tmp_path / "p.feather"
    _make_df().reset_index().to_feather(p)  # index explicitly as a column
    df = load_filled_data(p)
    assert df.shape == (3, 2)
