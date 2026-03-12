from pathlib import Path

import pandas as pd

from analysis.preprocess import load_filled_data


def _df():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    return pd.DataFrame(
        {"A": [100, 101, 102, 103, 104], "B": [50, 51, 52, 53, 54]}, index=idx
    )


def test_load_pkl(tmp_path: Path):
    p = tmp_path / "x.pkl"
    _df().to_pickle(p)
    got = load_filled_data(p)
    assert list(got.columns) == ["A", "B"] and got.index.tz is not None


def test_load_csv(tmp_path: Path):
    p = tmp_path / "x.csv"
    df = _df().copy()
    df.to_csv(p)
    got = load_filled_data(p)
    assert got.shape == df.shape


def test_load_parquet(tmp_path: Path):
    p = tmp_path / "x.parquet"
    _df().to_parquet(p)
    got = load_filled_data(p)
    assert got.shape[1] == 2


def test_load_feather(tmp_path: Path):
    p = tmp_path / "x.feather"
    df = _df().reset_index(names=["ts"])  # feather: index as a column
    df.to_feather(p)
    got = load_filled_data(p)
    assert got.shape[0] == 5 and got.index.tz is not None
