from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from processing.io_utils import read_tickers_file, write_csv


def test_read_tickers_file_csv_symbol_and_generic(tmp_path: Path):
    p1 = tmp_path / "t1.csv"
    df1 = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "aapl"]})
    df1.to_csv(p1, index=False)
    assert read_tickers_file(p1) == ["AAPL", "MSFT"]

    p2 = tmp_path / "t2.csv"
    df2 = pd.DataFrame({"anything": [" tsla ", " TSLA ", "AMD"]})
    df2.to_csv(p2, index=False)
    assert read_tickers_file(p2) == ["TSLA", "AMD"]


def test_read_tickers_file_plaintext_fallback_and_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Force fallback: pandas.read_csv intentionally raises an exception
    import processing.io_utils as io_utils

    def _boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("bad csv")

    monkeypatch.setattr(io_utils.pd, "read_csv", _boom)

    p = tmp_path / "tickers.txt"
    p.write_text("Apple AAPL\nMeta META\nAAPL\n", encoding="utf-8")

    # Fallback reads lines, takes the last token, and deduplicates stably
    out = read_tickers_file(p)
    assert out == ["AAPL", "META"]


def test_read_tickers_file_not_found(tmp_path: Path):
    p = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        _ = read_tickers_file(p)


def test_write_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "out.csv"
    rows = [(1, "a", 1.5), (2, "b", 2.5)]
    write_csv(p, rows, header=["id", "name", "val"])
    assert p.exists()

    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["id", "name", "val"]
        data = list(r)
        assert data == [["1", "a", "1.5"], ["2", "b", "2.5"]]
