# src/tests/processing/test_inspect_artifacts_extra.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from processing.scripts import inspect_artifacts as ins


def _df(rows: int = 3, cols: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    data = {f"C{i}": range(1, rows + 1) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


def test__latest_returns_none_on_empty_dir(tmp_path: Path):
    # Kein Match -> None
    got = ins._latest(str(tmp_path / "nothing*.pkl"))
    assert got is None


def test__read_df_any_csv_branch(tmp_path: Path):
    # Deckt die CSV-Branch von _read_df_any (index_col=0, parse_dates=True)
    p = tmp_path / "x.csv"
    df = _df(rows=4, cols=2)
    df.to_csv(p, index=True)
    got = ins._read_df_any(p)
    assert isinstance(got, pd.DataFrame)
    assert got.shape == (4, 2)
    assert got.index.tz is not None  # parse_dates=True behält UTC


def test_load_universe_paths_manifest_variants(tmp_path: Path):
    d = tmp_path / "runs" / "data"
    d.mkdir(parents=True, exist_ok=True)

    # 1) Manifest mit "artifacts.raw_prices"
    raw = d / "raw_prices.pkl"
    _df().to_pickle(raw)
    manifest1 = {"artifacts": {"raw_prices": str(raw)}}
    (d / "universe_manifest.json").write_text(json.dumps(manifest1), encoding="utf-8")
    u1 = ins.load_universe_paths(d)
    assert Path(u1["raw_prices"]) == raw

    # 2) Manifest mit "inputs.raw_prices.path"
    manifest2 = {"inputs": {"raw_prices": {"path": str(raw)}}}
    (d / "universe_manifest.json").write_text(json.dumps(manifest2), encoding="utf-8")
    u2 = ins.load_universe_paths(d)
    assert Path(u2["raw_prices"]) == raw

    # 3) Kein Manifest -> Fallback via latest()
    (d / "universe_manifest.json").unlink()
    # ensure raw_prices.pkl exists; latest() soll das nehmen
    u3 = ins.load_universe_paths(d)
    assert Path(u3["raw_prices"]) == raw


def test_main_verdict_likely_processing_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """
    Deckt main()-Branch:
      - Universe vorhanden (raw_prices.*),
      - Processing fehlt -> LIKELY_PROCESSING_ISSUE
    """
    data_dir = tmp_path / "runs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = data_dir / "raw_prices.pkl"
    _df(rows=5, cols=4).to_pickle(raw)  # nur Universe, kein filled_*

    rc = ins.main(["--data-dir", str(data_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "LIKELY_PROCESSING_ISSUE" in out


def test_main_verdict_likely_universe_issue_small(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """
    Deckt main()-Branch:
      - Sehr wenige Universe-Spalten (<=10) und filled_cols <= raw_cols -> LIKELY_UNIVERSE_ISSUE
    """
    data_dir = tmp_path / "runs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = data_dir / "raw_prices.pkl"
    filled = data_dir / "filled_prices_exec.pkl"

    _df(rows=5, cols=3).to_pickle(raw)  # 3 Ticker → "sehr wenig"
    _df(rows=5, cols=2).to_pickle(filled)  # <= raw_cols

    rc = ins.main(["--data-dir", str(data_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "LIKELY_UNIVERSE_ISSUE" in out
