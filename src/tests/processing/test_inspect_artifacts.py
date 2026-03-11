# src/tests/processing/test_inspect_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from processing.scripts import inspect_artifacts as ins


def _mk_df(rows: int = 5, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    data = {f"T{i}": range(1, rows + 1) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


def test__read_df_any_csv_and_pickle(tmp_path: Path):
    df = _mk_df()
    p_csv = tmp_path / "x.csv"
    p_pkl = tmp_path / "x.pkl"

    # CSV (Index as first column)
    df.to_csv(p_csv)
    got_csv = ins._read_df_any(p_csv)
    pd.testing.assert_index_equal(pd.DatetimeIndex(got_csv.index), df.index)
    assert list(got_csv.columns) == list(df.columns)

    # Pickle (DataFrame)
    df.to_pickle(p_pkl)
    got_pkl = ins._read_df_any(p_pkl)
    pd.testing.assert_frame_equal(got_pkl, df)

    # Pickle (Series -> DataFrame)
    s = df["T0"]
    p_ser = tmp_path / "s.pkl"
    s.to_pickle(p_ser)
    got_ser = ins._read_df_any(p_ser)
    assert isinstance(got_ser, pd.DataFrame)
    pd.testing.assert_series_equal(got_ser.iloc[:, 0], s, check_names=False)


def test_load_universe_paths_manifest_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)

    # Fallback: one file exists
    p_raw = d / "raw_prices.pkl"
    _mk_df().to_pickle(p_raw)

    # Without manifest -> fallback picks file
    u = ins.load_universe_paths(d)
    assert u["raw_prices"] == p_raw

    # With manifest -> prefer path from manifest
    m = {"inputs": {"raw_prices": {"path": str(d / "other.pkl")}}}
    (d / "other.pkl").write_bytes(p_raw.read_bytes())
    (d / "universe_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    u2 = ins.load_universe_paths(d)
    assert u2["raw_prices"] == d / "other.pkl"


def test_load_processing_paths_prefers_standard_and_alt(tmp_path: Path):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)

    # Current standard names
    exec_path = d / "filled_prices_exec.pkl"
    _mk_df().to_pickle(exec_path)
    (d / "filled.diag.json").write_text("{}", encoding="utf-8")
    (d / "filled_manifest.json").write_text("{}", encoding="utf-8")

    pr = ins.load_processing_paths(d)
    assert pr["exec"] == exec_path
    assert pr["filled"] == exec_path
    assert pr["diag"] == d / "filled.diag.json"
    assert pr["manifest"] == d / "filled_manifest.json"

    # Fallback for alternative names still works
    for p in ("filled_prices_exec.pkl", "filled.diag.json", "filled_manifest.json"):
        (d / p).unlink()
    alt = d / "filled_foo.pkl"
    _mk_df().to_pickle(alt)

    pr2 = ins.load_processing_paths(d)
    assert pr2["exec"] == alt
    assert pr2["filled"] == alt


def test_load_processing_paths_manifest_stale_exec_path_falls_back(tmp_path: Path):
    data_dir = tmp_path / "runs" / "data"
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    real_exec = processed / "filled_prices_exec.parquet"
    _mk_df(rows=3, cols=2).to_parquet(real_exec)

    stale_exec = processed / "stale_exec.parquet"
    manifest = {
        "extra": {
            "outputs": {
                "latest": {
                    "exec": str(stale_exec),
                    "diagnostics": str(processed / "filled.diag.json"),
                    "manifest": str(processed / "filled_manifest.json"),
                }
            }
        }
    }
    (processed / "filled_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    pr = ins.load_processing_paths(data_dir)
    assert pr["exec"] == real_exec
    assert pr["filled"] == real_exec


def test_summarize_df_basic():
    df = _mk_df(rows=3, cols=2)
    s = ins.summarize_df(df, "x")
    assert s["name"] == "x"
    assert s["rows"] == 3 and s["cols"] == 2
    assert 0.0 <= s["mean_nan_pct"] <= 1.0

    s2 = ins.summarize_df(None, "y")
    assert s2 == {"name": "y", "rows": 0, "cols": 0, "mean_nan_pct": 1.0}


def test_main_end_to_end(tmp_path: Path, capsys):
    data_dir = tmp_path / "runs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Universe
    p_raw = data_dir / "raw_prices.pkl"
    _mk_df(rows=4, cols=3).to_pickle(p_raw)

    # Processing
    p_exec = data_dir / "filled_prices_exec.pkl"
    _mk_df(rows=4, cols=2).to_pickle(p_exec)

    diag = {
        "schema_version": 2,
        "processing": {"kept": 2, "removed": 1, "grid_mode": "leader"},
        "filling": {"causal_only": True, "hard_drop": False},
    }
    (data_dir / "filled.diag.json").write_text(json.dumps(diag), encoding="utf-8")

    manifest = {"inputs": {"raw_prices": {"path": str(p_raw)}}}
    (data_dir / "filled_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    rc = ins.main(["--data-dir", str(data_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "[UNIVERSE] raw_prices:" in out
    assert "[PROCESSING] exec:" in out
    assert "=== VERDICT:" in out
    assert "Reason:" in out
    assert "grid=leader" in out
    assert "fill=causal_only=True,hard_drop=False" in out
