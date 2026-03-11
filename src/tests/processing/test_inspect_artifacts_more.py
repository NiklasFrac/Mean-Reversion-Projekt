from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from processing.scripts import inspect_artifacts as ins


def _mk_df(rows: int, cols: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    data = {f"C{i}": range(1, rows + 1) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


def test_main_verdict_processing_issue_big_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    data_dir = tmp_path / "runs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 60 symbols in universe
    raw = data_dir / "raw_prices.pkl"
    _mk_df(rows=10, cols=60).to_pickle(raw)

    # Only 5 symbols in processing output
    exec_path = data_dir / "filled_prices_exec.pkl"
    _mk_df(rows=10, cols=5).to_pickle(exec_path)

    (data_dir / "filled.diag.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "processing": {"kept": 5, "removed": 55, "grid_mode": "leader"},
                "filling": {"causal_only": True, "hard_drop": True},
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "filled_manifest.json").write_text(
        json.dumps({"inputs": {"raw_prices": {"path": str(raw)}}}),
        encoding="utf-8",
    )

    rc = ins.main(["--data-dir", str(data_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "LIKELY_PROCESSING_ISSUE" in out
    assert "Starker Drop" in out


def test_main_verdict_no_clear_fault(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    data_dir = tmp_path / "runs" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = data_dir / "raw_prices.pkl"
    _mk_df(rows=8, cols=30).to_pickle(raw)

    exec_path = data_dir / "filled_prices_exec.pkl"
    _mk_df(rows=8, cols=25).to_pickle(exec_path)

    rc = ins.main(["--data-dir", str(data_dir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NO_CLEAR_FAULT" in out
