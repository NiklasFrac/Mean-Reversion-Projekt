from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import processing.pipeline as pipeline
from processing.raw_loader import UniversePanelBundle


def test_sync_latest_outputs_transactional_rolls_back_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run1 = tmp_path / "run1.bin"
    run2 = tmp_path / "run2.bin"
    latest1 = tmp_path / "latest1.bin"
    latest2 = tmp_path / "latest2.bin"
    run1.write_bytes(b"new-1")
    run2.write_bytes(b"new-2")
    latest1.write_bytes(b"old-1")
    latest2.write_bytes(b"old-2")

    real_replace = pipeline.os.replace

    def _flaky_replace(src, dst):
        src_p = Path(src)
        dst_p = Path(dst)
        if dst_p == latest2 and ".tmp." in src_p.name:
            raise OSError("replace failed mid-commit")
        return real_replace(src, dst)

    monkeypatch.setattr(pipeline.os, "replace", _flaky_replace, raising=True)

    with pytest.raises(OSError, match="replace failed mid-commit"):
        pipeline._sync_latest_outputs_transactional([(run1, latest1), (run2, latest2)])

    # Rollback should restore both original latest files.
    assert latest1.read_bytes() == b"old-1"
    assert latest2.read_bytes() == b"old-2"


def test_pipeline_does_not_update_latest_on_run_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    latest_exec = tmp_path / "out" / "filled_prices_exec.parquet"
    latest_exec.parent.mkdir(parents=True, exist_ok=True)
    latest_exec.write_bytes(b"old-exec")

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(tmp_path / "out"),
            "filled_prices_exec_path": str(latest_exec),
            "filled_prices_panel_exec_path": str(
                tmp_path / "out" / "filled_prices_panel_exec.parquet"
            ),
            "removed_symbols_path": str(tmp_path / "out" / "filled_removed.pkl"),
            "diagnostics_path": str(tmp_path / "out" / "filled.diag.json"),
            "manifest_path": str(tmp_path / "out" / "filled_manifest.json"),
            "adv_out_path": str(tmp_path / "out" / "adv_map.pkl"),
        },
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_window": 2,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "pip_freeze": False,
        },
        "mlflow": {"enabled": False},
    }
    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda *_, **kw: (
            (cfg, tmp_path / "cfg.yaml") if kw.get("return_source") else cfg
        ),
    )

    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline, "process_and_fill_prices", lambda df, **k: (df.copy(), [], {})
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda index, **kwargs: pd.Series(True, index=index),
    )
    monkeypatch.setattr(pipeline, "collect_runtime_context", lambda **kwargs: {})
    monkeypatch.setattr(
        pipeline,
        "make_manifest",
        lambda cfg_path, inputs, extra=None: {
            "cfg_path": str(cfg_path),
            "inputs": inputs,
            "extra": extra,
        },
    )

    def _write_parquet(df, path, compression=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"new-parquet")

    def _write_pickle(obj, path):
        p = Path(path)
        if p.name == "filled_removed.pkl" and "by_run" in str(p):
            raise RuntimeError("boom_removed")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"new-pickle")

    monkeypatch.setattr(pipeline, "atomic_write_parquet", _write_parquet, raising=True)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", _write_pickle, raising=True)
    monkeypatch.setattr(
        pipeline,
        "atomic_write_json",
        lambda obj, path: Path(path).write_text("{}", encoding="utf-8"),
        raising=True,
    )

    with pytest.raises(RuntimeError, match="boom_removed"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    # Failure happened before the latest sync phase; latest must stay untouched.
    assert latest_exec.read_bytes() == b"old-exec"
