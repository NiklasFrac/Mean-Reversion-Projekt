from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import processing.pipeline as pipeline


def _mini_prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=6, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "AAA": np.linspace(100, 101, len(idx)),
            "BBB": np.linspace(200, 201, len(idx)),
        },
        index=idx,
    )


def _mini_volume(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {c: np.full(len(prices.index), 1000.0, dtype=float) for c in prices.columns},
        index=prices.index,
    )


def _base_dp() -> dict[str, object]:
    return {
        "max_gap_bars": 3,
        "n_jobs": 1,
        "adv_window": 2,
        "adv_gates": {
            "min_valid_ratio": 0.0,
            "min_total_windows_for_adv_gate": 1,
            "max_invalid_window_ratio": 1.0,
        },
    }


def _patch_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prices = _mini_prices()
    volume = _mini_volume(prices)
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda data_dir, **_kwargs: (
            prices.copy(),
            volume.copy(),
            {"prices": str(tmp_path / "raw.pkl"), "volume": str(tmp_path / "vol.pkl")},
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline, "atomic_write_pickle", lambda *a, **k: None, raising=True
    )
    monkeypatch.setattr(
        pipeline, "atomic_write_parquet", lambda *a, **k: None, raising=True
    )
    monkeypatch.setattr(
        pipeline, "atomic_write_json", lambda *a, **k: None, raising=True
    )
    monkeypatch.setattr(
        pipeline,
        "collect_runtime_context",
        lambda **_kwargs: {
            "timestamp": "2026-01-01T00:00:00Z",
            "python": "x",
            "platform": "x",
            "git_commit": "deadbeef",
            "git_dirty": False,
            "pip_lock_path": None,
            "pip_lock_sha1": None,
            "libs": {"pandas": "x", "numpy": "x"},
            "cpu_count": 1,
        },
        raising=True,
    )


def test_main_mlflow_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_common(monkeypatch, tmp_path)

    calls = {
        "uri": None,
        "exp": None,
        "run_name": None,
        "params": None,
        "metrics": None,
        "artifacts": [],
    }

    class _RunCtxt:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ml = types.ModuleType("mlflow")
    ml.set_tracking_uri = lambda uri: calls.__setitem__("uri", uri)  # type: ignore[attr-defined]
    ml.set_experiment = lambda name: calls.__setitem__("exp", name)  # type: ignore[attr-defined]
    ml.start_run = lambda run_name=None: (
        calls.__setitem__("run_name", run_name),
        _RunCtxt(),
    )[1]  # type: ignore[attr-defined]
    ml.log_params = lambda p: calls.__setitem__("params", dict(p))  # type: ignore[attr-defined]
    ml.log_metrics = lambda m: calls.__setitem__("metrics", dict(m))  # type: ignore[attr-defined]
    ml.log_artifact = lambda p: calls["artifacts"].append(str(p))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ml)

    dp = _base_dp()
    dp.update(
        {
            "keep_pct_threshold": 0.6,
            "grid_mode": "leader",
            "calendar": "XNYS",
            "max_start_na": 5,
            "max_end_na": 3,
            "parquet_compression": "zstd",
            "outliers": {
                "enabled": True,
                "zscore": 6.0,
                "window": 7,
                "use_log_returns": True,
            },
        }
    )
    cfg = {
        "data": {
            "dir": str(tmp_path / "u"),
            "filled_prices_path": str(tmp_path / "out" / "filled.pkl"),
        },
        "data_processing": dp,
        "mlflow": {
            "enabled": True,
            "tracking_uri": "file://" + str(tmp_path / "mlruns"),
            "experiment_name": "processing_tests",
            "run_name": "processing_run_test",
        },
    }
    cfg_path = tmp_path / "configs" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    pipeline.main(cfg_path)

    assert calls["uri"]
    if os.name == "nt":
        assert str(calls["uri"]).startswith("file:")
        assert not str(calls["uri"]).startswith("file://")
    assert calls["exp"] == "processing_tests"
    assert calls["run_name"] == "processing_run_test"
    assert isinstance(calls["params"], dict) and "grid_mode" in calls["params"]
    assert isinstance(calls["metrics"], dict) and "kept" in calls["metrics"]


def test_main_mlflow_defaults_to_sqlite_tracking_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_common(monkeypatch, tmp_path)

    calls = {"uri": None}

    class _RunCtxt:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ml = types.ModuleType("mlflow")
    ml.set_tracking_uri = lambda uri: calls.__setitem__("uri", uri)  # type: ignore[attr-defined]
    ml.set_experiment = lambda *a, **k: None  # type: ignore[attr-defined]
    ml.start_run = lambda *a, **k: _RunCtxt()  # type: ignore[attr-defined]
    ml.log_params = lambda *a, **k: None  # type: ignore[attr-defined]
    ml.log_metrics = lambda *a, **k: None  # type: ignore[attr-defined]
    ml.log_artifact = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ml)

    cfg = {
        "data": {
            "dir": str(tmp_path / "u"),
            "filled_prices_path": str(tmp_path / "out" / "filled.pkl"),
        },
        "data_processing": _base_dp(),
        "mlflow": {
            "enabled": True,
            "experiment_name": "processing_tests",
            "run_name": "processing_run_test",
        },
    }
    cfg_path = tmp_path / "configs" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    pipeline.main(cfg_path)
    assert isinstance(calls["uri"], str)
    assert str(calls["uri"]).startswith("sqlite:///")


def test_main_mlflow_logs_run_scoped_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_common(monkeypatch, tmp_path)
    # This test verifies artifact logging paths, so run-scoped files must exist.
    monkeypatch.setattr(
        pipeline,
        "atomic_write_pickle",
        lambda obj, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"PKL"),
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline,
        "atomic_write_parquet",
        lambda df, path, compression=None: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(b"PAR1"),
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline,
        "atomic_write_json",
        lambda payload, path: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_text("{}", encoding="utf-8"),
        ),
        raising=True,
    )

    calls = {"artifacts": []}

    class _RunCtxt:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ml = types.ModuleType("mlflow")
    ml.set_tracking_uri = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ml.set_experiment = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ml.start_run = lambda *_a, **_k: _RunCtxt()  # type: ignore[attr-defined]
    ml.log_params = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ml.log_metrics = lambda *_a, **_k: None  # type: ignore[attr-defined]
    ml.log_artifact = lambda p: calls["artifacts"].append(str(p))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ml)

    out_dir = tmp_path / "out"
    cfg = {
        "data": {
            "dir": str(tmp_path / "u"),
            "out_dir": str(out_dir),
            "filled_prices_exec_path": str(out_dir / "filled_prices_exec.pkl"),
            "diagnostics_path": str(out_dir / "filled.diag.json"),
            "manifest_path": str(out_dir / "filled_manifest.json"),
        },
        "data_processing": _base_dp(),
        "mlflow": {
            "enabled": True,
            "tracking_uri": "http://mlflow.local:5000",
            "run_name": "processing_run_test",
        },
    }
    cfg_path = tmp_path / "configs" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    pipeline.main(cfg_path)

    artifact_paths = [Path(p) for p in calls["artifacts"]]
    assert {p.name for p in artifact_paths} == {
        "filled_prices_exec.pkl",
        "filled.diag.json",
        "filled_manifest.json",
    }
    assert all("by_run" in p.parts for p in artifact_paths)


def test_main_mlflow_defaults_experiment_and_artifacts_under_runs_mlruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _patch_common(monkeypatch, tmp_path)

    calls = {
        "uri": None,
        "exp": None,
        "created": None,
        "run_name": None,
    }

    class _RunCtxt:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ml = types.ModuleType("mlflow")
    ml.set_tracking_uri = lambda uri: calls.__setitem__("uri", uri)  # type: ignore[attr-defined]
    ml.get_experiment_by_name = lambda name: None  # type: ignore[attr-defined]
    ml.create_experiment = lambda name, artifact_location=None, tags=None: (
        calls.__setitem__(  # type: ignore[attr-defined]
            "created",
            (name, artifact_location),
        )
    )
    ml.set_experiment = lambda name: calls.__setitem__("exp", name)  # type: ignore[attr-defined]
    ml.start_run = lambda run_name=None: (
        calls.__setitem__("run_name", run_name),
        _RunCtxt(),
    )[1]  # type: ignore[attr-defined]
    ml.log_params = lambda *a, **k: None  # type: ignore[attr-defined]
    ml.log_metrics = lambda *a, **k: None  # type: ignore[attr-defined]
    ml.log_artifact = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ml)

    cfg = {
        "data": {
            "dir": "runs/data",
            "filled_prices_path": "runs/data/processed/filled.pkl",
        },
        "data_processing": _base_dp(),
        "mlflow": {
            "enabled": True,
            "tracking_uri": "sqlite:///runs/metadata/mlflow.db",
            "run_name": "processing_run_test",
        },
    }
    cfg_path = tmp_path / "runs" / "configs" / "config_processing.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    workdir = tmp_path / "src"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    pipeline.main(None)

    assert (
        calls["uri"]
        == f"sqlite:///{(tmp_path / 'runs' / 'metadata' / 'mlflow.db').as_posix()}"
    )
    assert calls["exp"] == "processing"
    assert calls["created"] == (
        "processing",
        pipeline._default_mlflow_artifact_uri(project_root=tmp_path),
    )
    assert calls["run_name"] == "processing_run_test"
