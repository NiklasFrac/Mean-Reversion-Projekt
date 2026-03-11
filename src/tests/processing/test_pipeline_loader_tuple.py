from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from processing import pipeline


def test_pipeline_main_accepts_three_tuple_loader(tmp_path: Path, monkeypatch):
    # Minimal config and deterministic cwd
    cfg = tmp_path / "config_processing.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'data'",
                "  out_dir: 'runs/data'",
                "data_processing:",
                "  n_jobs: 1",
                "  max_gap_bars: 3",
                "  adv_window: 2",
                "  adv_gates:",
                "    min_valid_ratio: 0.0",
                "    min_total_windows_for_adv_gate: 1",
                "    max_invalid_window_ratio: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    calls: dict[str, bool] = {}

    def _loader(*args, **kwargs):
        calls["loader_called"] = True
        used_paths = {"prices": None, "volume": None, "panel": None}
        return prices, volume, used_paths  # 3-tuple fallback

    monkeypatch.setattr(pipeline, "load_raw_prices_from_universe", _loader)
    monkeypatch.setattr(
        pipeline, "process_and_fill_prices", lambda df, **k: (df, [], {})
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": df.shape[0], "cols": df.shape[1]}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "collect_runtime_context", lambda: {})
    monkeypatch.setattr(
        pipeline,
        "make_manifest",
        lambda cfg_path, inputs, extra=None: {
            "cfg_path": str(cfg_path),
            "inputs": inputs,
            "extra": extra,
        },
    )

    # Should not raise even though loader returns 3-tuple
    pipeline.main(cfg)
    assert calls.get("loader_called") is True


def test_pipeline_bubbles_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "config_processing.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'data'",
                "  out_dir: 'runs/data'",
                "data_processing:",
                "  n_jobs: 1",
                "  max_gap_bars: 3",
                "  adv_window: 2",
                "  adv_gates:",
                "    min_valid_ratio: 0.0",
                "    min_total_windows_for_adv_gate: 1",
                "    max_invalid_window_ratio: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    def _loader(*args, **kwargs):
        used_paths = {"prices": None, "volume": None, "panel": None}
        return prices, volume, used_paths

    monkeypatch.setattr(pipeline, "load_raw_prices_from_universe", _loader)
    monkeypatch.setattr(
        pipeline, "process_and_fill_prices", lambda df, **k: (df, [], {})
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": df.shape[0], "cols": df.shape[1]}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)

    def boom(*args, **kwargs):
        raise RuntimeError("parquet_write_failed")

    monkeypatch.setattr(pipeline, "atomic_write_parquet", boom)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
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

    with pytest.raises(RuntimeError, match="parquet_write_failed"):
        pipeline.main(cfg)


def test_pipeline_run_id_uses_loaded_config_hash_when_env_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    requested_cfg = tmp_path / "missing.yaml"
    loaded_cfg = tmp_path / "env_cfg.yaml"
    loaded_cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'data'",
                "  out_dir: 'runs/data'",
                "data_processing:",
                "  n_jobs: 1",
                "  max_gap_bars: 3",
                "  adv_window: 2",
                "  adv_gates:",
                "    min_valid_ratio: 0.0",
                "    min_total_windows_for_adv_gate: 1",
                "    max_invalid_window_ratio: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("BACKTEST_CONFIG", str(loaded_cfg))
    monkeypatch.chdir(tmp_path)

    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    captured: dict[str, str] = {}

    def _loader(*args, **kwargs):
        used_paths = {"prices": None, "volume": None, "panel": None}
        return prices, volume, used_paths

    monkeypatch.setattr(pipeline, "load_raw_prices_from_universe", _loader)
    monkeypatch.setattr(
        pipeline, "process_and_fill_prices", lambda df, **k: (df, [], {})
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": df.shape[0], "cols": df.shape[1]}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "collect_runtime_context", lambda **kwargs: {})

    def _manifest(cfg_path, inputs, extra=None):
        captured["cfg_path"] = str(cfg_path)
        return {"cfg_path": str(cfg_path), "inputs": inputs, "extra": extra}

    monkeypatch.setattr(pipeline, "make_manifest", _manifest)

    pipeline.main(requested_cfg)

    by_run = tmp_path / "data" / "by_run"
    run_dirs = [p for p in by_run.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1

    expected_hash_prefix = hashlib.sha1(loaded_cfg.read_bytes()).hexdigest()[:8]
    assert run_dirs[0].name.endswith(expected_hash_prefix)
    assert captured["cfg_path"] == str(loaded_cfg.resolve())


def test_pipeline_rejects_duplicate_output_basenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "config_processing.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'data'",
                "  out_dir: 'runs/data'",
                "  filled_prices_exec_path: 'a/out.parquet'",
                "  filled_prices_panel_exec_path: 'b/out.parquet'",
                "data_processing:",
                "  n_jobs: 1",
                "  max_gap_bars: 3",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="distinct basenames"):
        pipeline.main(cfg)


def test_pipeline_main_without_cfg_anchors_runs_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = tmp_path / "runs" / "configs" / "config_processing.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'runs/data'",
                "  out_dir: 'runs/data/processed'",
                "data_processing:",
                "  n_jobs: 1",
                "  max_gap_bars: 3",
                "  adv_window: 2",
                "  adv_gates:",
                "    min_valid_ratio: 0.0",
                "    min_total_windows_for_adv_gate: 1",
                "    max_invalid_window_ratio: 1.0",
                "mlflow:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    monkeypatch.chdir(src_dir)

    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    def _loader(*args, **kwargs):
        used_paths = {"prices": None, "volume": None, "panel": None}
        return prices, volume, used_paths

    monkeypatch.setattr(pipeline, "load_raw_prices_from_universe", _loader)
    monkeypatch.setattr(
        pipeline, "process_and_fill_prices", lambda df, **k: (df, [], {})
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": df.shape[0], "cols": df.shape[1]}},
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
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

    pipeline.main(None)

    run_root = tmp_path / "runs" / "data" / "by_run"
    assert run_root.exists()
    assert any(p.is_dir() for p in run_root.iterdir())
    assert not (src_dir / "runs" / "data" / "by_run").exists()
    assert not (tmp_path / "data" / "by_run").exists()
