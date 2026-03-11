from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import processing.pipeline as pipeline
from processing.raw_loader import UniversePanelBundle


def _run_pipeline_and_load_artifacts(tmp_path: Path, monkeypatch):
    out_dir = tmp_path / "out"
    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "filled_prices_exec_path": str(out_dir / "filled_prices_exec.parquet"),
            "filled_prices_panel_exec_path": str(
                out_dir / "filled_prices_panel_exec.parquet"
            ),
            "removed_symbols_path": str(out_dir / "filled_removed.pkl"),
            "diagnostics_path": str(out_dir / "filled.diag.json"),
            "manifest_path": str(out_dir / "filled_manifest.json"),
            "adv_out_path": str(out_dir / "adv_map.pkl"),
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
            "filling": {"causal_only": True, "hard_drop": True},
        },
        "mlflow": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda *_, **kw: (cfg, cfg_path) if kw.get("return_source") else cfg,
    )

    idx = pd.date_range("2024-01-01", periods=4, tz="UTC")
    prices = pd.DataFrame({"AAA": [10.0, 10.2, 10.4, 10.6]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 103.0]}, index=idx)
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
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (
            df.copy(),
            [],
            {
                col: {"non_na_pct": 1.0, "longest_gap": 0, "outliers_flagged": 0}
                for col in df.columns
            },
        ),
        raising=True,
    )
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda index, **kwargs: pd.Series(True, index=index),
        raising=True,
    )
    monkeypatch.setattr(pipeline, "collect_runtime_context", lambda **kwargs: {})

    pipeline.main(cfg_path=cfg_path)

    diag = json.loads((out_dir / "filled.diag.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (out_dir / "filled_manifest.json").read_text(encoding="utf-8")
    )
    return diag, manifest


def test_pipeline_writes_diagnostics_schema_v3_without_legacy_duplicates(
    tmp_path: Path, monkeypatch
):
    diag, _ = _run_pipeline_and_load_artifacts(tmp_path, monkeypatch)

    assert diag["schema_version"] == 3
    assert "quality" in diag and "processing" in diag
    assert "stages" in diag and "snapshots" in diag and "events" in diag
    assert {"pre_raw", "pre_exec", "post"} <= set(diag["quality"].keys())
    assert "pre_quality_exec" not in diag
    assert "processing_exec" not in diag
    assert "pre_quality" not in diag
    assert "post_quality" not in diag
    assert "kept_count" not in diag
    assert "removed_count" not in diag
    assert "processing_agg" not in diag


def test_manifest_uses_centralized_processing_summary(tmp_path: Path, monkeypatch):
    diag, manifest = _run_pipeline_and_load_artifacts(tmp_path, monkeypatch)

    extra = manifest.get("extra") or {}
    assert extra.get("processing") == diag.get("processing")
    assert "metrics" not in extra
    assert "causal_only" not in extra
    assert "hard_drop" not in extra
    assert extra.get("stages") == diag.get("stages")
    assert extra.get("events_summary") == diag.get("events", {}).get("summary")
    filling = extra.get("filling") or {}
    assert filling.get("causal_only") is True
    assert filling.get("hard_drop") is True
