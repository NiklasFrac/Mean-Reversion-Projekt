from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from processing import pipeline
from processing.raw_loader import UniversePanelBundle


def _patch_liquidity(monkeypatch: pytest.MonkeyPatch) -> None:
    liq_mod = types.ModuleType("processing.liquidity")

    def _build_adv(price_df, volume_df, window=21, adv_mode="shares", stat="mean"):
        return {
            str(col): {"adv": 1.0, "last_price": float(price_df.iloc[-1][col])}
            for col in price_df.columns
        }

    def _build_adv_with_gates(price_df, volume_df, window=21, **kwargs):
        adv = _build_adv(price_df, volume_df, window=window)
        metrics = {
            str(col): {
                "total_windows": max(1, len(price_df.index) - int(window) + 1),
                "valid_windows": max(1, len(price_df.index) - int(window) + 1),
                "invalid_windows": 0,
                "valid_window_ratio": 1.0,
                "invalid_window_ratio": 0.0,
                "gate_pass": True,
            }
            for col in price_df.columns
        }
        return adv, metrics

    liq_mod.build_adv_map_from_price_and_volume = _build_adv  # type: ignore[attr-defined]
    liq_mod.build_adv_map_with_gates = _build_adv_with_gates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.liquidity", liq_mod)
    processing_pkg = sys.modules.get("processing")
    if processing_pkg is not None:
        monkeypatch.setattr(processing_pkg, "liquidity", liq_mod, raising=False)


def test_pipeline_applies_quality_flags_on_naive_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    idx = pd.date_range("2020-01-02", periods=2, freq="B")  # naive index, both tradable
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    flags_path = tmp_path / "flags.csv"
    pd.DataFrame({"ts": idx, "symbol": ["AAA", "AAA"], "flag": [True, False]}).to_csv(
        flags_path, index=False
    )

    cfg = {
        "data": {"dir": "data", "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "grid_mode": "calendar",
            "stage2": {
                "keep_pct_threshold": 0.0,
                "max_gap_bars": 3,
                "max_start_na": 5,
                "max_end_na": 5,
            },
            "adv_window": 2,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "quality_flags": {"enabled": True, "path": str(flags_path)},
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

    used_paths = {"prices": None, "volume": None, "panel": None}
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            used_paths,
        ),
    )

    captured: list[pd.DataFrame] = []

    def _proc(df: pd.DataFrame, **kwargs):
        captured.append(df.copy())
        return df, [], {}

    monkeypatch.setattr(pipeline, "process_and_fill_prices", _proc)
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
    _patch_liquidity(monkeypatch)

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert captured, "process_and_fill_prices was not invoked"
    first = captured[0]
    # First row should be masked out by the quality flag
    assert pd.isna(first.iloc[0]["AAA"])
    assert first.iloc[1]["AAA"] == 2.0


def test_pipeline_quality_flags_respects_configured_column_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    idx = pd.date_range("2020-01-02", periods=2, freq="B")
    prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)

    flags_path = tmp_path / "flags_custom.csv"
    pd.DataFrame(
        {"date_col": idx, "asset_col": ["AAA", "AAA"], "bad_col": [True, False]}
    ).to_csv(flags_path, index=False)

    cfg = {
        "data": {"dir": "data", "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "grid_mode": "calendar",
            "stage2": {
                "keep_pct_threshold": 0.0,
                "max_gap_bars": 3,
                "max_start_na": 5,
                "max_end_na": 5,
            },
            "adv_window": 2,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "quality_flags": {
                "enabled": True,
                "path": str(flags_path),
                "format": "long",
                "ts_col": "date_col",
                "symbol_col": "asset_col",
                "flag_col": "bad_col",
            },
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
    used_paths = {"prices": None, "volume": None, "panel": None}
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            used_paths,
        ),
    )

    captured: list[pd.DataFrame] = []

    def _proc(df: pd.DataFrame, **kwargs):
        captured.append(df.copy())
        return df, [], {}

    monkeypatch.setattr(pipeline, "process_and_fill_prices", _proc)
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
    _patch_liquidity(monkeypatch)

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert captured
    first = captured[0]
    assert pd.isna(first.iloc[0]["AAA"])
    assert first.iloc[1]["AAA"] == 2.0
