from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from processing import pipeline
from processing.input_mode import ResolvedInputs
from processing.raw_loader import UniversePanelBundle


def test_pipeline_uses_upstream_vendor_tz_and_writes_empty_exec(
    tmp_path: Path, monkeypatch
):
    cfg = {
        "data": {"dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_window": 2,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "vendor_tz": "America/New_York",
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta={"data_policy": {"raw_index_naive_tz": "UTC"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
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

    seen_vendor_tz: list[str] = []

    def _ensure(df: pd.DataFrame, *, vendor_tz: str | None = "UTC"):
        seen_vendor_tz.append(str(vendor_tz))
        out = df.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(
            "America/New_York"
        )
        return out

    monkeypatch.setattr(pipeline, "ensure_ny_index", _ensure)
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (df.copy(), [], {}),
    )
    written_parquet_paths: list[Path] = []

    def _write_parquet(df: pd.DataFrame, path: Path, compression=None):
        written_parquet_paths.append(path)

    monkeypatch.setattr(pipeline, "atomic_write_parquet", _write_parquet)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_sync_latest_from_run", lambda *a, **k: None)
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

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert seen_vendor_tz
    assert seen_vendor_tz[0] == "UTC"
    assert any("filled_prices_exec" in p.name for p in written_parquet_paths)


def test_pipeline_vendor_tz_policy_require_match_is_rejected(
    tmp_path: Path, monkeypatch
):
    cfg = {
        "data": {"dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "vendor_tz": "America/New_York",
            "vendor_tz_policy": "require_match",
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
    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta={"data_policy": {"raw_index_naive_tz": "UTC"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    with pytest.raises(ValueError, match="vendor_tz_policy must be one of"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")


def test_pipeline_vendor_tz_policy_config_wins_strict_run_mode_allows_missing_upstream_metadata(
    tmp_path: Path, monkeypatch
):
    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(tmp_path / "out"),
            "strict_inputs": True,
        },
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "vendor_tz": "UTC",
            "vendor_tz_policy": "config_wins",
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
    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=tmp_path / "universe_manifest.json",
        universe_meta={"data_policy": {"download_interval": "1d"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("raw_loader_reached")),
    )

    with pytest.raises(RuntimeError, match="raw_loader_reached"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")


def test_pipeline_vendor_tz_policy_config_wins_uses_config_value(
    tmp_path: Path, monkeypatch
):
    cfg = {
        "data": {"dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_window": 2,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "vendor_tz": "America/New_York",
            "vendor_tz_policy": "config_wins",
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta={"data_policy": {"raw_index_naive_tz": "UTC"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
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

    seen_vendor_tz: list[str] = []

    def _ensure(df: pd.DataFrame, *, vendor_tz: str | None = "UTC"):
        seen_vendor_tz.append(str(vendor_tz))
        out = df.copy()
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(
            "America/New_York"
        )
        return out

    monkeypatch.setattr(pipeline, "ensure_ny_index", _ensure)
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (df.copy(), [], {}),
    )
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_sync_latest_from_run", lambda *a, **k: None)
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

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert seen_vendor_tz
    assert seen_vendor_tz[0] == "America/New_York"


def test_pipeline_vendor_tz_policy_rejects_invalid_value(tmp_path: Path, monkeypatch):
    cfg = {
        "data": {"dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "out")},
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "vendor_tz": "UTC",
            "vendor_tz_policy": "invalid_policy",
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
    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta={"data_policy": {"raw_index_naive_tz": "UTC"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    with pytest.raises(ValueError, match="vendor_tz_policy must be one of"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")


def test_pipeline_adv_uses_manifest_prices_unadjusted_artifact(
    tmp_path: Path, monkeypatch
):
    out_dir = tmp_path / "out"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "tickers_universe.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    idx = pd.date_range("2020-01-01", periods=3, tz="America/New_York")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx)
    unadj = pd.DataFrame({"AAA": [10.0, 20.0, 30.0]}, index=idx)
    p_unadj = (tmp_path / "external_raw_prices_unadj.pkl").resolve()
    unadj.to_pickle(p_unadj)

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "strict_inputs": True,
        },
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_mode": "dollar",
            "adv_stat": "mean",
            "adv_window": 2,
            "adv_price_source": "raw_unadjusted",
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "filling": {"causal_only": True, "hard_drop": True},
            "outliers": {"enabled": False},
            "return_caps": {"enabled": False},
            "staleness": {"enabled": False},
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=input_dir,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=input_dir / "universe_manifest.json",
        universe_meta={
            "artifacts": {"prices_unadjusted": str(p_unadj)},
            "data_policy": {"raw_index_naive_tz": "UTC"},
        },
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
    )
    monkeypatch.setattr(pipeline, "ensure_ny_index", lambda df, **kwargs: df.copy())
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (
            df.copy(),
            [],
            {"AAA": {"non_na_pct": 1.0, "longest_gap": 0, "outliers_flagged": 0}},
        ),
    )

    captured_adv_prices: list[pd.DataFrame] = []
    liq_mod = types.ModuleType("processing.liquidity")

    def _build_adv(price_df, volume_df, window=21, adv_mode="shares", stat="mean"):
        captured_adv_prices.append(price_df.copy())
        return {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}}

    def _build_adv_with_gates(price_df, volume_df, window=21, **kwargs):
        captured_adv_prices.append(price_df.copy())
        return (
            {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}},
            {
                "AAA": {
                    "total_windows": 2,
                    "valid_windows": 2,
                    "invalid_windows": 0,
                    "valid_window_ratio": 1.0,
                    "invalid_window_ratio": 0.0,
                    "gate_pass": True,
                }
            },
        )

    liq_mod.build_adv_map_from_price_and_volume = _build_adv  # type: ignore[attr-defined]
    liq_mod.build_adv_map_with_gates = _build_adv_with_gates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.liquidity", liq_mod)
    processing_pkg = sys.modules.get("processing")
    if processing_pkg is not None:
        monkeypatch.setattr(processing_pkg, "liquidity", liq_mod, raising=False)

    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_sync_latest_from_run", lambda *a, **k: None)
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

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert captured_adv_prices
    assert float(captured_adv_prices[0].iloc[0, 0]) == 10.0


def test_pipeline_adv_run_pinned_prefers_input_dir_unadjusted_over_workspace_relative_manifest(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "out"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "tickers_universe.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    idx = pd.date_range("2020-01-01", periods=3, tz="America/New_York")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx)

    # Pinned unadjusted should win.
    pinned_unadj = pd.DataFrame({"AAA": [10.0, 20.0, 30.0]}, index=idx)
    p_pinned_unadj = input_dir / "raw_prices_unadj.pkl"
    pinned_unadj.to_pickle(p_pinned_unadj)

    # Conflicting mutable workspace path referenced from manifest (relative path).
    mutable_dir = tmp_path / "runs" / "data"
    mutable_dir.mkdir(parents=True, exist_ok=True)
    mutable_unadj = pd.DataFrame({"AAA": [1000.0, 1001.0, 1002.0]}, index=idx)
    mutable_unadj.to_pickle(mutable_dir / "raw_prices_unadj.pkl")

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "strict_inputs": True,
        },
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_mode": "dollar",
            "adv_stat": "mean",
            "adv_window": 2,
            "adv_price_source": "raw_unadjusted",
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "filling": {"causal_only": True, "hard_drop": True},
            "outliers": {"enabled": False},
            "return_caps": {"enabled": False},
            "staleness": {"enabled": False},
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=input_dir,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=input_dir / "universe_manifest.json",
        universe_meta={
            "artifacts": {"prices_unadjusted": "runs/data/raw_prices_unadj.pkl"},
            "data_policy": {"raw_index_naive_tz": "UTC"},
        },
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
    )
    monkeypatch.setattr(pipeline, "ensure_ny_index", lambda df, **kwargs: df.copy())
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (
            df.copy(),
            [],
            {"AAA": {"non_na_pct": 1.0, "longest_gap": 0, "outliers_flagged": 0}},
        ),
    )

    captured_adv_prices: list[pd.DataFrame] = []
    liq_mod = types.ModuleType("processing.liquidity")

    def _build_adv(price_df, volume_df, window=21, adv_mode="shares", stat="mean"):
        captured_adv_prices.append(price_df.copy())
        return {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}}

    def _build_adv_with_gates(price_df, volume_df, window=21, **kwargs):
        captured_adv_prices.append(price_df.copy())
        return (
            {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}},
            {
                "AAA": {
                    "total_windows": 2,
                    "valid_windows": 2,
                    "invalid_windows": 0,
                    "valid_window_ratio": 1.0,
                    "invalid_window_ratio": 0.0,
                    "gate_pass": True,
                }
            },
        )

    liq_mod.build_adv_map_from_price_and_volume = _build_adv  # type: ignore[attr-defined]
    liq_mod.build_adv_map_with_gates = _build_adv_with_gates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.liquidity", liq_mod)
    processing_pkg = sys.modules.get("processing")
    if processing_pkg is not None:
        monkeypatch.setattr(processing_pkg, "liquidity", liq_mod, raising=False)

    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_sync_latest_from_run", lambda *a, **k: None)
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

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert captured_adv_prices
    # Must come from pinned input_dir/raw_prices_unadj.pkl, not mutable runs/data/raw_prices_unadj.pkl
    assert float(captured_adv_prices[0].iloc[0, 0]) == 10.0


def test_pipeline_adv_run_pinned_manifest_basename_fallback_for_custom_unadjusted(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "out"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "tickers_universe.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    idx = pd.date_range("2020-01-01", periods=3, tz="America/New_York")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx)

    # Only custom unadjusted file exists inside pinned outputs dir (no raw_prices_unadj.pkl).
    pinned_unadj = pd.DataFrame({"AAA": [10.0, 20.0, 30.0]}, index=idx)
    p_pinned_custom_unadj = input_dir / "custom_unadj_prices.pkl"
    pinned_unadj.to_pickle(p_pinned_custom_unadj)

    # Conflicting mutable workspace path referenced from manifest (relative path).
    mutable_dir = tmp_path / "runs" / "data"
    mutable_dir.mkdir(parents=True, exist_ok=True)
    mutable_unadj = pd.DataFrame({"AAA": [1000.0, 1001.0, 1002.0]}, index=idx)
    mutable_unadj.to_pickle(mutable_dir / "custom_unadj_prices.pkl")

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "strict_inputs": True,
        },
        "data_processing": {
            "n_jobs": 1,
            "max_gap_bars": 3,
            "adv_mode": "dollar",
            "adv_stat": "mean",
            "adv_window": 2,
            "adv_price_source": "raw_unadjusted",
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "filling": {"causal_only": True, "hard_drop": True},
            "outliers": {"enabled": False},
            "return_caps": {"enabled": False},
            "staleness": {"enabled": False},
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=input_dir,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=input_dir / "universe_manifest.json",
        universe_meta={
            "artifacts": {"prices_unadjusted": "runs/data/custom_unadj_prices.pkl"},
            "data_policy": {"raw_index_naive_tz": "UTC"},
        },
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
    )
    monkeypatch.setattr(pipeline, "ensure_ny_index", lambda df, **kwargs: df.copy())
    monkeypatch.setattr(
        pipeline,
        "build_tradable_mask",
        lambda idx, **kwargs: pd.Series(True, index=idx),
    )
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(
        pipeline,
        "process_and_fill_prices",
        lambda df, **kwargs: (
            df.copy(),
            [],
            {"AAA": {"non_na_pct": 1.0, "longest_gap": 0, "outliers_flagged": 0}},
        ),
    )

    captured_adv_prices: list[pd.DataFrame] = []
    liq_mod = types.ModuleType("processing.liquidity")

    def _build_adv(price_df, volume_df, window=21, adv_mode="shares", stat="mean"):
        captured_adv_prices.append(price_df.copy())
        return {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}}

    def _build_adv_with_gates(price_df, volume_df, window=21, **kwargs):
        captured_adv_prices.append(price_df.copy())
        return (
            {"AAA": {"adv": 1.0, "last_price": float(price_df.iloc[-1, 0])}},
            {
                "AAA": {
                    "total_windows": 2,
                    "valid_windows": 2,
                    "invalid_windows": 0,
                    "valid_window_ratio": 1.0,
                    "invalid_window_ratio": 0.0,
                    "gate_pass": True,
                }
            },
        )

    liq_mod.build_adv_map_from_price_and_volume = _build_adv  # type: ignore[attr-defined]
    liq_mod.build_adv_map_with_gates = _build_adv_with_gates  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.liquidity", liq_mod)
    processing_pkg = sys.modules.get("processing")
    if processing_pkg is not None:
        monkeypatch.setattr(processing_pkg, "liquidity", liq_mod, raising=False)

    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_sync_latest_from_run", lambda *a, **k: None)
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

    pipeline.main(cfg_path=tmp_path / "cfg.yaml")

    assert captured_adv_prices
    # Must come from pinned input_dir/custom_unadj_prices.pkl via basename fallback,
    # not mutable runs/data/custom_unadj_prices.pkl.
    assert float(captured_adv_prices[0].iloc[0, 0]) == 10.0


def test_pipeline_strict_symbol_contract_raises_on_mismatch(
    tmp_path: Path, monkeypatch
):
    input_dir = tmp_path / "inputs"
    out_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "tickers_universe.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "strict_inputs": True,
        },
        "data_processing": {"n_jobs": 1, "max_gap_bars": 3},
        "mlflow": {"enabled": False},
    }
    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda *_, **kw: (
            (cfg, tmp_path / "cfg.yaml") if kw.get("return_source") else cfg
        ),
    )

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=input_dir,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta=None,
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "EXTRA": [4.0, 5.0, 6.0]}, index=idx)
    volume = pd.DataFrame(
        {"AAA": [10.0, 11.0, 12.0], "EXTRA": [13.0, 14.0, 15.0]}, index=idx
    )
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices,
            volume,
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
    )

    with pytest.raises(RuntimeError, match="Symbol contract mismatch"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")


def test_pipeline_strict_symbol_contract_requires_tickers_file(
    tmp_path: Path, monkeypatch
):
    input_dir = tmp_path / "inputs"
    out_dir = tmp_path / "out"
    input_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(out_dir),
            "strict_inputs": True,
        },
        "data_processing": {"n_jobs": 1, "max_gap_bars": 3},
        "mlflow": {"enabled": False},
    }
    monkeypatch.setattr(
        pipeline,
        "load_config",
        lambda *_, **kw: (
            (cfg, tmp_path / "cfg.yaml") if kw.get("return_source") else cfg
        ),
    )

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=input_dir,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta=None,
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices,
            volume,
            UniversePanelBundle(),
            {"prices": None, "volume": None, "panel": None},
        ),
    )

    with pytest.raises(FileNotFoundError, match="strict_inputs=true requires"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml")
