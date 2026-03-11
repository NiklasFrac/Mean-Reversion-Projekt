from __future__ import annotations

from pathlib import Path

import pandas as pd

from processing import pipeline
from processing.input_mode import ResolvedInputs
from processing.raw_loader import UniversePanelBundle


def _run_pipeline_with_calendar_toggle(
    *,
    tmp_path: Path,
    monkeypatch,
    enabled: bool,
    csv_path: Path,
    csv_enabled: bool | None = None,
) -> dict[str, str]:
    calendars_cfg = {
        "default": "XNYS",
        "symbol_calendar_overrides_enabled": enabled,
        "symbol_calendar_csv": str(csv_path),
    }
    if csv_enabled is not None:
        calendars_cfg["symbol_calendar_csv_enabled"] = bool(csv_enabled)

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
            "vendor_guards": {"enabled": False},
            "calendars": calendars_cfg,
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
        mode="legacy_latest",
        input_dir=tmp_path,
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta=None,
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    idx = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0]}, index=idx)
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

    seen_map: dict[str, str] = {}

    def _process(df: pd.DataFrame, **kwargs):
        nonlocal seen_map
        seen_map = dict(kwargs.get("symbol_calendar_map") or {})
        return df.copy(), [], {}

    monkeypatch.setattr(pipeline, "process_and_fill_prices", _process)
    monkeypatch.setattr(
        pipeline,
        "validate_prices_wide",
        lambda df: {"checks": {"rows": int(df.shape[0]), "cols": int(df.shape[1])}},
    )
    monkeypatch.setattr(pipeline, "collect_runtime_context", lambda **kwargs: {})
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_parquet", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "atomic_write_json", lambda *a, **k: None)
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
    return seen_map


def test_symbol_calendar_csv_is_loaded_even_when_legacy_toggle_is_false(
    tmp_path: Path, monkeypatch
):
    csv_path = tmp_path / "symbol_calendar.csv"
    pd.DataFrame({"symbol": ["AAA"], "calendar_code": ["XNAS"]}).to_csv(
        csv_path, index=False
    )

    seen_map = _run_pipeline_with_calendar_toggle(
        tmp_path=tmp_path, monkeypatch=monkeypatch, enabled=False, csv_path=csv_path
    )
    assert seen_map == {"AAA": "XNAS"}


def test_symbol_calendar_overrides_enabled_loads_csv(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "symbol_calendar.csv"
    pd.DataFrame({"symbol": ["AAA"], "calendar_code": ["XNAS"]}).to_csv(
        csv_path, index=False
    )

    seen_map = _run_pipeline_with_calendar_toggle(
        tmp_path=tmp_path, monkeypatch=monkeypatch, enabled=True, csv_path=csv_path
    )
    assert seen_map == {"AAA": "XNAS"}


def test_symbol_calendar_csv_enabled_false_skips_loading(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "symbol_calendar.csv"
    pd.DataFrame({"symbol": ["AAA"], "calendar_code": ["XNAS"]}).to_csv(
        csv_path, index=False
    )

    seen_map = _run_pipeline_with_calendar_toggle(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        enabled=True,
        csv_path=csv_path,
        csv_enabled=False,
    )
    assert seen_map == {}
