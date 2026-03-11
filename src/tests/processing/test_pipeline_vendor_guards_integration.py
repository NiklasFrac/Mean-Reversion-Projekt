from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from processing import pipeline
from processing.input_mode import ResolvedInputs
from processing.raw_loader import UniversePanelBundle


def test_pipeline_vendor_guards_integration(tmp_path: Path, monkeypatch):
    idx = pd.date_range("2024-05-01", periods=8, tz="America/New_York")
    prices = pd.DataFrame(
        {
            "AAA": [10.0, 10.5, 10.7, 10.8, 11.0, 11.1, 11.2, 11.3],
            "MAN": [30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7],
            "SPLT": [220.0, 210.0, 205.0, 200.0, 4.0, 3.0, 2.5, 2.0],
        },
        index=idx,
    )
    volume = pd.DataFrame(
        {
            "AAA": [200.0, 210.0, 0.0, 220.0, 225.0, 230.0, 235.0, 240.0],
            "MAN": [500.0] * 8,
            "SPLT": [1.0] * 8,
        },
        index=idx,
    )
    panel_fields = {
        "close": prices.copy(),
        "open": prices + 0.1,
        "high": prices + 0.2,
        "low": prices - 0.2,
        "volume": volume.copy(),
    }
    # Sanity violation: high below open/close/low.
    panel_fields["high"].at[idx[1], "AAA"] = 1.0

    bad_rows_path = tmp_path / "vendor_bad_rows.parquet"
    pd.DataFrame(
        {
            "ts": [idx[4]],
            "symbol": ["MAN"],
            "field": ["close"],
            "reason": ["manual_close_nan"],
        }
    ).to_parquet(bad_rows_path, index=False)

    cfg = {
        "data": {
            "dir": str(tmp_path / "data"),
            "out_dir": str(tmp_path / "out"),
            "diagnostics_path": str(tmp_path / "out" / "filled.diag.json"),
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
            "stage1": {
                "ohlc_mask": {
                    "enabled": True,
                    "eps_abs": 1.0e-12,
                    "eps_rel": 1.0e-8,
                    "mask_nonpositive": True,
                },
                "zero_volume_with_price": {"enabled": True},
            },
            "stage2": {
                "keep_pct_threshold": 0.1,
                "max_start_na": 8,
                "max_end_na": 8,
                "max_gap_bars": 10,
                "hard_drop": True,
                "min_tradable_rows_for_ohl_gate": 2,
                "ohl_missing_pct_max": 1.0,
            },
            "reverse_split_v2": {"enabled": False},
            "filling": {"causal_only": True, "hard_drop": True},
            "outliers": {"enabled": False},
            "return_caps": {"enabled": False},
            "staleness": {"enabled": False},
            "vendor_guards": {
                "enabled": True,
                "bad_rows": {
                    "enabled": True,
                    "path": str(bad_rows_path),
                    "action": "mask_nan",
                },
                "output": {
                    "write_event_log": True,
                    "path": str(tmp_path / "out" / "vendor_anomalies.parquet"),
                },
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

    resolved = ResolvedInputs(
        mode="run_pinned",
        input_dir=tmp_path / "inputs",
        price_globs=[],
        volume_globs=[],
        universe_manifest_path=None,
        universe_meta={"data_policy": {"raw_index_naive_tz": "UTC"}},
    )
    monkeypatch.setattr(
        pipeline, "resolve_processing_inputs", lambda **kwargs: resolved
    )

    bundle = UniversePanelBundle(panel=None, fields=panel_fields)
    monkeypatch.setattr(
        pipeline,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            bundle,
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

    captured_proc_calls: list[dict[str, Any]] = []

    def _proc(df: pd.DataFrame, **kwargs):
        volume_df = kwargs.get("volume_df")
        captured_proc_calls.append(
            {
                "df": df.copy(),
                "volume_df": volume_df.copy()
                if isinstance(volume_df, pd.DataFrame)
                else None,
            }
        )
        return df.copy(), [], {}

    monkeypatch.setattr(pipeline, "process_and_fill_prices", _proc)

    written_json: dict[str, Any] = {}
    written_parquet: dict[str, pd.DataFrame] = {}

    def _write_json(obj: Any, path: Path):
        written_json[Path(path).name] = obj

    def _write_parquet(df: pd.DataFrame, path: Path, compression=None):
        written_parquet[Path(path).name] = df.copy()

    monkeypatch.setattr(pipeline, "atomic_write_json", _write_json)
    monkeypatch.setattr(pipeline, "atomic_write_parquet", _write_parquet)
    monkeypatch.setattr(pipeline, "atomic_write_pickle", lambda *a, **k: None)
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

    assert captured_proc_calls, "process_and_fill_prices was not called"
    first_df = captured_proc_calls[0]["df"]
    first_volume = captured_proc_calls[0]["volume_df"]
    assert "SPLT" in first_df.columns
    assert pd.isna(first_df.at[idx[4], "MAN"])
    assert pd.isna(first_volume.at[idx[2], "AAA"])

    diag = written_json["filled.diag.json"]
    assert diag["schema_version"] == 3
    assert diag["events"]["summary"]["total_events"] >= 1
    assert diag["stages"]["stage1"]["zero_volume_with_price"]["events"] >= 1
    assert "pre_reverse_split" in diag["snapshots"]
    assert "post_close_anchor" in diag["snapshots"]
    assert "post_ohlc" in diag["snapshots"]

    assert "vendor_anomalies.parquet" in written_parquet
    events = written_parquet["vendor_anomalies.parquet"]
    assert {"manual_bad_row", "zero_volume_with_price"}.issubset(set(events["rule"]))
