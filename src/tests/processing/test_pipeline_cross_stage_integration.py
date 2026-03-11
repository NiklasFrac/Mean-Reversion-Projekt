from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from processing.pipeline import main as pipeline_main


def _write_persisted_universe_outputs(base_data_dir: Path) -> Path:
    out_dir = (
        base_data_dir
        / "by_run"
        / "RUN-INTEG-20260101T000000Z-ABCDEF12_deadbeef"
        / "outputs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.bdate_range("2024-01-02", periods=40)
    symbols = ["AAA", "BBB"]

    close = {
        "AAA": pd.Series(100.0 + np.arange(len(idx)) * 0.4, index=idx),
        "BBB": pd.Series(50.0 + np.arange(len(idx)) * 0.2, index=idx),
    }
    open_ = {sym: s * 0.998 for sym, s in close.items()}
    high = {sym: s * 1.01 for sym, s in close.items()}
    low = {sym: s * 0.99 for sym, s in close.items()}

    # Inject one hard-cross violation and one zero-volume-with-price event.
    high["BBB"].iloc[5] = low["BBB"].iloc[5] * 0.95

    raw_prices = pd.DataFrame(index=idx)
    raw_prices_unadj = pd.DataFrame(index=idx)
    for sym in symbols:
        raw_prices[f"{sym}_open"] = open_[sym].astype(float)
        raw_prices[f"{sym}_high"] = high[sym].astype(float)
        raw_prices[f"{sym}_low"] = low[sym].astype(float)
        raw_prices[f"{sym}_close"] = close[sym].astype(float)

        raw_prices_unadj[f"{sym}_open"] = open_[sym].astype(float)
        raw_prices_unadj[f"{sym}_high"] = high[sym].astype(float)
        raw_prices_unadj[f"{sym}_low"] = low[sym].astype(float)
        raw_prices_unadj[f"{sym}_close"] = close[sym].astype(float)

    volume = pd.DataFrame(
        {
            "AAA": np.full(len(idx), 1_000_000.0),
            "BBB": np.full(len(idx), 900_000.0),
        },
        index=idx,
    )
    volume.iloc[8, volume.columns.get_loc("BBB")] = 0.0

    raw_prices.to_pickle(out_dir / "raw_prices.pkl")
    raw_prices_unadj.to_pickle(out_dir / "raw_prices_unadj.pkl")
    volume.to_pickle(out_dir / "raw_volume.pkl")
    volume.to_pickle(out_dir / "raw_volume_unadj.pkl")

    (out_dir / "tickers_universe.csv").write_text(
        "ticker\nAAA\nBBB\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.6.3",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "cfg_path": "runs/configs/config_universe.yaml",
        "cfg_hash": "deadbeef",
        "run_id": "RUN-INTEG-20260101T000000Z-ABCDEF12",
        "n_tickers_initial": 2,
        "n_tickers_final": 2,
        "monitoring": {},
        "extra": {
            "artifacts": {
                "prices": "runs/data/raw_prices.mock.pkl",
                "prices_canonical": "runs/data/raw_prices.pkl",
                "prices_unadjusted": "runs/data/raw_prices_unadj.pkl",
                "volumes": "runs/data/raw_volume.mock.pkl",
                "volumes_canonical": "runs/data/raw_volume.pkl",
                "volumes_unadjusted": "runs/data/raw_volume_unadj.pkl",
            },
            "data_policy": {
                "raw_index_naive_tz": "UTC",
                "raw_index_timezone": "UTC",
                "raw_index_is_tz_naive": True,
                "download_interval": "1d",
            },
        },
    }
    (out_dir / "universe_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return out_dir


def test_pipeline_run_pinned_cross_stage_contract(tmp_path: Path, monkeypatch) -> None:
    # Keep calendar behavior deterministic across environments with/without
    # pandas_market_calendars installed.
    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    monkeypatch.chdir(tmp_path)
    base_data_dir = tmp_path / "runs" / "data"
    pinned_outputs = _write_persisted_universe_outputs(base_data_dir)
    out_dir = tmp_path / "runs" / "data" / "processed" / "integration_contract"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "data": {
            "dir": str(base_data_dir),
            "out_dir": str(out_dir),
            "input_mode": "run_pinned",
            "pinned_universe_outputs_dir": str(pinned_outputs),
            "strict_inputs": True,
            "allow_fallback_to_legacy": False,
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
            "logging_level": "INFO",
            "calendars": {"default": "XNYS", "symbol_calendar_csv_enabled": False},
            "vendor_tz": "UTC",
            "vendor_tz_policy": "config_wins",
            "rth_only": True,
            "grid_mode": "calendar",
            "return_caps": {"enabled": False},
            "staleness": {"enabled": False},
            "filling": {"causal_only": True, "hard_drop": True},
            "vendor_guards": {"enabled": False},
            "stage1": {
                "ohlc_mask": {
                    "enabled": True,
                    "eps_abs": 1.0e-12,
                    "eps_rel": 1.0e-8,
                    "mask_nonpositive": True,
                },
                "zero_volume_with_price": {
                    "enabled": True,
                    "min_price_for_zero_volume_rule": 0.0,
                },
            },
            "stage2": {
                "keep_pct_threshold": 0.8,
                "max_start_na": 3,
                "max_end_na": 3,
                "max_gap_bars": 3,
                "hard_drop": True,
                "min_tradable_rows_for_ohl_gate": 1,
                "ohl_missing_pct_max": 1.0,
            },
            "reverse_split_v2": {"enabled": True},
            "panel_reconcile": {"eps_abs": 1.0e-12, "eps_rel": 1.0e-8},
            "adv_mode": "dollar",
            "adv_stat": "mean",
            "adv_price_source": "raw_unadjusted",
            "adv_window": 5,
            "adv_gates": {
                "min_valid_ratio": 0.0,
                "min_total_windows_for_adv_gate": 1,
                "max_invalid_window_ratio": 1.0,
            },
            "event_log": {
                "enabled": True,
                "path": str(out_dir / "processing_events.parquet"),
            },
            "parquet_compression": "zstd",
            "n_jobs": 1,
            "pip_freeze": False,
        },
        "mlflow": {"enabled": False},
    }

    cfg_dir = tmp_path / "runs" / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config_processing_integration_contract.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    pipeline_main(cfg_path)

    exec_path = out_dir / "filled_prices_exec.parquet"
    panel_path = out_dir / "filled_prices_panel_exec.parquet"
    diag_path = out_dir / "filled.diag.json"
    adv_path = out_dir / "adv_map.pkl"
    assert exec_path.exists()
    assert panel_path.exists()
    assert diag_path.exists()
    assert adv_path.exists()

    exec_df = pd.read_parquet(exec_path)
    panel_df = pd.read_parquet(panel_path)
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    adv = pd.read_pickle(adv_path)

    assert sorted(map(str, exec_df.columns)) == ["AAA", "BBB"]
    assert str(exec_df.index.tz) == "America/New_York"
    assert not exec_df.index.has_duplicates
    assert exec_df.index.is_monotonic_increasing

    assert isinstance(panel_df.columns, pd.MultiIndex)
    assert {"open", "high", "low", "close", "volume"}.issubset(
        set(map(str, panel_df.columns.get_level_values(1)))
    )

    assert diag["inputs"]["input_mode"] == "run_pinned"
    assert diag["inputs"]["vendor_tz_resolution"]["policy"] == "config_wins"
    assert diag["inputs"]["vendor_tz_resolution"]["upstream"] == "UTC"
    assert diag["stages"]["stage1"]["ohlc"]["hard_row_events"] >= 1
    assert diag["stages"]["stage1"]["zero_volume_with_price"]["events"] >= 1

    assert sorted(map(str, adv.keys())) == ["AAA", "BBB"]
