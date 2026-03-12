# src/tests/processing/test_main_e2e.py
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processing.pipeline import main as pipeline_main
from processing.timebase import ensure_utc_index


def _build_universe(tmp_root: Path) -> tuple[Path, Path, Path]:
    # Structure:
    # <tmp>/configs/config.yaml
    # <tmp>/backtest/data/raw_prices.pkl
    # <tmp>/backtest/data/raw_volume.pkl
    cfg_dir = tmp_root / "configs"
    data_dir = tmp_root / "backtest" / "data"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # prices: 20 business days, 3 tickers
    idx = pd.bdate_range("2020-01-01", periods=20)
    aaa = pd.Series(np.linspace(100.0, 119.0, len(idx)), index=idx)
    aaa.iloc[8] = np.nan
    bbb = pd.Series(np.nan, index=idx)
    bbb.iloc[0:10] = np.linspace(50.0, 59.0, 10)  # 0.5 Coverage
    ccc = pd.Series(np.linspace(200.0, 219.0, len(idx)), index=idx)
    ccc.iloc[12] = 2000.0  # outlier

    prices = pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc}, index=idx)
    prices = ensure_utc_index(prices)
    volumes = pd.DataFrame(
        {"AAA": 1_000, "BBB": 2_000, "CCC": 3_000}, index=idx
    )  # konstant
    volumes = ensure_utc_index(volumes)

    p_prices = data_dir / "raw_prices.pkl"
    p_volume = data_dir / "raw_volume.pkl"
    prices.to_pickle(p_prices)
    volumes.to_pickle(p_volume)

    # Important: relative path in YAML because load_raw_prices_from_universe globs relative paths
    cfg = cfg_dir / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dir: 'backtest/data'",  # <- RELATIV!
                "  filled_prices_path: 'backtest/data/filled_data.pkl'",
                "",
                "data_processing:",
                "  max_gap_bars: 12",
                "  n_jobs: 1",
                "  grid_mode: 'leader'",
                "  calendar: 'XNYS'",
                "  max_start_na: 5",
                "  max_end_na: 3",
                "  stage2:",
                "    keep_pct_threshold: 0.7",
                "    max_start_na: 5",
                "    max_end_na: 3",
                "    max_gap_bars: 12",
                "  adv_window: 5",
                "  adv_gates:",
                "    min_valid_ratio: 0.0",
                "    min_total_windows_for_adv_gate: 1",
                "    max_invalid_window_ratio: 1.0",
                "  parquet_compression: 'zstd'",
                "  outliers:",
                "    enabled: true",
                "    zscore: 6.0",
                "    window: 11",
                "    use_log_returns: true",
                "",
                "mlflow:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    return cfg, p_prices, p_volume


@pytest.mark.usefixtures("stub_processing_modules")
def test_main_e2e(
    tmp_path: Path,
    golden_dir: Path,
    update_golden: bool,
    mask_diag_payload,
    mask_manifest_payload,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg_path, p_prices, p_volume = _build_universe(tmp_path)

    # Keep calendar behavior deterministic across environments with/without
    # pandas_market_calendars installed.
    fake = types.ModuleType("pandas_market_calendars")
    fake.get_calendar = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)

    # IMPORTANT 1: switch into the tmp directory so relative paths from the YAML resolve
    monkeypatch.chdir(tmp_path)

    # IMPORTANT 2: force exactly this config to be used (no runs/configs/...).
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg_path))
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    # Execute (direct production path)
    pipeline_main(cfg_path)  # run

    # artifacts
    out_pkl = tmp_path / "backtest" / "data" / "filled_data.pkl"
    out_removed = tmp_path / "backtest" / "data" / "filled_data_removed.pkl"
    out_diag = tmp_path / "backtest" / "data" / "filled_data.diag.json"
    out_manifest = tmp_path / "backtest" / "data" / "filled_data_manifest.json"

    assert (
        out_pkl.exists()
        and out_removed.exists()
        and out_diag.exists()
        and out_manifest.exists()
    )

    # Check contents
    df = pd.read_pickle(out_pkl)
    removed = pd.read_pickle(out_removed)
    assert list(sorted(df.columns)) == ["AAA", "CCC"]
    assert set(removed) == {"BBB"}
    d_diag = json.loads(out_diag.read_text(encoding="utf-8"))
    d_manifest = json.loads(out_manifest.read_text(encoding="utf-8"))

    masked_diag = mask_diag_payload(d_diag)
    masked_manifest = mask_manifest_payload(d_manifest)

    g_diag = golden_dir / "diag_golden.json"
    g_manifest = golden_dir / "manifest_golden.json"

    # first-run convenience: create goldens automatically if they are missing
    if update_golden or not g_diag.exists() or not g_manifest.exists():
        g_diag.parent.mkdir(parents=True, exist_ok=True)
        g_diag.write_text(json.dumps(masked_diag, indent=2), encoding="utf-8")
        g_manifest.write_text(json.dumps(masked_manifest, indent=2), encoding="utf-8")
    else:
        exp_diag = json.loads(g_diag.read_text(encoding="utf-8"))
        exp_manifest = json.loads(g_manifest.read_text(encoding="utf-8"))
        assert masked_diag == exp_diag
        assert masked_manifest == exp_manifest
