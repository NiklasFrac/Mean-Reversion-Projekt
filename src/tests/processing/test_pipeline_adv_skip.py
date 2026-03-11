from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import processing.pipeline as pl
from processing.raw_loader import UniversePanelBundle


def test_pipeline_hard_fails_adv_when_no_symbols_kept(tmp_path, monkeypatch):
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2, tz="UTC")
    )
    volume = pd.DataFrame({"AAA": [100.0, 110.0]}, index=prices.index)

    monkeypatch.setattr(
        pl,
        "load_raw_prices_from_universe",
        lambda *a, **k: (
            prices.copy(),
            volume.copy(),
            UniversePanelBundle(),
            {"prices": "p.pkl", "volume": "v.pkl", "panel": None},
        ),
        raising=True,
    )

    # force all symbols to be dropped
    monkeypatch.setattr(
        pl,
        "process_and_fill_prices",
        lambda *a, **k: (pd.DataFrame(), ["AAA"], {}),
        raising=True,
    )

    written: list[Path] = []
    monkeypatch.setattr(
        pl,
        "atomic_write_parquet",
        lambda df, path, compression=None: written.append(Path(path)),
        raising=True,
    )
    monkeypatch.setattr(
        pl,
        "atomic_write_json",
        lambda obj, path: written.append(Path(path)),
        raising=True,
    )
    monkeypatch.setattr(
        pl,
        "atomic_write_pickle",
        lambda obj, path: written.append(Path(path)),
        raising=True,
    )

    cfg = {
        "data": {
            "dir": str(tmp_path),
            "out_dir": str(tmp_path / "out"),
            "filled_prices_exec_path": str(tmp_path / "out" / "exec.parquet"),
            "filled_prices_panel_exec_path": str(tmp_path / "out" / "panel.parquet"),
            "removed_symbols_path": str(tmp_path / "out" / "removed.pkl"),
            "diagnostics_path": str(tmp_path / "out" / "diag.json"),
            "manifest_path": str(tmp_path / "out" / "manifest.json"),
            "adv_out_path": str(tmp_path / "out" / "adv.pkl"),
        },
        "data_processing": {"n_jobs": 1, "max_gap_bars": 3, "pip_freeze": False},
        "mlflow": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(RuntimeError, match="final universe empty before ADV build"):
        pl.main(cfg_path)

    adv_path = Path(cfg["data"]["adv_out_path"])
    assert adv_path not in written
