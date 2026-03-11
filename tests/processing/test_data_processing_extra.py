# src/tests/processing/test_data_processing_extra.py
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processing.raw_loader import load_raw_prices_from_universe


def _mk_df(rows: int = 6, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="B", tz="UTC")
    data = {f"T{i}": np.linspace(10.0, 10.0 + i, rows) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


def test_load_raw_prices_from_universe_glob_star_and_missing_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Deckt zwei Zweige ab:
      1) Discovery über Pattern 'raw_prices.*.pkl' (Wildcard-Branch).
      2) Kein Volume vorhanden -> df_volume ist None und used_paths markiert None.
    """
    data_dir = tmp_path / "universe"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Wildcard-Variante (soll gefunden werden)
    p_star = data_dir / "raw_prices.20240101.pkl"
    _mk_df(5, 2).to_pickle(p_star)

    # _discover nutzt CWD → ins Datenverzeichnis wechseln und relativ aufrufen
    monkeypatch.chdir(data_dir)
    prices, volume, used = load_raw_prices_from_universe(Path("."))

    assert isinstance(prices, pd.DataFrame) and not prices.empty
    assert volume is None, "Wenn kein raw_volume.* existiert, muss df_volume None sein."
    assert isinstance(used, dict)
    # Korrekte Keys im used_paths:
    assert used.get("prices") is not None
    assert used.get("volume") in (None, "")


def test_load_raw_prices_prefers_newest_when_both_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Wenn sowohl 'raw_prices.pkl' als auch 'raw_prices.*.pkl' existieren,
    soll der *neuere* Artefakt gewählt werden (Discovery-Strategie).
    """
    data_dir = tmp_path / "u"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ältere Datei: plain
    p_plain = data_dir / "raw_prices.pkl"
    _mk_df(4, 2).to_pickle(p_plain)

    # Kurze Pause, damit mtime sicher größer wird
    time.sleep(0.02)

    # Neuere Datei: Wildcard
    p_star = data_dir / "raw_prices.20240202.pkl"
    _mk_df(4, 3).to_pickle(p_star)

    # Nur die Pfadwahl testen
    monkeypatch.chdir(data_dir)
    prices, volume, used = load_raw_prices_from_universe(Path("."))

    assert prices.shape[1] == 3  # neuere Wildcard-Datei hat 3 Spalten
    # used['prices'] ist relativ -> Namen vergleichen
    assert used.get("prices") is not None
    assert Path(used["prices"]).name == p_star.name


def test_normalize_config_maps_calendar_only():
    cfg = {
        "data_processing": {
            "max_gap": 9,
            "calendar": "XNAS",
        }
    }

    from processing import pipeline

    normalized = pipeline._normalize_config(cfg)
    assert "max_gap_bars" not in normalized["data_processing"]
    assert normalized["data_processing"]["calendars"]["default"] == "XNAS"
