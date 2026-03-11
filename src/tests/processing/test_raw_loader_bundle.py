from __future__ import annotations

from pathlib import Path

import pandas as pd

from processing.raw_loader import UniversePanelBundle, load_raw_prices_from_universe


def test_load_raw_prices_from_universe_returns_bundle(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "u"
    data_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=idx)
    prices.to_pickle(data_dir / "raw_prices.pkl")
    volume.to_pickle(data_dir / "raw_volume.pkl")

    # loader glob is relative to CWD
    monkeypatch.chdir(data_dir)

    prices_out, volume_out, bundle, used = load_raw_prices_from_universe(
        Path("."), include_bundle=True
    )

    assert isinstance(bundle, UniversePanelBundle)
    assert prices_out.equals(prices)
    assert volume_out.equals(volume)
    assert used["prices"].endswith("raw_prices.pkl")
    assert used["volume"].endswith("raw_volume.pkl")
