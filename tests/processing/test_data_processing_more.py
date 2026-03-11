from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processing.raw_loader import load_raw_prices_from_universe


def _mk_df(rows: int = 4, cols: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, tz="UTC", freq="B")
    data = {f"T{i}": np.arange(1, rows + 1, dtype=float) * (i + 1) for i in range(cols)}
    return pd.DataFrame(data, index=idx)


@pytest.mark.skipif(
    pytest.importorskip("pyarrow", reason="pyarrow required for parquet") is None,
    reason="pyarrow missing",
)
def test_load_raw_prices_from_universe_parquet_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Deckt den bislang ungetesteten Zweig ab:
      - Preise aus 'raw_prices.parquet'
      - Volume aus 'raw_volume.parquet'
      - _discover via CWD + relativer Pfad
    """
    data_dir = tmp_path / "u"
    data_dir.mkdir(parents=True, exist_ok=True)

    prices_parquet = data_dir / "raw_prices.parquet"
    volume_parquet = data_dir / "raw_volume.parquet"

    _mk_df(5, 3).to_parquet(prices_parquet)
    _mk_df(5, 3).to_parquet(volume_parquet)

    monkeypatch.chdir(data_dir)
    prices, volume, used = load_raw_prices_from_universe(Path("."))

    assert isinstance(prices, pd.DataFrame) and prices.shape == (5, 3)
    assert isinstance(volume, pd.DataFrame) and volume.shape == (5, 3)

    assert used.get("prices") and Path(used["prices"]).name == "raw_prices.parquet"
    assert used.get("volume") and Path(used["volume"]).name == "raw_volume.parquet"


def test_load_raw_prices_from_universe_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Deckt den Fehlerpfad ab, wenn GAR KEIN raw_prices.* vorhanden ist.
    """
    empty_dir = tmp_path / "empty_u"
    empty_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(empty_dir)
    with pytest.raises(FileNotFoundError) as ei:
        load_raw_prices_from_universe(Path("."))

    msg = str(ei.value)
    assert "raw_prices.* not found" in msg
    # robust: data_dir kann als '.' formatiert sein – akzeptiere beides
    assert str(empty_dir) in msg or " under ." in msg
