from __future__ import annotations

from pathlib import Path

import pandas as pd


from processing.raw_loader import _load_any_prices, load_raw_prices_from_universe


def test__load_any_prices_long_csv_pivots_to_wide(tmp_path: Path):
    """
    Deckt CSV + long→wide Pivot ab (ts/ticker/close), OHNE Discovery.
    """
    idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
    df_long = pd.DataFrame(
        {
            "ts": list(idx) * 2,
            "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "close": [1, 2, 3, 4, 10, 11, 12, 13],
        }
    )
    p_csv = tmp_path / "raw_prices.csv"
    df_long.to_csv(p_csv, index=False)

    out = _load_any_prices(p_csv)
    assert set(out.columns) == {"A", "B"}
    assert len(out) == 4
    # index should become UTC-naive after ensure_utc_index; here we only check that ts is parseable
    assert pd.api.types.is_datetime64_any_dtype(out.index)


def test_load_raw_prices_from_universe_parquet_long(tmp_path: Path, monkeypatch):
    """
    Covers discovery (Parquet); volume also as Parquet.
    """
    data_dir = tmp_path / "u"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Long-Form als Parquet (Discovery findet *.parquet)
    idx = pd.date_range("2020-01-01", periods=4, tz="UTC")
    df_long = pd.DataFrame(
        {
            "ts": list(idx) * 2,
            "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "close": [1, 2, 3, 4, 10, 11, 12, 13],
        }
    )
    (data_dir / "raw_prices.parquet").parent.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(data_dir / "raw_prices.parquet", index=False)

    # volume as Parquet (discovery supports this)
    vol = pd.DataFrame(
        {"A": [100, 100, 100, 100], "B": [200, 200, 200, 200]}, index=idx
    )
    vol.to_parquet(data_dir / "raw_volume.parquet")

    # _discover nutzt CWD -> relativ aufrufen
    monkeypatch.chdir(tmp_path)
    prices, volume, used = load_raw_prices_from_universe(Path("u"))

    assert set(prices.columns) == {"A", "B"}
    assert prices.shape[0] == 4
    assert volume is not None
    assert used["prices"].endswith("raw_prices.parquet")
    assert used["volume"].endswith("raw_volume.parquet")
