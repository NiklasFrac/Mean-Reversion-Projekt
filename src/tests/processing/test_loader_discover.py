from __future__ import annotations

from pathlib import Path

import pandas as pd


from processing.raw_loader import _load_any_prices, load_raw_prices_from_universe


def test_load_raw_prices_from_universe_with_wide_and_pivot(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "universe"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Wide-Format (einfaches DataFrame) als Pickle
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    wide = pd.DataFrame({"AAA": [1, 2, 3, 4, 5], "BBB": [2, 3, 4, 5, 6]}, index=idx)
    wide.to_pickle(data_dir / "raw_prices.pkl")

    # Volume optional
    vol = pd.DataFrame({"AAA": [10] * 5, "BBB": [20] * 5}, index=idx)
    vol.to_pickle(data_dir / "raw_volume.pkl")

    # IMPORTANT: _discover uses Path().glob relative to CWD -> switch into tmp and pass the path RELATIVELY
    monkeypatch.chdir(tmp_path)
    prices, volume, used = load_raw_prices_from_universe(Path("universe"))
    assert set(prices.columns) == {"AAA", "BBB"}
    assert volume is not None
    assert used["prices"].endswith("raw_prices.pkl")


def test__load_any_prices_pivot_from_long(tmp_path: Path):
    # long-form (ts, ticker, close) is pivoted to wide
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    df_long = pd.DataFrame(
        {
            "ts": list(idx) * 2,
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "close": [1, 2, 3, 10, 11, 12],
        }
    )
    p = tmp_path / "long.pkl"
    df_long.to_pickle(p)
    out = _load_any_prices(p)
    assert set(out.columns) == {"A", "B"}
    assert len(out) == 3


def test_load_raw_prices_large_universe_short_window_not_rejected(
    tmp_path: Path, monkeypatch
):
    data_dir = tmp_path / "universe_short"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Large expected universe triggers suspicious-artifact guard.
    symbols = [f"S{i:03d}" for i in range(120)]
    (data_dir / "tickers_universe.csv").write_text(
        "ticker\n" + "\n".join(symbols) + "\n", encoding="utf-8"
    )

    idx = pd.bdate_range("2020-01-01", periods=40, tz="UTC")
    # Legitimate short horizon with full symbol coverage should remain valid.
    prices = pd.DataFrame({f"{s}_close": range(40) for s in symbols}, index=idx)
    volume = pd.DataFrame({s: [1000] * 40 for s in symbols}, index=idx)
    prices.to_pickle(data_dir / "raw_prices.pkl")
    volume.to_pickle(data_dir / "raw_volume_unadj.pkl")

    monkeypatch.chdir(tmp_path)
    loaded_prices, loaded_volume, used = load_raw_prices_from_universe(
        Path("universe_short")
    )

    assert loaded_prices.shape == prices.shape
    assert loaded_volume is not None
    assert loaded_volume.shape == volume.shape
    assert used["prices"].endswith("raw_prices.pkl")
