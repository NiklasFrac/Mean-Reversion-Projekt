from __future__ import annotations

from pathlib import Path

import pandas as pd

from processing.raw_loader import (
    UniversePanelBundle,
    _extract_panel_from_suffixes,
    _load_any_prices,
    load_raw_prices_from_universe,
)


def test_extract_panel_from_suffixes_builds_panel_and_close():
    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    df = pd.DataFrame(
        {
            "AAA_open": [1.0, 2.0],
            "AAA_close": [10.0, 11.0],
            "BBB_meta": [0.0, 0.0],  # ignored (no suffix)
        },
        index=idx,
    )

    fields, panel, close_df = _extract_panel_from_suffixes(df)

    assert set(fields) == {"open", "close"}
    assert ("AAA", "open") in panel.columns
    assert ("AAA", "close") in panel.columns
    pd.testing.assert_frame_equal(
        close_df, pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)
    )


def test_extract_panel_from_suffixes_includes_volume():
    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    df = pd.DataFrame(
        {"AAA_close": [1.0, 2.0], "AAA_volume": [100.0, 200.0]}, index=idx
    )

    fields, panel, close_df = _extract_panel_from_suffixes(df)

    assert "volume" in fields
    assert ("AAA", "volume") in panel.columns
    assert close_df is not None


def test_load_any_prices_supports_dict_payload(tmp_path: Path):
    path = tmp_path / "raw_prices.pkl"
    payload = {"AAA": [1.0, 2.0], "BBB": pd.Series([3.0, 4.0])}
    pd.to_pickle(payload, path)

    df = _load_any_prices(path)

    assert list(df.columns) == ["AAA", "BBB"]
    pd.testing.assert_index_equal(df.index, pd.RangeIndex(0, 2))


def test_load_raw_prices_from_universe_falls_back_to_panel_fields(
    tmp_path: Path, monkeypatch
):
    data_dir = tmp_path / "u"
    data_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [1.0, 2.0, 3.0],
            ("AAA", "volume"): [10.0, 11.0, 12.0],
        },
        index=idx,
    )
    panel.to_pickle(data_dir / "raw_prices_panel.pkl")

    # loader globbing is relative to CWD
    monkeypatch.chdir(data_dir)

    prices, volume, bundle, used = load_raw_prices_from_universe(
        Path("."), include_bundle=True
    )

    assert isinstance(bundle, UniversePanelBundle)
    expected_prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    expected_prices.columns.name = "symbol"
    pd.testing.assert_frame_equal(prices, expected_prices)
    expected_volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)
    expected_volume.columns.name = "symbol"
    pd.testing.assert_frame_equal(volume, expected_volume)
    assert used["prices"].endswith("[close]")
    assert used["panel"].endswith("raw_prices_panel.pkl")


def test_load_raw_prices_handles_field_symbol_order(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "u2"
    data_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2020-01-01", periods=2, tz="UTC")
    # MultiIndex is (field, symbol)
    cols = pd.MultiIndex.from_product(
        [["close", "volume"], ["AAA"]], names=["field", "symbol"]
    )
    panel = pd.DataFrame([[1.0, 10.0], [2.0, 11.0]], index=idx, columns=cols)
    panel.to_pickle(data_dir / "raw_prices_panel.pkl")

    monkeypatch.chdir(data_dir)

    prices, volume, bundle, used = load_raw_prices_from_universe(
        Path("."), include_bundle=True
    )

    expected_prices = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    expected_prices.columns.name = "symbol"
    pd.testing.assert_frame_equal(prices, expected_prices)
    expected_volume = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)
    expected_volume.columns.name = "symbol"
    pd.testing.assert_frame_equal(volume, expected_volume)
    assert used["panel"].endswith("raw_prices_panel.pkl")
