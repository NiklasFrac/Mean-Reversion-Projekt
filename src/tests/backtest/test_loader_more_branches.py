from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import loader


def test_select_field_from_panel_field_level_inferred() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    df = pd.DataFrame(
        {
            ("AAA", "close"): [1.0, 1.1],
            ("AAA", "open"): [0.9, 1.0],
            ("BBB", "close"): [2.0, 2.1],
            ("BBB", "open"): [1.9, 2.0],
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    out = loader.select_field_from_panel(df, field="close")
    assert list(out.columns) == ["AAA", "BBB"]


def test_load_price_data_required_symbols_and_unsupported(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    flat = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]}, index=idx)
    csv_path = tmp_path / "prices.csv"
    flat.to_csv(csv_path)

    df = loader.load_price_data(
        csv_path,
        prefer_col="close",
        required_symbols=["BBB"],
        apply_corporate_actions=True,
        coerce_timezone="naive",
    )
    assert list(df.columns) == ["BBB"]

    bad_path = tmp_path / "prices.txt"
    bad_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        loader.load_price_data(bad_path)


def test_load_price_panel_parquet(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [1.0, 2.0, 3.0],
            ("BBB", "close"): [2.0, 3.0, 4.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)

    loaded = loader.load_price_panel(path, coerce_timezone="utc")
    assert not loaded.empty
    assert str(loaded.index.tz) == "UTC"


def test_load_adv_map_series_and_error(tmp_path: Path) -> None:
    adv_series = pd.Series({"AAA": 1000.0, "BBB": 2000.0})
    pkl = tmp_path / "adv.pkl"
    adv_series.to_pickle(pkl)
    out = loader.load_adv_map(pkl)
    assert out["AAA"] == 1000.0

    bad = pd.DataFrame({"symbol": ["AAA", "BBB"], "note": ["x", "y"]})
    bad_path = tmp_path / "adv_bad.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        loader.load_adv_map(bad_path)


def test_resolve_borrow_rate_with_ctx_events_and_availability() -> None:
    class Ctx:
        def resolve_borrow_rate(self, *_args, **_kwargs):
            return None

        def events_for_range(self, *_args, **_kwargs):
            return pd.DataFrame({"rate_annual": [0.25]})

    rate = loader.resolve_borrow_rate("AAA", "2024-01-02", borrow_ctx=Ctx())
    assert rate == 0.25

    availability = pd.DataFrame({"symbol": ["AAA"], "rate_annual": [0.3]})
    rate2 = loader.resolve_borrow_rate(
        "AAA", "2024-01-02", availability_df=availability
    )
    assert rate2 == 0.3


def test_prepare_pairs_data_prefilter_range_empty() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(1.0, 2.0, len(idx)),
            "BBB": np.linspace(2.0, 3.0, len(idx)),
        },
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    out = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=True,
        prefilter_range=("2030-01-01", "2030-01-05"),
    )
    assert "AAA-BBB" in out


def test_prefilter_ok_exception_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    y = pd.Series([1.0, 2.0, 3.0], index=idx)
    x = pd.Series([1.0, 2.0, 3.0], index=idx)

    def bad_prefilter(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "pair_prefilter", bad_prefilter)
    assert loader._prefilter_ok(y, x) is True


def test_as_price_mapping_with_dataframe_values() -> None:
    idx = pd.bdate_range("2024-01-02", periods=2)
    df = pd.DataFrame({"open": [1.0, 1.1], "vwap": [1.0, 1.1]}, index=idx)
    mapping = loader.as_price_mapping({"AAA": df}, prefer_col="close")
    assert "AAA" in mapping


def test_load_filtered_pairs_pickle_dataframe(tmp_path: Path) -> None:
    df = pd.DataFrame([{"t1": "AAA", "t2": "BBB"}])
    path = tmp_path / "pairs.pkl"
    df.to_pickle(path)
    out = loader.load_filtered_pairs(path)
    assert "AAA-BBB" in out


def test_load_adv_map_csv_sniff_fallback(tmp_path: Path) -> None:
    p = tmp_path / "adv_semicolon.csv"
    p.write_text("ticker;adv_usd\nAAA;1000\n", encoding="utf-8")
    out = loader.load_adv_map(p)
    assert out["AAA"] == 1000.0


def test_series_price_at_before_first_returns_none() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    s = pd.Series([1.0, 2.0], index=idx)
    val = loader.series_price_at(s, "AAA", pd.Timestamp("2023-12-31"))
    assert val is None


def test_coerce_ts_like_index_tz() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    ts = loader._coerce_ts_like_index("2024-01-01", idx)
    assert ts.tz is not None


def test_resolve_borrow_rate_bad_day_and_fallbacks() -> None:
    class BadCtx:
        def resolve_borrow_rate(self, *_args, **_kwargs):
            raise RuntimeError("boom")

        def events_for_range(self, *_args, **_kwargs):
            raise RuntimeError("boom")

        @property
        def default_rate_annual(self):
            raise RuntimeError("boom")

    availability = pd.DataFrame({"ticker": ["AAA"], "rate_annual": [0.5]})
    rate = loader.resolve_borrow_rate(
        "AAA",
        "bad-date",
        borrow_ctx=BadCtx(),
        default_rate_annual=0.12,
        availability_df=availability,
    )
    assert rate == 0.12


def test_prepare_pairs_data_returns_deterministic_pair_order() -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame(
        {
            "CCC": np.linspace(10.0, 11.0, len(idx)),
            "AAA": np.linspace(20.0, 21.0, len(idx)),
            "BBB": np.linspace(30.0, 31.0, len(idx)),
        },
        index=idx,
    )
    pairs = {
        "CCC-AAA": {"t1": "CCC", "t2": "AAA"},
        "AAA-BBB": {"t1": "AAA", "t2": "BBB"},
        "BBB-CCC": {"t1": "BBB", "t2": "CCC"},
    }
    out = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=True,
    )
    assert list(out.keys()) == sorted(out.keys())
