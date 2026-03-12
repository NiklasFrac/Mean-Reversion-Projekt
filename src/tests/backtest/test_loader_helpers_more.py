from pathlib import Path

import pandas as pd
import pytest

from backtest import loader


def test_loader_prefilter_ok_dataframe_and_error_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    y = pd.Series([1.0, 2.0], index=idx)
    x = pd.Series([1.0, 2.0], index=idx)

    def pf_df(df: pd.DataFrame, *, coint_alpha: float, min_obs: int) -> bool:
        assert list(df.columns) == ["y", "x"]
        assert coint_alpha == pytest.approx(0.05)
        assert min_obs == 30
        return True

    monkeypatch.setattr(loader, "pair_prefilter", pf_df)
    assert loader._prefilter_ok(y, x) is True

    def pf_raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "pair_prefilter", pf_raise)
    assert loader._prefilter_ok(y, x) is True


def test_loader_coerce_ts_and_series_price_at() -> None:
    idx_tz = pd.date_range("2024-01-01", periods=2, tz="America/New_York")
    ts = loader._coerce_ts_like_index(pd.Timestamp("2024-01-01"), idx_tz)
    assert ts.tz is not None

    idx_naive = pd.date_range("2024-01-01", periods=2)
    ts2 = loader._coerce_ts_like_index(pd.Timestamp("2024-01-01", tz="UTC"), idx_naive)
    assert ts2.tz is None

    price_map = {"AAA": pd.Series([1.0, 2.0], index=idx_naive)}
    assert loader.series_price_at(price_map, "AAA", pd.Timestamp("2023-12-31")) is None
    assert loader.series_price_at(price_map, "AAA", pd.Timestamp("2024-01-02")) == 2.0
    assert (
        loader.series_price_at(price_map, "MISSING", pd.Timestamp("2024-01-02")) is None
    )

    df_prices = pd.DataFrame({"close": [10.0, 11.0]}, index=idx_naive)
    assert (
        loader.series_price_at({"AAA": df_prices}, "AAA", pd.Timestamp("2024-01-02"))
        == 11.0
    )


def test_loader_as_price_mapping_variants() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]}, index=idx)
    mapping = loader.as_price_mapping(df)
    assert set(mapping.keys()) == {"AAA", "BBB"}
    assert mapping["AAA"].index.tz is None

    idx_tz = pd.date_range("2024-01-01", periods=2, freq="D", tz="America/New_York")
    mapping_tz = loader.as_price_mapping(
        pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx_tz)
    )
    assert str(mapping_tz["AAA"].index.tz) in {"America/New_York", "EST5EDT"}

    df2 = pd.DataFrame({"open": [1.0, 2.0], "other": [3.0, 4.0]}, index=idx)
    mapping2 = loader.as_price_mapping({"AAA": df2}, prefer_col="close")
    assert "AAA" in mapping2


def test_select_field_from_panel_fallbacks() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df_mi = pd.DataFrame(
        {
            ("AAA", "foo"): [1.0, 2.0],
            ("BBB", "bar"): [3.0, 4.0],
        },
        index=idx,
    )
    df_mi.columns = pd.MultiIndex.from_tuples(df_mi.columns)
    with pytest.raises(ValueError):
        loader._select_field_from_panel(df_mi, prefer_col="close")

    df_flat = pd.DataFrame(
        {"AAA_close": [1.0, 2.0], "BBB_close": [2.0, 3.0], "CCC_open": [3.0, 4.0]},
        index=idx,
    )
    out_flat = loader._select_field_from_panel(df_flat, prefer_col="close")
    assert list(out_flat.columns) == ["AAA_close", "BBB_close", "CCC_open"]


def test_load_price_data_timezone_and_unknown(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"AAA": [1.0, 2.0]}, index=idx)
    path = tmp_path / "prices.csv"
    df.to_csv(path)

    df_utc = loader.load_price_data(path, coerce_timezone="utc")
    assert str(df_utc.index.tz) == "UTC"

    df_weird = loader.load_price_data(path, coerce_timezone="weird")
    assert df_weird.index.tz is not None


def test_load_price_panel_dataframe_keep() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [1.0, 2.0],
            ("BBB", "close"): [2.0, 3.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    out = loader.load_price_panel(panel, coerce_timezone="keep")
    assert isinstance(out.index, pd.DatetimeIndex)


def test_normalize_pairs_and_load_filtered_pairs_csv(tmp_path: Path) -> None:
    pairs = [("AAA", "BBB"), "CCC/DDD"]
    out = loader.load_filtered_pairs(pairs)
    assert "pair_0" in out and "pair_1" in out

    df = pd.DataFrame({"pair": ["AAA-BBB", "CCC-DDD"]})
    path = tmp_path / "pairs.csv"
    df.to_csv(path, index=False)
    out_csv = loader.load_filtered_pairs(path)
    assert "AAA-BBB" in out_csv


def test_load_adv_map_from_pickle_structures(tmp_path: Path) -> None:
    obj = {"AAA": {"adv": 1000.0}, "BBB": 2000.0}
    pkl = tmp_path / "adv.pkl"
    pd.to_pickle(obj, pkl)
    out = loader.load_adv_map(pkl)
    assert out["AAA"] == 1000.0
    assert out["BBB"] == 2000.0

    wide = pd.DataFrame(
        {
            "AAA": {"adv": 100.0, "last": 1.0},
            "BBB": {"adv": 200.0, "last": 2.0},
        }
    )
    wide_path = tmp_path / "adv_wide.pkl"
    pd.to_pickle(wide, wide_path)
    out_wide = loader.load_adv_map(wide_path)
    assert out_wide["AAA"] == 100.0


def test_filter_by_adv_and_save(tmp_path: Path) -> None:
    df = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [3.0, 4.0]})
    adv = {"AAA": 100.0, "BBB": 50.0}
    out = loader.filter_by_adv(df, adv, min_adv=80.0)
    assert list(out.columns) == ["AAA"]

    out_missing = loader.filter_by_adv(
        df, {"AAA": 100.0}, min_adv=90.0, keep_missing=True
    )
    assert "BBB" in out_missing.columns

    out_none = loader.filter_by_adv(df, adv, min_adv=200.0)
    assert out_none.shape[1] == 0

    path = tmp_path / "adv.csv"
    loader.save_adv_map(adv, path)
    assert path.exists()


def test_resolve_borrow_rate_from_default_and_availability() -> None:
    rate = loader.resolve_borrow_rate("AAA", "2024-01-01", default_rate_annual=0.05)
    assert rate == 0.05

    availability = pd.DataFrame({"symbol": ["AAA"], "rate_annual": [0.2]})
    rate2 = loader.resolve_borrow_rate(
        "AAA", "2024-01-01", availability_df=availability
    )
    assert rate2 == 0.2


def test_prepare_pairs_data_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAA": [1, 2, 3, 4, 5], "BBB": [2, 3, 4, 5, 6]}, index=idx)
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    out = loader.prepare_pairs_data(prices, pairs, disable_prefilter=True)
    assert "AAA-BBB" in out

    monkeypatch.setattr(
        loader,
        "evaluate_pair_cointegration",
        lambda *_args, **_kwargs: {"passed": False, "reject_reason": "eg_failed"},
    )
    out2 = loader.prepare_pairs_data(prices, pairs, disable_prefilter=False)
    assert out2 == {}

    out3 = loader.prepare_pairs_data(
        prices, pairs, disable_prefilter=True, prefilter_range=("bad", "bad")
    )
    assert "AAA-BBB" in out3


def test_prepare_pairs_data_filters_non_positive_beta_without_prefilter() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"AAA": [5, 4, 3, 2, 1], "BBB": [1, 2, 3, 4, 5]},
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}

    out = loader.prepare_pairs_data(prices, pairs, disable_prefilter=True)
    assert out == {}


def test_prepare_pairs_data_passes_pair_prefilter_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAA": [1, 2, 3, 4, 5], "BBB": [2, 3, 4, 5, 6]}, index=idx)
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    seen: dict[str, float] = {}

    def fake_eval(
        _df: pd.DataFrame, *, coint_alpha: float, min_obs: int, half_life_cfg
    ):
        seen["coint_alpha"] = float(coint_alpha)
        seen["min_obs"] = float(min_obs)
        assert isinstance(half_life_cfg, dict)
        return {
            "passed": True,
            "beta": 1.3,
            "z_window": 9,
            "max_hold_days": 18,
            "half_life": 9.0,
        }

    monkeypatch.setattr(loader, "evaluate_pair_cointegration", fake_eval)
    out = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=False,
        pair_prefilter_cfg={
            "coint_alpha": 1.0,
            "min_obs": 2,
            "half_life": {
                "min_days": 5,
                "max_days": 60,
                "max_hold_multiple": 2.0,
                "min_derived_days": 5,
            },
        },
    )

    assert "AAA-BBB" in out
    assert seen == {"coint_alpha": 1.0, "min_obs": 2.0}
    assert out["AAA-BBB"]["meta"]["cointegration"]["z_window"] == 9


def test_prepare_pairs_data_adv_modes() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAA": [1, 2, 3, 4, 5], "BBB": [2, 3, 4, 5, 6]}, index=idx)
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    adv_map = {"AAA": 100.0, "BBB": 400.0}
    out = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=True,
        adv_map=adv_map,
        pair_adv_mode="geometric",
    )
    meta = out["AAA-BBB"]["meta"]
    assert meta["adv_pair_usd"] == pytest.approx(200.0)

    adv_map_single = {"AAA": 100.0}
    out_single = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=True,
        adv_map=adv_map_single,
        pair_adv_mode="max",
    )
    assert out_single["AAA-BBB"]["meta"]["adv_pair_usd"] == pytest.approx(100.0)


def test_prepare_pairs_data_missing_ticker_verbose() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAA": [1, 2, 3, 4, 5]}, index=idx)
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    out = loader.prepare_pairs_data(prices, pairs, disable_prefilter=True, verbose=True)
    assert out == {}
