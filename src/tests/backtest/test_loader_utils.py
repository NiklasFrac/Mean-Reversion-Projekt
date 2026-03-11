from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import calendars
from backtest.loader import (
    as_price_mapping,
    filter_by_adv,
    load_adv_map,
    load_filtered_pairs,
    load_price_data,
    load_price_panel,
    prepare_pairs_data,
    resolve_borrow_rate,
    save_adv_map,
    select_field_from_panel,
    series_price_at,
)


def test_calendars_mapping_and_lag() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    price_data = {"AAA": pd.Series(range(5), index=idx)}
    cal = calendars.build_trading_calendar(price_data)
    assert len(cal) == len(idx)

    ts = pd.Timestamp("2024-01-03")
    assert calendars.map_to_calendar(ts, cal, policy="prior") == ts
    assert calendars.map_to_calendar(
        ts + pd.Timedelta(days=3), cal, policy="prior"
    ) == pd.Timestamp("2024-01-05")
    assert calendars.apply_settlement_lag(ts, cal, lag_bars=2) == pd.Timestamp(
        "2024-01-05"
    )


def test_loader_price_panel_and_select(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [1.0, 2.0, 3.0],
            ("AAA", "open"): [1.0, 2.0, 3.0],
            ("BBB", "close"): [10.0, 11.0, 12.0],
            ("BBB", "open"): [10.0, 11.0, 12.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    p = tmp_path / "panel.pkl"
    panel.to_pickle(p)

    loaded = load_price_panel(p, coerce_timezone="naive")
    close = select_field_from_panel(loaded, field="close")
    assert list(close.columns) == ["AAA", "BBB"]

    flat = pd.DataFrame({"AAA": [1, 2], "BBB": [3, 4]}, index=idx[:2])
    p2 = tmp_path / "flat.csv"
    flat.to_csv(p2)
    loaded2 = load_price_data(p2, prefer_col="close", coerce_timezone="naive")
    assert list(loaded2.columns) == ["AAA", "BBB"]

    tsv = tmp_path / "panel.tsv"
    flat.to_csv(tsv, sep="\t")
    with pytest.raises(ValueError):
        load_price_panel(tsv, coerce_timezone="keep")


def test_loader_pairs_adv_and_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(
        {"AAA": np.linspace(10, 12, len(idx)), "BBB": np.linspace(20, 18, len(idx))},
        index=idx,
    )

    pairs_csv = tmp_path / "pairs.csv"
    pd.DataFrame([{"t1": "AAA", "t2": "BBB"}]).to_csv(pairs_csv, index=False)
    pairs = load_filtered_pairs(pairs_csv)
    assert "AAA-BBB" in pairs

    adv_path = tmp_path / "adv.csv"
    save_adv_map({"AAA": 1000.0, "BBB": 2000.0}, adv_path)
    adv_map = load_adv_map(adv_path)
    filt = filter_by_adv(prices, adv_map, min_adv=500.0)
    assert list(filt.columns) == ["AAA", "BBB"]

    monkeypatch.setattr(
        "backtest.loader.evaluate_pair_cointegration",
        lambda *_args, **_kwargs: {
            "passed": True,
            "z_window": 8,
            "max_hold_days": 16,
            "half_life": 8.0,
        },
    )
    out = prepare_pairs_data(prices, pairs, disable_prefilter=False)
    assert "AAA-BBB" in out

    pairs2 = load_filtered_pairs(["AAA-BBB", "CCC-DDD"])
    assert "pair_0" in pairs2


def test_loader_misc_helpers(tmp_path: Path) -> None:
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame(
        {"AAA": [1.0, 1.1, 1.2, 1.3], "BBB": [2.0, 2.1, 2.2, 2.3]}, index=idx
    )
    mapping = as_price_mapping(prices)
    assert set(mapping) == {"AAA", "BBB"}

    val = series_price_at(mapping, "AAA", pd.Timestamp("2024-01-03"))
    assert val is not None

    borrow_rate = resolve_borrow_rate(
        "AAA", pd.Timestamp("2024-01-03"), default_rate_annual=0.1
    )
    assert borrow_rate == 0.1

    _ = load_filtered_pairs({"P1": ["AAA", "BBB"]})


def test_build_trading_calendar_intraday() -> None:
    idx = pd.date_range("2024-01-02 10:00", periods=3, freq="h")
    cal = calendars.build_trading_calendar({"AAA": pd.Series(range(3), index=idx)})
    assert len(cal) == 3
