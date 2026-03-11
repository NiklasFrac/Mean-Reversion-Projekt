from pathlib import Path

import pandas as pd

from backtest import loader


def test_load_price_data_timezones_and_formats(tmp_path: Path) -> None:
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

    pkl = tmp_path / "prices.pkl"
    panel.to_pickle(pkl)
    df_utc = loader.load_price_data(pkl, prefer_col="close", coerce_timezone="utc")
    assert df_utc.index.tz is not None and str(df_utc.index.tz) == "UTC"

    df_naive = loader.load_price_data(pkl, prefer_col="close", coerce_timezone="naive")
    assert df_naive.index.tz is None

    df_keep = loader.load_price_data(pkl, prefer_col="close", coerce_timezone="keep")
    assert df_keep.index.tz is None

    pq = tmp_path / "prices.parquet"
    panel.to_parquet(pq)
    df_unknown = loader.load_price_data(
        pq, prefer_col="close", coerce_timezone="unknown"
    )
    assert df_unknown.index.tz is not None


def test_load_filtered_pairs_variants(tmp_path: Path) -> None:
    pair_df = pd.DataFrame({"pair": ["AAA/BBB", "CCC-DDD"]})
    pair_csv = tmp_path / "pairs.csv"
    pair_df.to_csv(pair_csv, index=False)
    pairs = loader.load_filtered_pairs(pair_csv)
    assert pairs["AAA/BBB"]["t1"] == "AAA"

    yx_df = pd.DataFrame({"y": ["AAA"], "x": ["BBB"]})
    yx_csv = tmp_path / "pairs_yx.csv"
    yx_df.to_csv(yx_csv, index=False)
    pairs_yx = loader.load_filtered_pairs(yx_csv)
    assert "AAA-BBB" in pairs_yx

    pairs_map = loader.load_filtered_pairs({"P1": "AAA/BBB"})
    assert pairs_map["P1"]["t2"] == "BBB"

    pairs_list = loader.load_filtered_pairs(["EEE-FFF"])
    assert pairs_list["pair_0"]["t1"] == "EEE"


def test_load_adv_map_wide_and_index(tmp_path: Path) -> None:
    wide = pd.DataFrame(
        {"AAA": [1_000.0, 50.0], "BBB": [2_000.0, 60.0]}, index=["adv", "last_price"]
    )
    wide_path = tmp_path / "adv_wide.pkl"
    wide.to_pickle(wide_path)
    adv_wide = loader.load_adv_map(wide_path)
    assert adv_wide["AAA"] == 1_000.0

    idx_df = pd.DataFrame(
        {"adv": [1_500.0, 2_500.0]}, index=pd.Index(["AAA", "BBB"], name="symbol")
    )
    idx_path = tmp_path / "adv_idx.pkl"
    idx_df.to_pickle(idx_path)
    adv_idx = loader.load_adv_map(idx_path)
    assert adv_idx["BBB"] == 2_500.0
