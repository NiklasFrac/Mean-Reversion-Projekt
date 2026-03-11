# src/backtest/tests/test_tr_audit_and_universe.py
from pathlib import Path

import numpy as np
import pandas as pd

from universe.tr_audit import audit_corporate_actions
from universe.universe_provider import apply_universe, load_universe


def test_universe_interval_masks_prices(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": [1, 2, 3, 4, 5],
            "BBB": [10, 11, 12, 13, 14],
        },
        index=dates,
        dtype=float,
    )

    uni = tmp_path / "uni.csv"
    uni.write_text(
        "symbol,start_date,end_date\nAAA,2020-01-02,\nBBB,2020-01-01,2020-01-03\n",
        encoding="utf-8",
    )
    schema, dfu = load_universe(uni, schema="auto")
    masked = apply_universe(prices, schema, dfu)

    # AAA erst ab 2020-01-02
    assert np.isnan(masked.loc[pd.Timestamp("2020-01-01"), "AAA"])
    assert masked.loc[pd.Timestamp("2020-01-02"), "AAA"] == 2.0
    # BBB nur bis inkl. 2020-01-03
    assert masked.loc[pd.Timestamp("2020-01-03"), "BBB"] == 12.0
    assert np.isnan(masked.loc[pd.Timestamp("2020-01-06"), "BBB"])


def test_universe_interval_masks_symbols_not_in_definition(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [10.0, 11.0, 12.0],
        },
        index=dates,
    )

    uni = tmp_path / "uni.csv"
    uni.write_text("symbol,start_date,end_date\nAAA,2020-01-01,\n", encoding="utf-8")
    schema, dfu = load_universe(uni, schema="auto")
    masked = apply_universe(prices, schema, dfu)

    assert list(masked.columns) == ["AAA", "BBB"]
    assert masked["AAA"].notna().all()
    assert masked["BBB"].isna().all()


def test_universe_interval_normalizes_share_class_symbols(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"BRK-B": [300.0, 301.0, 302.0]}, index=dates)

    uni = tmp_path / "uni.csv"
    uni.write_text("symbol,start_date,end_date\nBRK.B,2020-01-02,\n", encoding="utf-8")
    schema, dfu = load_universe(uni, schema="auto")
    masked = apply_universe(prices, schema, dfu)

    assert dfu.loc[0, "symbol"] == "BRK-B"
    assert np.isnan(masked.loc[pd.Timestamp("2020-01-01"), "BRK-B"])
    assert masked.loc[pd.Timestamp("2020-01-02"), "BRK-B"] == 301.0


def test_universe_interval_masks_lowercase_price_columns(tmp_path: Path):
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"aaa": [1.0, 2.0, 3.0]}, index=dates)

    uni = tmp_path / "uni.csv"
    uni.write_text("symbol,start_date,end_date\nAAA,2020-01-02,\n", encoding="utf-8")
    schema, dfu = load_universe(uni, schema="auto")
    masked = apply_universe(prices, schema, dfu)

    assert np.isnan(masked.loc[pd.Timestamp("2020-01-01"), "aaa"])
    assert masked.loc[pd.Timestamp("2020-01-02"), "aaa"] == 2.0
    assert masked.loc[pd.Timestamp("2020-01-03"), "aaa"] == 3.0


def test_tr_audit_flags_continuity_failure(tmp_path: Path):
    dates = pd.date_range("2020-02-03", periods=6, freq="B")
    # raw: großer Sprung am 5.2. -> simuliertes Split-Event
    raw = pd.DataFrame(
        {"AAA": [100, 102, 104, 210, 212, 214]}, index=dates, dtype=float
    )
    # adjusted falsch (keine Korrektur durchgeführt) => Audit soll Continuity-Fail melden
    adj = raw.copy()

    splits = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "effective_date": [pd.Timestamp("2020-02-05")],
            "ratio": [2.0],  # z.B. 2:1
        }
    )

    issues, summary = audit_corporate_actions(
        raw, adj, splits=splits, tolerance_return=0.40
    )
    assert (issues["type"] == "continuity_fail").any()
    assert summary["n_issues"] >= 1
