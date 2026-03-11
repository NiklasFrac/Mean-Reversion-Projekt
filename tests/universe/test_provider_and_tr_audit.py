from __future__ import annotations

import pandas as pd
import pytest

from universe.tr_audit import audit_corporate_actions
from universe.universe_provider import apply_universe, load_universe


def test_load_universe_symbols_normalizes_and_deduplicates(tmp_path):
    csv = tmp_path / "universe_symbols.csv"
    csv.write_text("ticker\nbrk.b\nBRK/B\n\n", encoding="utf-8")

    schema, df = load_universe(csv)

    assert schema == "symbols"
    assert df.to_dict(orient="records") == [{"symbol": "BRK-B"}]


def test_load_universe_interval_normalizes_dates(tmp_path):
    csv = tmp_path / "universe_interval.csv"
    csv.write_text(
        "symbol,start_date,end_date\n"
        "aaa,2024-01-02T10:00:00-05:00,2024-01-04\n"
        ",2024-01-01,2024-01-10\n",
        encoding="utf-8",
    )

    schema, df = load_universe(csv)

    assert schema == "interval"
    assert df.shape[0] == 1
    assert df.loc[0, "symbol"] == "AAA"
    assert df.loc[0, "start_date"] == pd.Timestamp("2024-01-02")
    assert df.loc[0, "end_date"] == pd.Timestamp("2024-01-04")


def test_apply_universe_symbols_masks_unlisted_symbols():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0], "BBB": [10.0, 11.0, 12.0]}, index=idx
    )
    universe_df = pd.DataFrame({"symbol": ["AAA"]})

    out = apply_universe(prices, "symbols", universe_df)

    assert out["AAA"].tolist() == [1.0, 2.0, 3.0]
    assert out["BBB"].isna().all()


def test_apply_universe_interval_masks_outside_windows_and_other_symbols():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"AAA": [1, 2, 3, 4, 5], "BBB": [11, 12, 13, 14, 15]}, index=idx
    )
    universe_df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "start_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")],
            "end_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")],
        }
    )

    out = apply_universe(prices, "interval", universe_df)

    assert out["AAA"].isna().tolist() == [True, False, False, True, False]
    assert out.loc[idx[1], "AAA"] == 2.0
    assert out.loc[idx[2], "AAA"] == 3.0
    assert out.loc[idx[4], "AAA"] == 5.0
    assert out["BBB"].isna().all()


def test_audit_corporate_actions_empty_without_splits():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"AAA": [100.0, 50.0, 52.0]}, index=idx)
    adj = pd.DataFrame({"AAA": [50.0, 50.0, 52.0]}, index=idx)

    issues, summary = audit_corporate_actions(raw, adj, splits=None)

    assert issues.empty
    assert summary == {"n_issues": 0, "types": {}}


def test_audit_corporate_actions_requires_expected_split_columns():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    prices = pd.DataFrame({"AAA": [10.0, 11.0]}, index=idx)
    bad_splits = pd.DataFrame({"symbol": ["AAA"], "effective_date": ["2024-01-02"]})

    with pytest.raises(ValueError, match="missing required columns"):
        audit_corporate_actions(prices, prices, splits=bad_splits)


def test_audit_corporate_actions_flags_large_adjusted_discontinuity():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"AAA": [100.0, 50.0, 52.0]}, index=idx)
    # A split day should be roughly continuous in adjusted series; this is intentionally broken.
    adj = pd.DataFrame({"AAA": [50.0, 20.0, 21.0]}, index=idx)
    splits = pd.DataFrame(
        {"symbol": ["AAA"], "effective_date": ["2024-01-02"], "ratio": [2.0]}
    )

    issues, summary = audit_corporate_actions(
        raw, adj, splits=splits, tolerance_return=0.4
    )

    assert not issues.empty
    assert summary["n_issues"] == 1
    row = issues.iloc[0]
    assert row["type"] == "continuity_fail"
    assert row["symbol"] == "AAA"
    assert row["segment"] == "pre_to_event"
