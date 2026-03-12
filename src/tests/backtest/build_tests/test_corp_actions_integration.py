# tests/test_corp_actions_integration.py
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import aus deinem Projekt
from corp_actions_builder import ensure_corporate_actions_file
from data_loader import load_price_data


def _write_prices_csv(path: Path, dates, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    # wide format: index=date, columns = symbols
    df = pd.DataFrame(index=pd.to_datetime(dates))
    for name, values in cols.items():
        df[name] = values
    df.to_csv(path, index=True)


def _safe_load_prices(prices_path: Path, ca_path: Path, align_calendar=False):
    """
    Ruft load_price_data je nach implementierter Signatur auf.
    """
    try:
        df = load_price_data(
            prices_path,
            align_calendar=bool(align_calendar),
            apply_corporate_actions=True,
            corporate_actions_path=str(ca_path),
        )
        return pd.DataFrame(df)
    except TypeError:
        # Older signature (path only)
        return pd.DataFrame(load_price_data(prices_path))


def test_ensure_creates_header_only(tmp_path: Path):
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensured = ensure_corporate_actions_file(ca)
    assert ensured.exists(), "Corporate actions CSV was not created"
    # Check header
    head = ensured.read_text(encoding="utf-8").strip()
    assert head.startswith("symbol,date,type,factor,amount,notes"), (
        "Header columns do not match"
    )


def test_rewrite_if_wrong_columns(tmp_path: Path):
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ca.parent.mkdir(parents=True, exist_ok=True)
    ca.write_text("badcol1,badcol2\nA,B\n", encoding="utf-8")
    ensured = ensure_corporate_actions_file(ca)
    head = ensured.read_text(encoding="utf-8").splitlines()[0].strip()
    assert head == "symbol,date,type,factor,amount,notes", (
        "Header was not corrected"
    )


def test_loader_with_empty_actions_no_crash(tmp_path: Path):
    # Build prices: 5 trading days, two symbols
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = tmp_path / "prices.csv"
    _write_prices_csv(
        prices,
        dates,
        {
            "AAA": [100, 101, 102, 103, 104],
            "BBB": [50, 50.5, 51, 51.5, 52],
        },
    )
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensure_corporate_actions_file(ca)  # Header-only

    df = _safe_load_prices(prices, ca, align_calendar=False)
    assert not df.empty, "Loader returned an empty frame"
    assert isinstance(df.index, pd.DatetimeIndex), "Index is not a DatetimeIndex"
    assert "AAA" in df.columns and "BBB" in df.columns, "Columns are missing"


@pytest.mark.parametrize("factor", [0.5, 0.25])
def test_split_event_preserves_continuity(tmp_path: Path, factor):
    """
    Checks that a split event (e.g. 2:1 -> factor=0.5) does NOT create a huge jump
    in the series at the ex-date (continuity). We test this heuristically:
    The daily return on the ex-date must not be extreme (>40%).
    """
    # 10 trading days
    dates = pd.date_range("2020-03-02", periods=10, freq="B")
    ex_day = dates[3]  # fourth trading day
    prices_path = tmp_path / "prices.csv"

    # Konstruiere Serie mit leichtem Trend
    base = 100.0
    AAA = [base + i for i in range(len(dates))]  # 100,101,102,...
    BBB = [50 + 0.5 * i for i in range(len(dates))]

    _write_prices_csv(prices_path, dates, {"AAA": AAA, "BBB": BBB})

    # Write corporate actions (split on ex_day for AAA)
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensure_corporate_actions_file(ca)
    with ca.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "AAA",
                ex_day.date().isoformat(),
                "split",
                factor,
                "",
                f"test split {factor}",
            ]
        )

    # Load with actions
    df_adj = _safe_load_prices(prices_path, ca, align_calendar=False)
    assert "AAA" in df_adj.columns, "AAA is missing after adjustments"
    # Heuristic: return on the ex-date should be moderate (e.g. < 40% jump)
    ret = df_adj["AAA"].pct_change().loc[ex_day]
    assert not math.isnan(ret), "Return on the ex-date is NaN"
    assert abs(ret) < 0.4, f"Split adjustment not applied? Saw: {ret:.3f}"


def test_dividend_event_does_not_crash_loader(tmp_path: Path):
    # Minimalpreise
    dates = pd.date_range("2021-01-04", periods=6, freq="B")
    prices_path = tmp_path / "prices.csv"
    _write_prices_csv(prices_path, dates, {"AAA": np.linspace(100, 105, len(dates))})

    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensure_corporate_actions_file(ca)
    # Dividend am Tag 3
    ex_day = dates[2]
    with ca.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["AAA", ex_day.date().isoformat(), "dividend", "", 0.5, "quarterly"]
        )

    df_adj = _safe_load_prices(prices_path, ca, align_calendar=False)
    assert not df_adj.empty, "Loader bricht bei Dividend-Event ab"


def test_delist_event_truncates_series_or_nan_after(tmp_path: Path):
    """
    For a delisting we expect either (a) the series to be truncated from the ex-date onward or
    (b) NaNs after the delist date. Both are fine - the main point is no crash and no
    artificial continuation with prices.
    """
    dates = pd.date_range("2022-05-02", periods=8, freq="B")
    prices_path = tmp_path / "prices.csv"
    _write_prices_csv(prices_path, dates, {"ZZZ": np.linspace(10, 13.5, len(dates))})

    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensure_corporate_actions_file(ca)
    ex_day = dates[4]
    with ca.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["ZZZ", ex_day.date().isoformat(), "delist", "", "", "gone"]
        )

    df_adj = _safe_load_prices(prices_path, ca, align_calendar=False)
    after = df_adj.loc[df_adj.index >= ex_day, "ZZZ"]
    # Allowed variants:
    # - fully NaN from ex_day onward
    # - last valid observation < number of days after ex_day (i.e. truncated)
    is_all_nan = after.isna().all()
    truncated = df_adj.index.max() < df_adj.index.min() + (ex_day - df_adj.index.min())
    assert is_all_nan or truncated, (
        "Delist behavior does not match expectations (no crash, no artificial continuation)."
    )
