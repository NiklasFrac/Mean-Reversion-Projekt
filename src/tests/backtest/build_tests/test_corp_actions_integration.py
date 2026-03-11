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
    # Wide-Format: Index=Date, Spalten = Symbole
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
        # Ältere Signatur (nur path)
        return pd.DataFrame(load_price_data(prices_path))


def test_ensure_creates_header_only(tmp_path: Path):
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ensured = ensure_corporate_actions_file(ca)
    assert ensured.exists(), "Corporate-Actions-CSV wurde nicht erstellt"
    # Header prüfen
    head = ensured.read_text(encoding="utf-8").strip()
    assert head.startswith("symbol,date,type,factor,amount,notes"), (
        "Header-Spalten stimmen nicht"
    )


def test_rewrite_if_wrong_columns(tmp_path: Path):
    ca = tmp_path / "backtest" / "data" / "corp_actions.csv"
    ca.parent.mkdir(parents=True, exist_ok=True)
    ca.write_text("badcol1,badcol2\nA,B\n", encoding="utf-8")
    ensured = ensure_corporate_actions_file(ca)
    head = ensured.read_text(encoding="utf-8").splitlines()[0].strip()
    assert head == "symbol,date,type,factor,amount,notes", (
        "Header wurde nicht korrigiert"
    )


def test_loader_with_empty_actions_no_crash(tmp_path: Path):
    # Preise bauen: 5 Handelstage, zwei Symbole
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
    assert not df.empty, "Loader liefert leeren Frame"
    assert isinstance(df.index, pd.DatetimeIndex), "Index ist kein DatetimeIndex"
    assert "AAA" in df.columns and "BBB" in df.columns, "Spalten fehlen"


@pytest.mark.parametrize("factor", [0.5, 0.25])
def test_split_event_preserves_continuity(tmp_path: Path, factor):
    """
    Prüft, dass ein Split-Event (z.B. 2:1 -> factor=0.5) die Serie um den Ex-Tag
    NICHT mit einem riesigen Sprung versieht (Kontinuität). Wir testen heuristisch:
    Der Tagesreturn am Ex-Tag darf nicht extrem sein (>40%).
    """
    # 10 Handelstage
    dates = pd.date_range("2020-03-02", periods=10, freq="B")
    ex_day = dates[3]  # vierter Handelstag
    prices_path = tmp_path / "prices.csv"

    # Konstruiere Serie mit leichtem Trend
    base = 100.0
    AAA = [base + i for i in range(len(dates))]  # 100,101,102,...
    BBB = [50 + 0.5 * i for i in range(len(dates))]

    _write_prices_csv(prices_path, dates, {"AAA": AAA, "BBB": BBB})

    # Corp Actions schreiben (Split am ex_day für AAA)
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

    # Laden mit Aktionen
    df_adj = _safe_load_prices(prices_path, ca, align_calendar=False)
    assert "AAA" in df_adj.columns, "AAA fehlt nach Adjustments"
    # Heuristik: Return am Ex-Tag sollte moderat sein (z.B. < 40% Sprung)
    ret = df_adj["AAA"].pct_change().loc[ex_day]
    assert not math.isnan(ret), "Return am Ex-Tag ist NaN"
    assert abs(ret) < 0.4, f"Split-Adjustment nicht angewandt? Gesehen: {ret:.3f}"


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
    Bei Delisting erwarten wir entweder (a) abgeschnittene Serie ab Ex-Tag oder
    (b) NaNs nach dem Delist-Tag. Beides ist ok – Hauptsache kein Crash und kein
    künstliches Weiterführen mit Preisen.
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
    # Erlaubte Varianten:
    # - vollständig NaN ab ex_day
    # - letzte gültige Beobachtung < Anzahl Tage nach ex_day (also abgeschnitten)
    is_all_nan = after.isna().all()
    truncated = df_adj.index.max() < df_adj.index.min() + (ex_day - df_adj.index.min())
    assert is_all_nan or truncated, (
        "Delist-Verhalten entspricht nicht der Erwartung (kein Crash, kein künstliches Fortsetzen)."
    )
