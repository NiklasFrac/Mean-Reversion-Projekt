from __future__ import annotations

import os

import pytest

from universe import exchange_source as xs


def test_pre_filter_symbols_strips_prefix_suffix_and_regex():
    symbols = ["$ABC", "XYZ-W", "GOOD", "BAD^X"]
    out = xs.pre_filter_symbols(
        symbols,
        drop_prefixes=["$"],
        drop_suffixes=["-W"],
        drop_regex=[r"\^"],
        drop_contains=["BAD"],
    )
    assert out == ["GOOD"]


def test_load_exchange_tickers_reads_screener_and_filters(tmp_path):
    screener = tmp_path / "nasdaq_screener_2024.csv"
    screener.write_text(
        "Symbol,Name\nAAA,FirmA\nTESTX,FirmB\nBBB-WS,FirmC\n", encoding="utf-8"
    )

    tickers = xs.load_exchange_tickers(
        filters_cfg={
            "drop_prefixes": ["TEST"],
            "drop_suffixes": ["-WS"],
            "drop_contains": ["^"],
        },
        universe_cfg={"screener_glob": str(screener)},
    )
    assert tickers == ["AAA"]


def test_load_exchange_tickers_errors_when_missing_screener(tmp_path):
    with pytest.raises(RuntimeError):
        xs.load_exchange_tickers(
            filters_cfg={},
            universe_cfg={"screener_glob": str(tmp_path / "nasdaq_screener_*.csv")},
        )


def test_load_exchange_tickers_raises_on_ambiguous_glob_by_default(tmp_path):
    first = tmp_path / "nasdaq_screener_one.csv"
    second = tmp_path / "nasdaq_screener_two.csv"
    first.write_text("Symbol,Name\nAAA,FirmA\n", encoding="utf-8")
    second.write_text("Symbol,Name\nBBB,FirmB\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Ambiguous universe.screener_glob"):
        xs.load_exchange_tickers(
            filters_cfg={},
            universe_cfg={"screener_glob": str(tmp_path / "nasdaq_screener_*.csv")},
        )


def test_load_exchange_tickers_latest_mtime_mode_selects_newest(tmp_path):
    first = tmp_path / "nasdaq_screener_old.csv"
    second = tmp_path / "nasdaq_screener_new.csv"
    first.write_text("Symbol,Name\nAAA,FirmA\n", encoding="utf-8")
    second.write_text("Symbol,Name\nBBB,FirmB\n", encoding="utf-8")
    os.utime(first, (1_700_000_000, 1_700_000_000))
    os.utime(second, (1_800_000_000, 1_800_000_000))

    tickers = xs.load_exchange_tickers(
        filters_cfg={},
        universe_cfg={
            "screener_glob": str(tmp_path / "nasdaq_screener_*.csv"),
            "screener_selection_mode": "latest_mtime",
        },
    )

    assert tickers == ["BBB"]


def test_is_common_equity_allows_trust_common_stock():
    row = {"Name": "Acadia Realty Trust Common Stock", "Type": ""}
    assert xs._is_common_equity(row) is True


def test_is_common_equity_rejects_trust_preferred():
    row = {"Name": "BancFirst Cumulative Trust Preferred Securities", "Type": ""}
    assert xs._is_common_equity(row) is False


def test_is_common_equity_rejects_beneficial_interest_trust():
    row = {
        "Name": "BlackRock Capital Allocation Term Trust Common Shares of Beneficial Interest",
        "Type": "",
    }
    assert xs._is_common_equity(row) is False


def test_load_exchange_tickers_keeps_trust_common_but_drops_trust_preferred(tmp_path):
    screener = tmp_path / "nasdaq_screener_2024.csv"
    screener.write_text(
        (
            "Symbol,Name\n"
            "AAT,American Assets Trust Inc. Common Stock\n"
            "BANFP,BancFirst Cumulative Trust Preferred Securities\n"
            "BCAT,BlackRock Capital Allocation Term Trust Common Shares of Beneficial Interest\n"
        ),
        encoding="utf-8",
    )

    tickers = xs.load_exchange_tickers(
        filters_cfg={},
        universe_cfg={"screener_glob": str(screener)},
    )

    assert tickers == ["AAT"]
