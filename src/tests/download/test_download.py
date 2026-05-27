from __future__ import annotations

import pandas as pd
import pytest
import yaml

from download import runner_download as dl


def test_config_requires_iso_dates_and_paths(tmp_path) -> None:
    cfg = {
        "input": {"screener_path": str(tmp_path / "s.csv"), "symbol_col": "Symbol"},
        "download": {"start": "today", "end": "2026-01-01"},
        "quality": {"min_coverage": 0.95, "max_gap": 3, "min_final_tickers": 1},
        "output": {
            "raw_close": "a.csv",
            "filled_close": "b.csv",
            "dropped_tickers": "c.csv",
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        dl._load_cfg(path)


def test_screener_filters_hard_etf_column_only(tmp_path, caplog) -> None:
    screener = tmp_path / "s.csv"
    pd.DataFrame(
        {"Symbol": [" aapl ", "brk.b", "spy"], "ETF": ["", "", "true"]}
    ).to_csv(screener, index=False)
    cfg = {
        "input": {
            "screener_path": screener,
            "symbol_col": "Symbol",
            "etf_filter": {"enabled": True, "column": None, "exclude_values": ["true"]},
        }
    }
    symbols, dropped = dl._read_symbols(cfg)
    assert symbols == ["AAPL", "BRK-B"]
    assert dropped == [{"ticker": "SPY", "reason": "etf_filtered"}]

    pd.DataFrame({"Symbol": ["ABC"], "Name": ["ETF Words Ignored"]}).to_csv(
        screener, index=False
    )
    symbols, dropped = dl._read_symbols(cfg)
    assert symbols == ["ABC"]
    assert dropped == []
    assert "ETF filter skipped" in caplog.text


def test_screener_filters_configured_structured_securities(tmp_path) -> None:
    screener = tmp_path / "s.csv"
    pd.DataFrame(
        {
            "Symbol": ["AAPL", "AACBR", "AACIU", "AACI", "GPCR", "KTN", "ABR^D"],
            "Name": [
                "Apple Inc. Common Stock",
                "Artius II Acquisition Inc. Rights",
                "Artius II Acquisition Inc. Units",
                "Armada Acquisition Corp. III Class A Ordinary Share",
                "Structure Therapeutics Inc. American Depositary Shares",
                "Structured Products Corp Trust Securities",
                "Arbor Realty Trust Preferred Stock",
            ],
            "Industry": [
                "Computer Manufacturing",
                "",
                "",
                "Blank Checks",
                "Biotechnology: Pharmaceutical Preparations",
                "Finance: Consumer Services",
                "",
            ],
        }
    ).to_csv(screener, index=False)
    cfg = {
        "input": {
            "screener_path": screener,
            "symbol_col": "Symbol",
            "security_filter": {
                "enabled": True,
                "name_column": "Name",
                "exclude_name_patterns": [
                    r"\bRights?\b",
                    r"\bUnits?\b",
                    r"\bStructured Products?\b",
                    r"\bPreferred\b",
                    r"\bTrust Securities\b",
                ],
                "exclude_symbol_patterns": [r"\^"],
                "exclude_column_values": {"Industry": ["Blank Checks"]},
            },
        }
    }
    symbols, dropped = dl._read_symbols(cfg)
    assert symbols == ["AAPL", "GPCR"]
    assert {r["ticker"]: r["reason"] for r in dropped} == {
        "AACBR": "security_filtered",
        "AACIU": "security_filtered",
        "AACI": "security_filtered",
        "KTN": "security_filtered",
        "ABR^D": "security_filtered",
    }


def test_download_extracts_close_and_marks_missing(monkeypatch) -> None:
    calls = {"n": 0}
    idx = pd.date_range("2024-01-01", periods=2)

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()
        cols = pd.MultiIndex.from_product([["Close"], ["AAA"]])
        return pd.DataFrame([[1.0], [2.0]], index=idx, columns=cols)

    monkeypatch.setattr(dl.yf, "download", fake_download)
    raw, missing = dl._download_close(
        ["AAA", "BBB"],
        {
            "start": "2024-01-01",
            "end": "2024-01-03",
            "batch_size": 2,
            "retries": 1,
            "pause_seconds": 0.0,
        },
    )
    assert list(raw.columns) == ["AAA"]
    assert missing == ["BBB"]


def test_quality_filters_and_fills() -> None:
    idx = pd.date_range("2024-01-01", periods=6, tz="UTC")
    raw = pd.DataFrame(
        {
            "OK": [1.0, 2.0, None, 4.0, 5.0, 6.0],
            "EDGE": [None, 2.0, 3.0, 4.0, 5.0, 6.0],
            "BAD": [1.0, 2.0, 0.0, 4.0, 5.0, 6.0],
            "GAP": [1.0, None, None, None, 5.0, 6.0],
        },
        index=idx,
    )
    filled, dropped = dl._quality(
        raw, {"min_coverage": 0.5, "max_gap": 2, "min_final_tickers": 1}
    )
    assert list(filled.columns) == ["OK"]
    assert filled["OK"].iloc[2] == 2.0
    assert {r["ticker"]: r["reason"] for r in dropped} == {
        "EDGE": "edge_gap",
        "BAD": "nonpositive_price",
        "GAP": "gap_too_large",
    }


def test_cli_smoke_writes_csvs(tmp_path, monkeypatch) -> None:
    screener = tmp_path / "s.csv"
    pd.DataFrame({"Symbol": ["AAA", "BBB"]}).to_csv(screener, index=False)
    cfg = {
        "input": {
            "screener_path": str(screener),
            "symbol_col": "Symbol",
            "etf_filter": {"enabled": False},
        },
        "download": {
            "start": "2024-01-01",
            "end": "2024-01-05",
            "batch_size": 10,
            "retries": 0,
            "pause_seconds": 0.0,
        },
        "quality": {"min_coverage": 1.0, "max_gap": 1, "min_final_tickers": 1},
        "output": {
            "raw_close": str(tmp_path / "raw.csv"),
            "filled_close": str(tmp_path / "filled.csv"),
            "dropped_tickers": str(tmp_path / "dropped.csv"),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def fake_download(*args, **kwargs):
        idx = pd.date_range("2024-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
        return pd.DataFrame([[1, 2], [2, 3], [3, 4]], index=idx, columns=cols)

    monkeypatch.setattr(dl.yf, "download", fake_download)
    dl.run_download(cfg_path)
    assert (tmp_path / "raw.csv").exists()
    assert (tmp_path / "filled.csv").exists()
    assert (tmp_path / "dropped.csv").exists()
