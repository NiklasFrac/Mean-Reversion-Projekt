from __future__ import annotations

import os

import pandas as pd
import pytest

from universe import checkpoint, filters


def test_apply_filters_computes_free_float_and_thresholds():
    df = pd.DataFrame(
        {
            "price": [10.0, 5.0, 12.0],
            "market_cap": [1e9, 1e8, 1e9],
            "volume": [1_000_000, 500_000, 750_000],
            "float_pct": [0.6, 0.1, 0.5],
            "dividend": [True, True, True],
            "is_etf": [False, False, False],
            "shares_out": [200_000_000, 900_000_000, 50_000_000],
        },
        index=["PASS", "FAIL_MCAP", "FAIL_SHARES"],
    )

    filt_cfg = {"min_free_float_shares": 80_000_000, "min_free_float_dollar_cap": 5e8}
    filtered, reasons = filters._apply_filters_with_reasons(df, filt_cfg)

    assert list(filtered.index) == ["PASS"]
    assert reasons["REASON_FF_SHARES"] == 1
    assert reasons["REASON_FF_DOLLAR_CAP"] == 1
    assert filtered.loc["PASS", "free_float_shares"] == pytest.approx(120_000_000.0)


def test_whitelist_re_adds_symbol_dropped_by_core_filter():
    df = pd.DataFrame(
        {"price": [10.0, 0.0], "market_cap": [1e9, 1e9], "volume": [1.0, 1.0]},
        index=["AAA", "BBB"],
    )
    cfg = {"symbol_whitelist": ["BBB"], "drop_zero": True, "drop_na": True}

    filtered, _ = filters._apply_filters_with_reasons(df, cfg)

    assert "BBB" in filtered.index
    assert filtered.loc["BBB", "price"] == 0.0


def test_whitelist_re_add_preserves_derived_columns():
    df = pd.DataFrame(
        {
            "price": [25.0, 10.0],
            "market_cap": [2e9, 1e9],
            "volume": [2_000_000.0, 100.0],
            "float_pct": [0.5, 0.4],
            "shares_out": [200_000_000.0, 150_000_000.0],
        },
        index=["KEEP", "READD"],
    )
    cfg = {
        "drop_zero": True,
        "drop_na": True,
        "min_price": 20.0,
        "min_dollar_adv": 1_000_000.0,
        "symbol_whitelist": ["READD"],
    }

    filtered, _ = filters._apply_filters_with_reasons(df, cfg)

    assert "READD" in filtered.index
    assert float(filtered.loc["READD", "price_eff"]) == pytest.approx(10.0)
    assert float(filtered.loc["READD", "dollar_adv_eff"]) == pytest.approx(1000.0)
    assert float(filtered.loc["READD", "free_float_shares"]) == pytest.approx(
        60_000_000.0
    )
    assert float(filtered.loc["READD", "free_float_mcap"]) == pytest.approx(
        400_000_000.0
    )


def test_whitelist_duplicate_entries_do_not_duplicate_output_rows():
    df = pd.DataFrame(
        {
            "price": [25.0, 10.0],
            "market_cap": [1e9, 1e9],
            "volume": [1_000_000.0, 100.0],
        },
        index=["KEEP", "READD"],
    )
    cfg = {
        "drop_zero": True,
        "drop_na": True,
        "min_price": 20.0,
        "symbol_whitelist": ["READD", "READD"],
    }

    filtered, _ = filters._apply_filters_with_reasons(df, cfg)

    assert filtered.index.is_unique
    assert list(filtered.index) == ["KEEP", "READD"]


def test_blacklist_uses_normalized_symbol_matching():
    df = pd.DataFrame(
        {"price": [300.0], "market_cap": [1e11], "volume": [1_000_000.0]},
        index=["BRK-B"],
    )
    cfg = {"symbol_blacklist": ["BRK.B"], "drop_zero": False, "drop_na": False}

    filtered, reasons = filters._apply_filters_with_reasons(df, cfg)

    assert filtered.empty
    assert reasons["REASON_BLACKLISTED"] == 1


def test_whitelist_readd_uses_normalized_symbol_matching():
    df = pd.DataFrame(
        {"price": [300.0], "market_cap": [1e11], "volume": [1_000_000.0]},
        index=["BRK-B"],
    )
    cfg = {
        "min_price": 500.0,
        "symbol_whitelist": ["BRK.B"],
        "drop_zero": False,
        "drop_na": False,
    }

    filtered, reasons = filters._apply_filters_with_reasons(df, cfg)

    assert "BRK-B" in filtered.index
    assert reasons["REASON_LOW_PRICE"] == 1


def test_price_eff_requires_warmup_when_column_present():
    df = pd.DataFrame(
        {
            "price": [10.0, 12.0],
            "market_cap": [1e9, 1e9],
            "volume": [1_000_000, 1_000_000],
            "price_warmup_med": [pd.NA, 12.0],
        },
        index=["A", "B"],
    )
    cfg = {"min_price": 5.0, "drop_zero": False, "drop_na": False}

    filtered, reasons = filters._apply_filters_with_reasons(df, cfg)

    assert list(filtered.index) == ["B"]
    assert reasons["REASON_LOW_PRICE"] == 1
    assert float(filtered.loc["B", "price_eff"]) == pytest.approx(12.0)


def test_drop_na_uses_effective_warmup_fields_not_snapshot():
    df = pd.DataFrame(
        {
            "price": [pd.NA, 10.0],
            "market_cap": [1e9, 1e9],
            "volume": [pd.NA, 1_000_000.0],
            "price_warmup_med": [12.0, pd.NA],
            "volume_warmup_avg": [2_000_000.0, 2_000_000.0],
        },
        index=["A", "B"],
    )
    cfg = {"drop_na": True, "drop_zero": False}

    filtered, reasons = filters._apply_filters_with_reasons(df, cfg)

    assert list(filtered.index) == ["A"]
    assert reasons["REASON_NA_CORE_FIELD"] == 1


def test_drop_zero_uses_effective_warmup_fields_not_snapshot():
    df = pd.DataFrame(
        {
            "price": [10.0, 10.0],
            "market_cap": [1e9, 1e9],
            "volume": [0.0, 1_000_000.0],
            "price_warmup_med": [10.0, 10.0],
            "volume_warmup_avg": [500_000.0, 0.0],
        },
        index=["KEEP_ON_WARMUP", "DROP_ON_WARMUP"],
    )
    cfg = {"drop_na": False, "drop_zero": True}

    filtered, reasons = filters._apply_filters_with_reasons(df, cfg)

    assert list(filtered.index) == ["KEEP_ON_WARMUP"]
    assert reasons["REASON_ZERO_CORE_FIELD"] == 1


def test_norm_symbol_converts_unwanted_dots_and_keeps_whitelisted_suffixes():
    assert checkpoint.norm_symbol("abc.mx") == "ABC-MX"
    assert checkpoint.norm_symbol("abc.to") == "ABC.TO"
    assert checkpoint.norm_symbol("a.b.c") == "A-B-C"


def test_checkpointer_persistence_and_pruning(tmp_path):
    path = tmp_path / "ckpt.json"
    cp = checkpoint.Checkpointer(path)
    cp.load()
    cp.mark_done("abc.mx", cfg_hash="cfg1", timestamp=1.0)
    cp.mark_done("xyz.to", cfg_hash="cfg2", timestamp=2.0)
    cp.store_symbol_seed(["abc.mx", "xyz.to"], cfg_hash="cfg2")

    assert cp.is_done("ABC-MX", cfg_hash="cfg1")
    assert cp.symbol_seed(cfg_hash="cfg2") == ["ABC-MX", "XYZ.TO"]

    removed = cp.retain_only(["abc.mx"])
    assert removed == 1
    cp.drop_many(["abc.mx"])
    assert not cp.entries()


def test_checkpointer_mark_done_many_persists_once(tmp_path, monkeypatch):
    path = tmp_path / "ckpt_many.json"
    cp = checkpoint.Checkpointer(path)
    cp.load()

    calls: list[int] = []
    monkeypatch.setattr(cp, "_persist", lambda: calls.append(1))

    updated = cp.mark_done_many(
        ["AAA", "BBB"],
        cfg_hash="cfgX",
        timestamps={"AAA": 1.5},
        default_timestamp=2.5,
    )

    assert updated == 2
    assert len(calls) == 1
    entries = cp.entries()
    assert entries["AAA"]["ts"] == pytest.approx(1.5)
    assert entries["BBB"]["ts"] == pytest.approx(2.5)
    assert entries["AAA"]["cfg_hash"] == "cfgX"
    assert entries["BBB"]["cfg_hash"] == "cfgX"


def test_checkpointer_failed_symbols_roundtrip(tmp_path):
    path = tmp_path / "ckpt_failed.json"
    cp = checkpoint.Checkpointer(path)
    cp.load()
    cp.mark_done_many(["AAA"], cfg_hash="cfgA", default_timestamp=10.0)
    cp.mark_failed_many(["BAD1", "BAD2"], cfg_hash="cfgA", default_timestamp=11.0)

    failed = cp.failed_symbols(cfg_hash="cfgA")
    assert failed == {"BAD1", "BAD2"}

    cp2 = checkpoint.Checkpointer(path)
    cp2.load()
    failed2 = cp2.failed_symbols(cfg_hash="cfgA")
    assert failed2 == {"BAD1", "BAD2"}
    assert cp2.is_done("BAD1", cfg_hash="cfgA")


def test_checkpointer_logs_when_primary_replace_locked(tmp_path, monkeypatch, caplog):
    path = tmp_path / "ckpt_lock.json"
    cp = checkpoint.Checkpointer(path)
    cp.load()

    real_replace = os.replace
    calls = {"n": 0}

    def fake_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if dst == str(path) and calls["n"] <= 8:
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fake_replace)

    with caplog.at_level("WARNING", logger="runner_universe"):
        cp.mark_done("AAA", cfg_hash="cfg", timestamp=1.0)

    backups = list(tmp_path.glob("ckpt_lock.json.bak_*"))
    assert backups
    assert any(
        "Checkpoint persist fallback used" in rec.message for rec in caplog.records
    )


def test_min_float_pct_treat_missing_flag():
    df = pd.DataFrame(
        {
            "price": [10.0, 10.0],
            "market_cap": [1e9, 1e9],
            "volume": [1_000_000, 1_000_000],
            "float_pct": [pd.NA, 0.2],
            "dividend": [True, True],
            "is_etf": [False, False],
        },
        index=["MISSING", "LOWFLOAT"],
    )

    cfg_default = {"min_float_pct": 0.3, "drop_na": True, "drop_zero": True}
    kept_default, _ = filters._apply_filters_with_reasons(df, cfg_default)
    assert "MISSING" not in kept_default.index  # default now strict on missing float

    cfg_lenient = {**cfg_default, "treat_missing_float_as_pass": True}
    kept_lenient, reasons = filters._apply_filters_with_reasons(df, cfg_lenient)
    assert "MISSING" in kept_lenient.index
    assert reasons["REASON_FLOAT_PCT"] >= 1  # LOWFLOAT dropped


def test_filter_boolean_flags_parse_string_values():
    df = pd.DataFrame(
        {
            "price": [0.0],
            "market_cap": [pd.NA],
            "volume": [1_000_000.0],
            "dividend": [False],
        },
        index=["AAA"],
    )
    cfg = {
        "drop_na": "false",
        "drop_zero": "false",
        "require_dividend": "false",
    }

    filtered, _ = filters._apply_filters_with_reasons(df, cfg)

    assert list(filtered.index) == ["AAA"]
