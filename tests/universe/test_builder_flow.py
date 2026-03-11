from __future__ import annotations

import pandas as pd
import pytest

from universe import builder


def _fake_funda_df(symbols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ticker": symbols,
            "price": [10.0 + i for i in range(len(symbols))],
            "market_cap": [1e9 + i * 1e8 for i in range(len(symbols))],
            "volume": [1_000_000 + i * 10_000 for i in range(len(symbols))],
            "float_pct": [0.5] * len(symbols),
            "dividend": [True] * len(symbols),
            "is_etf": [False] * len(symbols),
            "shares_out": [200_000_000] * len(symbols),
        }
    ).set_index("ticker")
    return df


def _patch_adv(monkeypatch):
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )

    def _fake_adv_map(tickers, *a, **k):
        adv = pd.DataFrame(
            {
                "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                "volume_warmup_avg": pd.Series({t: 1_000_000.0 for t in tickers}),
                "adv_window": float(2),
                "adv_asof": pd.Timestamp("2024-01-02"),
            }
        )
        adv.index.name = "ticker"
        meta = {"fingerprint": "TEST"}
        return (adv, meta) if k.get("return_meta", False) else adv

    monkeypatch.setattr(builder, "load_or_compute_adv_map", _fake_adv_map)


def test_merge_fundamentals_prefers_fresh_and_sets_updated():
    existing = pd.DataFrame(
        {"price": [1.0], "market_cap": [1e6], "volume": [10_000]},
        index=pd.Index(["AAA"], name="ticker"),
    )
    fresh = pd.DataFrame(
        {"price": [2.0], "market_cap": [2e6], "volume": [20_000]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    merged = builder._merge_fundamentals_frames(existing, fresh)

    assert merged.loc["AAA", "price"] == 2.0
    assert "updated_at" in merged.columns
    assert merged.index.name == "ticker"


def test_merge_fundamentals_keeps_existing_values_when_fresh_is_nan():
    existing = pd.DataFrame(
        {
            "price": [100.0],
            "market_cap": [1e9],
            "volume": [1_000_000.0],
            "float_pct": [0.6],
        },
        index=pd.Index(["AAA"], name="ticker"),
    )
    fresh = pd.DataFrame(
        {
            "price": [101.0],
            "market_cap": [float("nan")],
            "volume": [1_100_000.0],
            "float_pct": [float("nan")],
        },
        index=pd.Index(["AAA"], name="ticker"),
    )

    merged = builder._merge_fundamentals_frames(existing, fresh)

    assert float(merged.loc["AAA", "price"]) == pytest.approx(101.0)
    assert float(merged.loc["AAA", "volume"]) == pytest.approx(1_100_000.0)
    assert float(merged.loc["AAA", "market_cap"]) == pytest.approx(1e9)
    assert float(merged.loc["AAA", "float_pct"]) == pytest.approx(0.6)


def test_normalize_seed_symbols_reuses_shared_normalizer():
    out = builder._normalize_seed_symbols(
        [" aaa ", "AAA", "brk.b", "BRK-B", "^BAD", "TESTX"],
        junk_filter=lambda s: str(s).startswith("^") or str(s).startswith("TEST"),
    )
    assert out == ["AAA", "BRK-B"]


def test_build_universe_end_to_end_happy_path(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA", "BBB"])
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )

    def fake_fetch(**kwargs):
        df = _fake_funda_df(["AAA", "BBB"])
        return df, {"failed": []}

    monkeypatch.setattr(
        builder, "fetch_fundamentals_parallel", lambda **kwargs: fake_fetch(**kwargs)
    )
    _patch_adv(monkeypatch)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {"drop_zero": True, "drop_na": True},
        "runtime": {
            "checkpoint_path": None,
            "workers": 2,
            "canary": {"min_valid_tickers": 1},
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    df_universe, df_funda, monitoring, extra = builder.build_universe(
        cfg, cfg_path, run_id="RUN-1"
    )

    assert list(df_universe.index) == ["AAA", "BBB"]
    assert monitoring["failed"] == []
    assert extra["n_tickers_total"] == 2


def test_build_universe_adv_provenance_fingerprint_uses_adv_meta_for_partial_fundamentals(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(
        builder, "load_exchange_tickers", lambda **_: ["AAA", "BBB", "ZZZINVALID"]
    )
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    monkeypatch.setattr(
        builder,
        "fetch_fundamentals_parallel",
        lambda **kwargs: (_fake_funda_df(["AAA", "BBB"]), {"failed": []}),
    )
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )

    seen: dict[str, object] = {}

    def _fake_adv_map(tickers, *args, **kwargs):
        seen["adv_tickers"] = list(tickers)
        adv = pd.DataFrame(
            {
                "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                "volume_warmup_avg": pd.Series({t: 1_000_000.0 for t in tickers}),
                "adv_window": float(2),
                "adv_asof": pd.Timestamp("2024-01-02"),
            }
        )
        adv.index.name = "ticker"
        meta = {"fingerprint": "META-FP-PARTIAL"}
        return (adv, meta) if kwargs.get("return_meta", False) else adv

    monkeypatch.setattr(builder, "load_or_compute_adv_map", _fake_adv_map)

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError(
            "adv_fingerprint fallback should not run when adv_meta fingerprint exists"
        )

    monkeypatch.setattr(builder, "adv_fingerprint", _unexpected_fallback)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {"drop_zero": True, "drop_na": True},
        "runtime": {
            "checkpoint_path": None,
            "workers": 2,
            "canary": {"min_valid_tickers": 1},
        },
        "data": {"fundamentals_cache_enabled": False, "adv_window": 2},
    }

    _, _, _, extra = builder.build_universe(cfg, cfg_path, run_id="RUN-ADV-FP")

    assert seen.get("adv_tickers") == ["AAA", "BBB"]
    assert extra["n_tickers_total"] == 3
    assert extra["n_fundamentals_ok"] == 2
    assert extra["adv_provenance"]["fingerprint"] == "META-FP-PARTIAL"


def test_build_universe_raises_on_empty_seed(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: [])
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)

    cfg = {
        "universe": {"output_tickers_csv": "out.csv", "manifest": "man.json"},
        "filters": {},
        "runtime": {"checkpoint_path": None, "canary": {"min_valid_tickers": 1}},
        "data": {"fundamentals_cache_enabled": False},
    }

    with pytest.raises(RuntimeError, match="No seed tickers loaded"):
        builder.build_universe(cfg, cfg_path, run_id="RUN-EMPTY")


def test_build_universe_treats_null_checkpoint_path_as_disabled(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    monkeypatch.setattr(
        builder,
        "fetch_fundamentals_parallel",
        lambda **kwargs: (_fake_funda_df(["AAA"]), {"failed": []}),
    )
    _patch_adv(monkeypatch)

    created = {"n": 0}

    class _NeverInitCheckpointer:
        def __init__(self, *_args, **_kwargs):
            created["n"] += 1
            raise AssertionError(
                "Checkpointer must not be instantiated when checkpoint is disabled"
            )

    monkeypatch.setattr(builder, "Checkpointer", _NeverInitCheckpointer)

    cfg = {
        "universe": {
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {"checkpoint_path": "null", "canary": {"min_valid_tickers": 1}},
        "data": {"fundamentals_cache_enabled": False},
    }

    df_universe, _, _, _ = builder.build_universe(cfg, cfg_path, run_id="RUN-NO-CKPT")
    assert list(df_universe.index) == ["AAA"]
    assert created["n"] == 0


def test_build_universe_reuses_checkpoint_seed(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )
    cfg_hash = builder._sha1(cfg_path)

    cp_path = tmp_path / "ckpt.json"
    cp = builder.Checkpointer(cp_path)
    cp.store_symbol_seed(["AAA", "BBB"], cfg_hash=cfg_hash)

    calls: list[dict] = []

    def loader(**kwargs):
        calls.append(dict(kwargs))
        return ["AAA", "BBB"]

    monkeypatch.setattr(builder, "load_exchange_tickers", loader)
    monkeypatch.setattr(
        builder,
        "get_last_screener_meta",
        lambda: {"path": "runs/data/nasdaq_screener_test.csv", "sha1": "TESTSHA1"},
    )
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    monkeypatch.setattr(
        builder,
        "fetch_fundamentals_parallel",
        lambda **kwargs: (_fake_funda_df(["AAA", "BBB"]), {"failed": []}),
    )
    _patch_adv(monkeypatch)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": True,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    df_universe, _, _, extra = builder.build_universe(cfg, cfg_path, run_id="RUN-2")

    assert list(df_universe.index) == ["AAA", "BBB"]
    assert extra["tickers_all"] == ["AAA", "BBB"]
    assert extra.get("screener_provenance", {}).get("sha1") == "TESTSHA1"
    assert calls, "Expected screener CSV reload for provenance"

    cp2 = builder.Checkpointer(cp_path)
    cp2.load()
    entry = cp2.symbol_seed_entry(cfg_hash=cfg_hash)
    assert entry is not None
    assert entry.get("provenance", {}).get("sha1") == "TESTSHA1"


def test_build_universe_raises_when_checkpoint_seed_provenance_reload_fails(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )
    cfg_hash = builder._sha1(cfg_path)

    cp_path = tmp_path / "ckpt.json"
    cp = builder.Checkpointer(cp_path)
    cp.store_symbol_seed(["AAA"], cfg_hash=cfg_hash)

    def _boom(**kwargs):
        raise RuntimeError("screener unavailable")

    monkeypatch.setattr(builder, "load_exchange_tickers", _boom)
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)

    cfg = {
        "universe": {"output_tickers_csv": "out.csv", "manifest": "man.json"},
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": True,
            "canary": {"min_valid_tickers": 1},
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    with pytest.raises(RuntimeError, match="Screener CSV reload for provenance failed"):
        builder.build_universe(cfg, cfg_path, run_id="RUN-SEED-STRICT")


def test_build_universe_does_not_skip_when_cache_disabled(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )
    cfg_hash = builder._sha1(cfg_path)

    cp_path = tmp_path / "ckpt.json"
    cp = builder.Checkpointer(cp_path)
    cp.load()
    cp.mark_done("AAA", cfg_hash=cfg_hash, timestamp=0.0)

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )

    seen: dict[str, list[str]] = {}

    def fake_fetch(**kwargs):
        seen["tickers"] = kwargs["tickers"]
        return _fake_funda_df(kwargs["tickers"]), {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", fake_fetch)
    _patch_adv(monkeypatch)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": False,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    builder.build_universe(cfg, cfg_path, run_id="RUN-3")

    assert seen.get("tickers") == ["AAA"]


def test_build_universe_raises_on_canary_violation(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder,
        "fetch_fundamentals_parallel",
        lambda **kwargs: (_fake_funda_df(["AAA"]), {"failed": []}),
    )
    monkeypatch.setattr(builder, "_enforce_canary", builder._enforce_canary)
    _patch_adv(monkeypatch)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": None,
            "canary": {"min_valid_tickers": 2, "max_nan_pct": 0.0},
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    with pytest.raises(RuntimeError):
        builder.build_universe(cfg, cfg_path, run_id="RUN-3")


def test_build_universe_wires_vendor_config_into_fundamentals(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    _patch_adv(monkeypatch)

    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen["vendor"] = kwargs.get("vendor")
        seen["use_token_bucket"] = kwargs.get("use_token_bucket")
        seen["request_timeout"] = kwargs.get("request_timeout")
        seen["request_retries"] = kwargs.get("request_retries")
        seen["request_backoff"] = kwargs.get("request_backoff")
        return _fake_funda_df(["AAA"]), {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)

    cfg = {
        "universe": {
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": None,
            "workers": 1,
            "request_timeout": 17,
            "request_retries": 5,
            "request_backoff": 2.4,
            "canary": {"min_valid_tickers": 1},
        },
        "data": {"fundamentals_cache_enabled": False},
        "vendor": {
            "rate_limit_per_sec": 1.25,
            "max_retries": 7,
            "base_backoff": 0.2,
            "backoff_factor": 1.1,
        },
    }

    builder.build_universe(cfg, cfg_path, run_id="RUN-VENDOR")

    vendor = seen.get("vendor")
    assert vendor is not None
    assert seen.get("use_token_bucket") is False
    assert seen.get("request_timeout") == pytest.approx(17.0)
    assert seen.get("request_retries") == 5
    assert seen.get("request_backoff") == pytest.approx(2.4)
    assert vendor.cfg.rate_limit_per_sec == pytest.approx(1.25)
    assert vendor.cfg.max_retries == 7
    assert vendor.cfg.base_backoff == pytest.approx(0.2)
    assert vendor.cfg.backoff_factor == pytest.approx(1.1)


def test_build_universe_defaults_max_inflight_to_workers(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(builder, "load_fundamentals_store", lambda path: pd.DataFrame())
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    _patch_adv(monkeypatch)

    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen["max_inflight"] = kwargs.get("max_inflight")
        return _fake_funda_df(["AAA"]), {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)

    cfg = {
        "universe": {
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": None,
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": None,
            "workers": 3,
            "canary": {"min_valid_tickers": 1},
        },
        "data": {"fundamentals_cache_enabled": False},
    }

    builder.build_universe(cfg, cfg_path, run_id="RUN-INFLIGHT-DEFAULT")
    assert seen.get("max_inflight") == 3
