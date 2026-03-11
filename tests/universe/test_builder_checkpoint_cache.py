from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from universe import builder, checkpoint


def test_build_universe_uses_cache_and_cleans_checkpoint(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    test_hash = "TESTHASH"
    cp_path = tmp_path / "ckpt.json"
    cp = checkpoint.Checkpointer(cp_path)
    now_ts = time.time()
    cp.mark_done("AAA", cfg_hash=test_hash, timestamp=now_ts)
    cp.mark_done("ORPHAN", cfg_hash=test_hash, timestamp=now_ts)

    cached_df = pd.DataFrame(
        {"price": [10.0], "market_cap": [1_000_000_000.0], "volume": [1_000_000.0]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    monkeypatch.setattr(builder, "_sha1", lambda path: test_hash)
    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA", "BBB"])
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "load_fundamentals_store", lambda path: cached_df.copy()
    )

    saved = {}

    def _fake_save(df: pd.DataFrame, path: Path) -> Path:
        saved["df"] = df.copy()
        return path

    monkeypatch.setattr(builder, "save_fundamentals_store", _fake_save)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )

    def _fake_fetch(**kwargs):
        assert kwargs["checkpoint_filter"] == {"AAA"}
        df_new = pd.DataFrame(
            {
                "price": [20.0],
                "market_cap": [2_000_000_000.0],
                "volume": [2_000_000.0],
            },
            index=pd.Index(["BBB"], name="ticker"),
        )
        return df_new, {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        builder,
        "load_or_compute_adv_map",
        lambda tickers, *a, **k: (
            (
                pd.DataFrame(
                    {
                        "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                        "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                        "volume_warmup_avg": pd.Series(
                            {t: 1_000_000.0 for t in tickers}
                        ),
                        "adv_window": float(2),
                        "adv_asof": pd.Timestamp("2024-01-02"),
                    }
                ),
                {"fingerprint": "TEST"},
            )
            if k.get("return_meta", False)
            else pd.DataFrame()
        ),
    )

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": str(tmp_path / "fundamentals.pkl"),
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": False,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {
            "fundamentals_cache_enabled": True,
            "fundamentals_cache_ttl_days": 1.0,
        },
    }

    df_uni, df_funda, monitoring, extra = builder.build_universe(
        cfg, cfg_path, run_id="RUN-CACHE"
    )

    cp_fresh = checkpoint.Checkpointer(cp_path)
    cp_fresh.load()

    assert set(df_funda.index) == {"AAA", "BBB"}
    assert "ORPHAN" not in cp_fresh.entries()
    assert saved["df"].shape[0] == 2  # merged cache + fresh
    assert monitoring["failed"] == []
    assert extra["n_tickers_total"] == 2


def test_build_universe_does_not_reuse_stale_checkpoint_cache(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    test_hash = "TESTHASH"
    cp_path = tmp_path / "ckpt.json"
    cp = checkpoint.Checkpointer(cp_path)
    cp.mark_done("AAA", cfg_hash=test_hash, timestamp=0.0)

    cached_df = pd.DataFrame(
        {"price": [10.0], "market_cap": [1_000_000_000.0], "volume": [1_000_000.0]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    monkeypatch.setattr(builder, "_sha1", lambda path: test_hash)
    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA"])
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "load_fundamentals_store", lambda path: cached_df.copy()
    )
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )

    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen["tickers"] = list(kwargs["tickers"])
        seen["checkpoint_filter"] = kwargs.get("checkpoint_filter")
        df_new = pd.DataFrame(
            {"price": [50.0], "market_cap": [5_000_000_000.0], "volume": [5_000_000.0]},
            index=pd.Index(["AAA"], name="ticker"),
        )
        return df_new, {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        builder,
        "load_or_compute_adv_map",
        lambda tickers, *a, **k: (
            (
                pd.DataFrame(
                    {
                        "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                        "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                        "volume_warmup_avg": pd.Series(
                            {t: 1_000_000.0 for t in tickers}
                        ),
                        "adv_window": float(2),
                        "adv_asof": pd.Timestamp("2024-01-02"),
                    }
                ),
                {"fingerprint": "TEST"},
            )
            if k.get("return_meta", False)
            else pd.DataFrame()
        ),
    )

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": str(tmp_path / "fundamentals.pkl"),
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": False,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {
            "fundamentals_cache_enabled": True,
            "fundamentals_cache_ttl_days": 30.0,
        },
    }

    _, df_funda, _, extra = builder.build_universe(
        cfg, cfg_path, run_id="RUN-CACHE-STALE"
    )

    assert seen["tickers"] == ["AAA"]
    assert seen["checkpoint_filter"] is None
    assert float(df_funda.loc["AAA", "price"]) == 50.0
    assert extra["fundamentals_provenance"]["cache_used"] is False


def test_build_universe_keeps_cache_when_checkpoint_is_empty(monkeypatch, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    test_hash = "TESTHASH"
    cp_path = tmp_path / "ckpt.json"
    checkpoint.Checkpointer(cp_path).reset()

    cached_df = pd.DataFrame(
        {
            "price": [10.0, 11.0],
            "market_cap": [1_000_000_000.0, 1_100_000_000.0],
            "volume": [1_000_000.0, 1_200_000.0],
        },
        index=pd.Index(["AAA", "BBB"], name="ticker"),
    )

    monkeypatch.setattr(builder, "_sha1", lambda path: test_hash)
    monkeypatch.setattr(builder, "load_exchange_tickers", lambda **_: ["AAA", "BBB"])
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "load_fundamentals_store", lambda path: cached_df.copy()
    )
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )

    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen["tickers"] = list(kwargs["tickers"])
        seen["checkpoint_filter"] = kwargs.get("checkpoint_filter")
        return pd.DataFrame(), {"failed": []}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        builder,
        "load_or_compute_adv_map",
        lambda tickers, *a, **k: (
            (
                pd.DataFrame(
                    {
                        "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                        "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                        "volume_warmup_avg": pd.Series(
                            {t: 1_000_000.0 for t in tickers}
                        ),
                        "adv_window": float(2),
                        "adv_asof": pd.Timestamp("2024-01-02"),
                    }
                ),
                {"fingerprint": "TEST"},
            )
            if k.get("return_meta", False)
            else pd.DataFrame()
        ),
    )

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": str(tmp_path / "fundamentals.pkl"),
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": False,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {
            "fundamentals_cache_enabled": True,
            "fundamentals_cache_ttl_days": 30.0,
        },
    }

    _, df_funda, _, extra = builder.build_universe(
        cfg, cfg_path, run_id="RUN-CACHE-EMPTY-CKPT"
    )

    assert seen["tickers"] == ["AAA", "BBB"]
    assert seen["checkpoint_filter"] == {"AAA", "BBB"}
    assert set(df_funda.index) == {"AAA", "BBB"}
    assert extra["fundamentals_provenance"]["cache_used"] is True


def test_build_universe_persists_and_reuses_failed_fundamentals_symbols(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n"
    )

    test_hash = "TESTHASH"
    cp_path = tmp_path / "ckpt_failed_reuse.json"
    cp = checkpoint.Checkpointer(cp_path)
    cp.load()
    cp.mark_done_many(["AAA"], cfg_hash=test_hash, default_timestamp=time.time())
    cp.mark_failed_many(["BAD1"], cfg_hash=test_hash, default_timestamp=time.time())

    cached_df = pd.DataFrame(
        {"price": [10.0], "market_cap": [1_000_000_000.0], "volume": [1_000_000.0]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    monkeypatch.setattr(builder, "_sha1", lambda path: test_hash)
    monkeypatch.setattr(
        builder, "load_exchange_tickers", lambda **_: ["AAA", "BAD1", "BAD2"]
    )
    monkeypatch.setattr(builder, "set_junk_overrides", lambda **_: None)
    monkeypatch.setattr(
        builder, "load_fundamentals_store", lambda path: cached_df.copy()
    )
    monkeypatch.setattr(builder, "save_fundamentals_store", lambda df, path: path)
    monkeypatch.setattr(
        builder, "_enforce_canary", lambda df, **__: {"n_rows": len(df)}
    )
    monkeypatch.setattr(
        builder,
        "load_price_volume_panels",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        builder,
        "load_or_compute_adv_map",
        lambda tickers, *a, **k: (
            (
                pd.DataFrame(
                    {
                        "dollar_adv_hist": pd.Series({t: 1_000_000.0 for t in tickers}),
                        "price_warmup_med": pd.Series({t: 10.0 for t in tickers}),
                        "volume_warmup_avg": pd.Series(
                            {t: 1_000_000.0 for t in tickers}
                        ),
                        "adv_window": float(2),
                        "adv_asof": pd.Timestamp("2024-01-02"),
                    }
                ),
                {"fingerprint": "TEST"},
            )
            if k.get("return_meta", False)
            else pd.DataFrame()
        ),
    )

    calls: list[set[str]] = []

    def _fake_fetch(**kwargs):
        filt = set(kwargs.get("checkpoint_filter") or set())
        calls.append(filt)
        return pd.DataFrame(), {"failed": [], "postfill_unresolved_all": ["BAD2"]}

    monkeypatch.setattr(builder, "fetch_fundamentals_parallel", _fake_fetch)

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": "out.csv",
            "manifest": "man.json",
            "fundamentals_out": str(tmp_path / "fundamentals.pkl"),
        },
        "filters": {},
        "runtime": {
            "checkpoint_path": str(cp_path),
            "reuse_exchange_seed": False,
            "canary": {"min_valid_tickers": 1},
            "workers": 1,
        },
        "data": {
            "fundamentals_cache_enabled": True,
            "fundamentals_cache_ttl_days": 30.0,
        },
    }

    builder.build_universe(cfg, cfg_path, run_id="RUN-FAILED-1")
    builder.build_universe(cfg, cfg_path, run_id="RUN-FAILED-2")

    assert calls, "expected fetch to be called"
    assert calls[0] == {"AAA", "BAD1"}
    assert calls[1] == {"AAA", "BAD1", "BAD2"}

    cp_fresh = checkpoint.Checkpointer(cp_path)
    cp_fresh.load()
    failed = cp_fresh.failed_symbols(cfg_hash=test_hash, max_age=3600.0)
    assert "BAD1" in failed
    assert "BAD2" in failed
