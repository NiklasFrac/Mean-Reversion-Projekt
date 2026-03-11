from __future__ import annotations

import pickle
import types

import pandas as pd
import pytest

from universe import coercion
from universe import config, utils
from universe import fundamentals as funda
from universe import numeric_utils
from universe import storage
from universe.artifact_defaults import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_TICKERS_CSV,
    DEFAULT_RAW_PRICES_PATH,
)
from universe.symbol_filter_defaults import (
    DEFAULT_DROP_CONTAINS,
    DEFAULT_DROP_PREFIXES,
    DEFAULT_DROP_SUFFIXES,
)


def test_load_cfg_or_default_returns_minimal_structure():
    cfg = config.load_cfg_or_default(None)

    assert set(cfg) >= {"universe", "filters", "runtime"}
    assert "exchange" not in cfg["universe"]
    assert cfg["universe"]["manifest"]
    assert cfg["filters"] == {}


def test_validate_cfg_sets_defaults_and_rejects_missing_manifest():
    with pytest.raises(ValueError):
        config.validate_cfg({})

    raw = {
        "universe": {"output_tickers_csv": "tickers.csv", "manifest": "manifest.json"}
    }
    validated = config.validate_cfg(raw)

    assert "exchange" not in validated.universe
    assert validated.filters["drop_prefixes"]
    assert validated.runtime["checkpoint_path"] == DEFAULT_CHECKPOINT_PATH
    assert validated.data["raw_prices_cache"] == DEFAULT_RAW_PRICES_PATH
    assert validated.data["adv_min_valid_ratio"] == pytest.approx(0.7)


def test_config_and_storage_share_artifact_path_defaults():
    validated = config.validate_cfg(
        {"universe": {"output_tickers_csv": "tickers.csv", "manifest": "manifest.json"}}
    )
    paths = storage.resolve_artifact_paths(
        universe_cfg=validated.universe,
        data_cfg=validated.data,
        runtime_cfg=validated.runtime,
    )

    assert validated.universe["output_tickers_csv"] == "tickers.csv"
    assert validated.universe["manifest"] == "manifest.json"
    assert validated.data["raw_prices_cache"] == DEFAULT_RAW_PRICES_PATH
    assert str(paths.raw_prices_cache).replace("\\", "/") == DEFAULT_RAW_PRICES_PATH
    assert str(paths.manifest).replace("\\", "/") == "manifest.json"

    minimal = config.load_cfg_or_default(None)
    assert minimal["universe"]["output_tickers_csv"] == DEFAULT_OUTPUT_TICKERS_CSV
    assert minimal["universe"]["manifest"] == DEFAULT_MANIFEST_PATH


def test_symbol_filter_defaults_are_centralized():
    validated = config.validate_cfg(
        {"universe": {"output_tickers_csv": "tickers.csv", "manifest": "manifest.json"}}
    )
    assert tuple(validated.filters["drop_prefixes"]) == DEFAULT_DROP_PREFIXES
    assert tuple(validated.filters["drop_suffixes"]) == DEFAULT_DROP_SUFFIXES
    assert tuple(validated.filters["drop_contains"]) == DEFAULT_DROP_CONTAINS


def test_validate_cfg_removes_deprecated_knobs():
    raw = {
        "universe": {
            "output_tickers_csv": "tickers.csv",
            "manifest": "manifest.json",
            "adv_path": "legacy_adv.csv",
            "metadata_cache": "legacy_meta.json",
            "default_name": "legacy",
        },
        "runtime": {
            "parallel_exchange_load": True,
            "rate_limit_per_sec": 9.0,
            "rng_seed": 123,
            "skip_if_fresh_hours": 1.0,
            "adv_executor": "thread",
            "adv_workers": 2,
            "profile": {"enabled": True},
            "persist_snapshots": True,
            "snapshots_dir": "runs/snapshots",
        },
        "data": {
            "adv_path": "runs/data/adv.csv",
            "fundamentals_cache_dir": "runs/data/tmp/fundamentals",
            "fundamentals_use_major_holders": True,
        },
        "monitoring": {
            "enabled": True,
            "heartbeat_sec": 30,
            "prometheus": {
                "enabled": True,
                "port": 9108,
                "namespace": "x",
                "keep_alive_sec": 5,
            },
        },
        "vendor": {"cache_root": "runs/data/tmp/yahoo"},
        "mlflow": {"enabled": True},
    }

    cfg = config.validate_cfg(raw)

    assert "adv_path" not in cfg.universe
    assert "metadata_cache" not in cfg.universe
    assert "default_name" not in cfg.universe

    for key in (
        "parallel_exchange_load",
        "rate_limit_per_sec",
        "rng_seed",
        "skip_if_fresh_hours",
        "adv_executor",
        "adv_workers",
        "profile",
        "persist_snapshots",
        "snapshots_dir",
    ):
        assert key not in cfg.runtime

    assert "fundamentals_cache_dir" not in cfg.data
    assert "fundamentals_use_major_holders" not in cfg.data
    assert "cache_root" not in cfg.vendor
    assert "mlflow" not in cfg.raw
    assert "enabled" not in cfg.monitoring
    assert "heartbeat_sec" not in cfg.monitoring
    assert "namespace" not in cfg.monitoring.get("prometheus", {})
    assert "keep_alive_sec" not in cfg.monitoring.get("prometheus", {})


def test_enforce_canary_uses_float_basis_and_raises_on_excess_nan():
    df = pd.DataFrame(
        {"float_pct": [0.1, None, None], "price": [10, 12, 9], "volume": [1, 1, 1]},
        index=["A", "B", "C"],
    )

    with pytest.raises(RuntimeError):
        utils._enforce_canary(df, min_valid_tickers=2, max_nan_pct=0.3)


def test_enforce_canary_numeric_basis_reports_stats_when_float_missing():
    df = pd.DataFrame(
        {
            "price": [10, None, 8],
            "volume": [1_000, 2_000, None],
            "market_cap": [1e9, None, 5e8],
        },
        index=["A", "B", "C"],
    )

    stats = utils._enforce_canary(df, min_valid_tickers=1, max_nan_pct=0.6)

    assert stats["nan_basis_used"] == "core_fields"
    assert 0.0 <= stats["nan_share_checked"] <= 0.6


def test_enforce_canary_checks_core_even_when_float_pct_is_complete():
    df = pd.DataFrame(
        {
            "float_pct": [0.5, 0.4, 0.6],
            "price": [10.0, None, 12.0],
            "market_cap": [1e9, 9e8, None],
            "volume": [1_000_000, None, 900_000],
        },
        index=["AAA", "BBB", "CCC"],
    )

    with pytest.raises(RuntimeError, match="core_fields"):
        utils._enforce_canary(df, min_valid_tickers=1, max_nan_pct=0.2)


def test_validate_cfg_raises_for_invalid_pydantic_field():
    if not getattr(config, "_HAS_PYDANTIC", False):
        pytest.skip("pydantic not installed")
    raw = {
        "universe": {"output_tickers_csv": "tickers.csv", "manifest": "manifest.json"},
        "filters": {"min_float_pct": 1.5},
    }
    with pytest.raises(ValueError, match="validation failed"):
        config.validate_cfg(raw)


def test_is_junk_uses_overrides_and_base_rules(monkeypatch):
    funda.set_junk_overrides(suffixes={"-P"}, contains={"X"})
    try:
        assert funda.is_junk("ABC-P")
        assert funda.is_junk("XTICK")
        assert not funda.is_junk("CLEAN")
    finally:
        funda.set_junk_overrides(suffixes=None, contains=None)


def test_atomic_write_pickle_retries_replace(monkeypatch, tmp_path):
    out = tmp_path / "obj.pkl"
    calls = {"n": 0}
    real_replace = utils.os.replace

    def _fake_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(utils.os, "replace", _fake_replace)
    payload = {"a": 1, "b": 2}
    utils._atomic_write_pickle(payload, out, attempts=5)

    with out.open("rb") as fh:
        loaded = pickle.load(fh)
    assert loaded == payload
    assert calls["n"] == 3


def test_coercion_shared_numeric_parsers():
    assert coercion.coerce_int("7", 3, min_value=1) == 7
    assert coercion.coerce_int("bad", 3, min_value=1) == 3
    assert coercion.coerce_int(-2, 3, min_value=1) == 1

    assert coercion.coerce_float("1.5", 0.5, min_value=0.0) == pytest.approx(1.5)
    assert coercion.coerce_float("bad", 0.5, min_value=0.0) == pytest.approx(0.5)
    assert coercion.coerce_float(-2.0, 0.5, min_value=0.0) == pytest.approx(0.0)
    assert coercion.coerce_float(0.0, 0.5, strictly_positive=True) == pytest.approx(0.5)
    assert coercion.clamp01(1.2) == pytest.approx(1.0)
    assert coercion.clamp01(-0.2) == pytest.approx(0.0)
    assert coercion.clamp01("bad", default=0.3) == pytest.approx(0.3)


def test_cfg_wrappers_reuse_shared_coercion_helpers():
    test_logger = types.SimpleNamespace(warning=lambda *args, **kwargs: None)
    cfg = {"a": "bad", "b": -1, "c": "false"}

    assert (
        coercion.cfg_int(cfg, "a", 9, min_value=1, logger=test_logger, section_name="x")
        == 9
    )
    assert coercion.cfg_float(
        cfg, "b", 1.0, min_value=0.0, logger=test_logger, section_name="x"
    ) == pytest.approx(0.0)
    assert coercion.cfg_bool(cfg, "c", True) is False
    assert coercion.is_truthy("YES")
    assert not coercion.is_truthy("false")


def test_replace_inf_with_nan_shared_numeric_helper():
    s = pd.Series([1.0, float("inf"), float("-inf")])
    out = numeric_utils.replace_inf_with_nan(s)
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
