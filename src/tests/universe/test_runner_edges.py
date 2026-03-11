from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from universe import runner_universe as ru


def _basic_build(tickers: list[str]):
    def _fake_build(cfg, cfg_path, run_id, stop_event=None):
        idx = pd.Index(tickers, name="ticker")
        df_uni = pd.DataFrame(
            {
                "price": [10.0 + i for i in range(len(idx))],
                "market_cap": [1_000_000_000.0 for _ in idx],
                "volume": [1_000_000.0 for _ in idx],
                "float_pct": [0.5 for _ in idx],
                "dividend": [True for _ in idx],
                "is_etf": [False for _ in idx],
                "shares_out": [100_000_000 for _ in idx],
            },
            index=idx,
        )
        monitoring = {"failed": []}
        extra = {
            "n_tickers_total": len(idx),
            "n_filtered": len(idx),
        }
        return df_uni, df_uni.copy(), monitoring, extra

    return _fake_build


def _cfg_file(tmp_path: Path, *, allow_incomplete_history: bool) -> Path:
    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(tmp_path / "tickers.csv"),
            "output_tickers_ext_csv": str(tmp_path / "tickers_ext.csv"),
            "manifest": str(tmp_path / "manifest.json"),
            "adv_cache": str(tmp_path / "adv_cache.pkl"),
        },
        "runtime": {
            "use_hashed_artifacts": False,
            "progress_bar": False,
        },
        "data": {
            "allow_incomplete_history": allow_incomplete_history,
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "raw_prices_unadj_cache": str(tmp_path / "prices_unadj.pkl"),
            "raw_prices_unadj_warmup_cache": str(tmp_path / "prices_unadj_warmup.pkl"),
            "volume_path": str(tmp_path / "volumes.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "volumes_unadj.pkl"),
            "adv_path": str(tmp_path / "adv_map.csv"),
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 10,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_runner_errors_when_history_missing_not_allowed(monkeypatch, tmp_path):
    cfg_path = _cfg_file(tmp_path, allow_incomplete_history=False)
    monkeypatch.setattr(ru, "build_universe", _basic_build(["AAA"]))
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )
    monkeypatch.setattr(
        ru, "fetch_price_volume_data", lambda *a, **k: (pd.DataFrame(), pd.DataFrame())
    )
    monkeypatch.setattr(
        ru,
        "_retry_missing_history",
        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), ["AAA"]),
    )

    with pytest.raises(RuntimeError):
        ru.main(cfg_path)


def test_runner_drops_missing_history_when_allowed(monkeypatch, tmp_path):
    cfg_path = _cfg_file(tmp_path, allow_incomplete_history=True)
    monkeypatch.setattr(ru, "build_universe", _basic_build(["AAA", "BBB"]))
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )
    price_panel = pd.DataFrame(
        {"AAA_close": [1.0, 1.1]}, index=pd.date_range("2024-01-01", periods=2)
    )
    vol_panel = pd.DataFrame({"AAA": [100.0, 110.0]}, index=price_panel.index)
    monkeypatch.setattr(
        ru,
        "fetch_price_volume_data",
        lambda *a, **k: (price_panel, vol_panel),
    )
    monkeypatch.setattr(
        ru,
        "_retry_missing_history",
        lambda *a, **k: (price_panel, vol_panel, ["BBB"]),
    )

    ru.main(cfg_path)

    out_csv = Path(json.loads(cfg_path.read_text())["universe"]["output_tickers_csv"])
    rows = out_csv.read_text().strip().splitlines()
    assert rows == ["ticker", "AAA"]
    ext_csv = Path(
        json.loads(cfg_path.read_text())["universe"]["output_tickers_ext_csv"]
    )
    ext_rows = ext_csv.read_text().strip().splitlines()
    assert ext_rows[0].startswith("ticker")
    assert any("AAA" in line for line in ext_rows[1:])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["extra"]["n_filtered"] == 1


def test_runner_does_not_write_final_outputs_when_download_step_fails(
    monkeypatch, tmp_path
):
    cfg_path = _cfg_file(tmp_path, allow_incomplete_history=True)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(ru, "build_universe", _basic_build(["AAA"]))

    def _raise_download(*args, **kwargs):
        raise RuntimeError("download boom")

    monkeypatch.setattr(ru, "download_and_persist_history", _raise_download)

    with pytest.raises(RuntimeError, match="download boom"):
        ru.main(cfg_path)

    assert not Path(cfg["universe"]["output_tickers_csv"]).exists()
    assert not Path(cfg["universe"]["output_tickers_ext_csv"]).exists()
    assert not Path(cfg["universe"]["manifest"]).exists()


def test_data_quality_ohlc_ignores_missing_rows():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices_unadjusted = pd.DataFrame(
        {
            "AAA_open": [pd.NA, 10.0],
            "AAA_high": [pd.NA, 11.0],
            "AAA_low": [pd.NA, 9.0],
            "AAA_close": [pd.NA, 10.5],
        },
        index=idx,
    )
    volumes_reported = pd.DataFrame({"AAA": [1000, 2000]}, index=idx)

    report = ru._build_data_quality_report(
        prices_unadjusted=prices_unadjusted,
        volumes_reported=volumes_reported,
        max_examples=5,
    )
    assert report["ohlc"]["violations"] == 0


def test_data_quality_ohlc_flags_real_violation():
    idx = pd.to_datetime(["2024-01-02"])
    prices_unadjusted = pd.DataFrame(
        {
            "AAA_open": [10.0],
            "AAA_high": [9.0],
            "AAA_low": [8.0],
            "AAA_close": [8.5],
        },
        index=idx,
    )
    volumes_reported = pd.DataFrame({"AAA": [1000]}, index=idx)

    report = ru._build_data_quality_report(
        prices_unadjusted=prices_unadjusted,
        volumes_reported=volumes_reported,
        max_examples=5,
    )
    assert report["ohlc"]["violations"] == 1


def test_data_quality_reports_volume_when_prices_panel_is_empty():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    volumes_reported = pd.DataFrame({"AAA": [-10.0, 0.0, 15.0]}, index=idx)

    report = ru._build_data_quality_report(
        prices_unadjusted=pd.DataFrame(),
        volumes_reported=volumes_reported,
        max_examples=5,
    )

    assert report["ohlc"]["violations"] == 0
    assert report["volume"]["negative_values"] == 1
    assert report["volume"]["zero_values"] == 1
    assert report["volume"]["examples"]
    assert report["volume"]["examples"][0]["ticker"] == "AAA"


def test_runner_raises_on_non_monotonic_panel(monkeypatch, tmp_path):
    cfg_path = _cfg_file(tmp_path, allow_incomplete_history=True)
    monkeypatch.setattr(ru, "build_universe", _basic_build(["AAA"]))
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )
    # Return duplicate/non-monotonic index and bypass normalization to trigger guard.
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-01", "2024-01-01"])
    prices = pd.DataFrame({"AAA_close": [1.0, 1.1, 1.2]}, index=idx)
    vols = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)

    monkeypatch.setattr(
        ru,
        "_normalize_panel_for_universe",
        lambda df, interval: df,
    )
    monkeypatch.setattr(
        ru,
        "fetch_price_volume_data",
        lambda *a, **k: (prices, vols),
    )
    monkeypatch.setattr(
        ru,
        "_retry_missing_history",
        lambda *a, **k: (prices, vols, []),
    )

    with pytest.raises(RuntimeError):
        ru.main(cfg_path)


def test_runner_passes_corporate_action_flags(monkeypatch, tmp_path):
    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(tmp_path / "tickers.csv"),
            "output_tickers_ext_csv": str(tmp_path / "tickers_ext.csv"),
            "manifest": str(tmp_path / "manifest.json"),
            "adv_cache": str(tmp_path / "adv_cache.pkl"),
        },
        "runtime": {
            "use_hashed_artifacts": False,
            "progress_bar": False,
        },
        "data": {
            "allow_incomplete_history": True,
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "raw_prices_unadj_cache": str(tmp_path / "prices_unadj.pkl"),
            "raw_prices_unadj_warmup_cache": str(tmp_path / "prices_unadj_warmup.pkl"),
            "volume_path": str(tmp_path / "volumes.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "volumes_unadj.pkl"),
            "adv_path": str(tmp_path / "adv_map.csv"),
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 10,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0],
            "market_cap": [1_000_000_000.0],
            "volume": [1_000_000.0],
            "float_pct": [0.5],
            "dividend": [True],
            "is_etf": [False],
            "shares_out": [100_000_000],
        },
        index=idx,
    )

    monkeypatch.setattr(
        ru,
        "build_universe",
        lambda *a, **k: (
            df_uni,
            df_uni.copy(),
            {"failed": []},
            {
                "n_tickers_total": 1,
                "n_filtered": 1,
                "tickers_all": df_uni.index.tolist(),
                "artifacts": {},
            },
        ),
    )
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_copy.pkl",
            tmp_path / "v_copy.pkl",
        ),
    )

    prices = pd.DataFrame(
        {"AAA_close": [1.0, 1.1]}, index=pd.date_range("2024-01-01", periods=2)
    )
    vols = pd.DataFrame({"AAA": [100.0, 110.0]}, index=prices.index)
    fetch_calls: list[dict[str, object]] = []

    def _fake_fetch(*args, **kwargs):
        fetch_calls.append(dict(kwargs))
        return prices, vols

    retry_calls: list[dict[str, object]] = []

    def _fake_retry(*args, **kwargs):
        retry_calls.append(dict(kwargs))
        return prices, vols, []

    monkeypatch.setattr(ru, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(ru, "_retry_missing_history", _fake_retry)

    ru.main(cfg_path)

    # First fetch is adjusted closes for returns; second fetch (for ADV) is unadjusted.
    auto_adjust_flags = [call.get("auto_adjust") for call in fetch_calls]
    assert any(flag is True for flag in auto_adjust_flags)
    assert any(flag is False for flag in auto_adjust_flags)
    retry_flags = [call.get("auto_adjust") for call in retry_calls]
    assert any(flag is True for flag in retry_flags)
    assert any(flag is False for flag in retry_flags)
    for call in fetch_calls:
        assert "actions" not in call and "split_adjustment" not in call


def test_runner_fails_when_unadjusted_matches_adjusted(monkeypatch, tmp_path):
    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(tmp_path / "tickers.csv"),
            "output_tickers_ext_csv": str(tmp_path / "tickers_ext.csv"),
            "manifest": str(tmp_path / "manifest.json"),
        },
        "runtime": {"use_hashed_artifacts": False, "progress_bar": False},
        "data": {
            "allow_incomplete_history": True,
            "strict_unadjusted_validation": True,
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "raw_prices_unadj_cache": str(tmp_path / "prices_unadj.pkl"),
            "volume_path": str(tmp_path / "volumes.pkl"),
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 5,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0],
            "market_cap": [1_000_000_000.0],
            "volume": [1_000_000.0],
            "float_pct": [0.5],
            "dividend": [True],
            "is_etf": [False],
            "shares_out": [100_000_000],
        },
        index=idx,
    )
    extra_base = {
        "n_tickers_total": 1,
        "n_fundamentals_ok": 1,
        "n_failed": 0,
        "n_failed_hard": 0,
        "n_incomplete_core": 0,
        "n_filtered": 1,
        "cfg_path": str(cfg_path),
        "reason_codes": {},
        "artifacts": {},
        "tickers_all": ["AAA"],
        "run_id": "RUN-TEST",
        "canary": {},
    }

    monkeypatch.setattr(
        ru,
        "build_universe",
        lambda *a, **k: (df_uni, df_uni.copy(), {"failed": []}, dict(extra_base)),
    )
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )

    def _fake_fetch(tickers, *args, **kwargs):
        idx = pd.date_range("2024-01-01", periods=6)
        prices = pd.DataFrame({f"{tickers[0]}_close": [1.0] * len(idx)}, index=idx)
        vols = pd.DataFrame({tickers[0]: [100.0] * len(idx)}, index=idx)
        return prices, vols

    monkeypatch.setattr(ru, "fetch_price_volume_data", _fake_fetch)

    def _fake_retry(*args, **kwargs):
        prices, vols = _fake_fetch(["AAA"])
        return prices, vols, []

    monkeypatch.setattr(ru, "_retry_missing_history", _fake_retry)

    with pytest.raises(RuntimeError, match="Unadjusted price validation:"):
        ru.main(cfg_path)


def test_runner_preserves_warmup_unadjusted_path(monkeypatch, tmp_path):
    warmup_path = tmp_path / "raw_unadj.pkl"
    warmup_path.write_bytes(b"warmup-panel")

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(tmp_path / "tickers.csv"),
            "output_tickers_ext_csv": str(tmp_path / "tickers_ext.csv"),
            "manifest": str(tmp_path / "manifest.json"),
            "adv_cache": str(tmp_path / "adv_cache.pkl"),
        },
        "runtime": {"use_hashed_artifacts": False, "progress_bar": False},
        "data": {
            "allow_incomplete_history": True,
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "raw_prices_unadj_cache": str(warmup_path),
            "raw_prices_unadj_warmup_cache": str(warmup_path),
            "volume_path": str(tmp_path / "volumes.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "volumes_unadj.pkl"),
            "adv_path": str(tmp_path / "adv_map.csv"),
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 5,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0],
            "market_cap": [1_000_000_000.0],
            "volume": [1_000_000.0],
            "float_pct": [0.5],
            "dividend": [True],
            "is_etf": [False],
            "shares_out": [100_000_000],
        },
        index=idx,
    )
    extra_base = {
        "n_tickers_total": 1,
        "n_fundamentals_ok": 1,
        "n_failed": 0,
        "n_failed_hard": 0,
        "n_incomplete_core": 0,
        "n_filtered": 1,
        "cfg_path": str(cfg_path),
        "reason_codes": {},
        "artifacts": {},
        "tickers_all": ["AAA"],
        "run_id": "RUN-TEST",
        "canary": {},
    }
    monkeypatch.setattr(
        ru,
        "build_universe",
        lambda *a, **k: (df_uni, df_uni.copy(), {"failed": []}, dict(extra_base)),
    )
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )

    def _fake_fetch(tickers, *args, **kwargs):
        idx = pd.date_range("2024-01-01", periods=6)
        factor = 2.0 if not kwargs.get("auto_adjust", True) else 1.0
        prices = pd.DataFrame({f"{tickers[0]}_close": [factor] * len(idx)}, index=idx)
        vols = pd.DataFrame({tickers[0]: [100.0] * len(idx)}, index=idx)
        return prices, vols

    monkeypatch.setattr(ru, "fetch_price_volume_data", _fake_fetch)

    def _fake_retry_unadj(*args, **kwargs):
        prices, vols = _fake_fetch(["AAA"], auto_adjust=kwargs.get("auto_adjust", True))
        return prices, vols, []

    monkeypatch.setattr(ru, "_retry_missing_history", _fake_retry_unadj)

    ru.main(cfg_path)

    assert warmup_path.read_bytes() == b"warmup-panel"
    filtered_path = warmup_path.with_name(
        warmup_path.stem + "_filtered" + warmup_path.suffix
    )
    assert filtered_path.exists()


def test_runner_respects_adjust_dividends_false(monkeypatch, tmp_path):
    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(tmp_path / "tickers.csv"),
            "output_tickers_ext_csv": str(tmp_path / "tickers_ext.csv"),
            "manifest": str(tmp_path / "manifest.json"),
            "adv_cache": str(tmp_path / "adv_cache.pkl"),
        },
        "runtime": {"use_hashed_artifacts": False, "progress_bar": False},
        "data": {
            "allow_incomplete_history": True,
            "raw_prices_cache": str(tmp_path / "prices.pkl"),
            "volume_path": str(tmp_path / "volumes.pkl"),
            "adv_path": str(tmp_path / "adv_map.csv"),
            "raw_prices_unadj_cache": str(tmp_path / "prices_unadj.pkl"),
            "raw_prices_unadj_warmup_cache": str(tmp_path / "prices_unadj_warmup.pkl"),
            "raw_volume_unadj_cache": str(tmp_path / "volumes_unadj.pkl"),
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 5,
            "adjust_dividends": False,
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0],
            "market_cap": [1_000_000_000.0],
            "volume": [1_000_000.0],
            "float_pct": [0.5],
            "dividend": [True],
            "is_etf": [False],
            "shares_out": [100_000_000],
        },
        index=idx,
    )
    extra_base = {
        "n_tickers_total": 1,
        "n_fundamentals_ok": 1,
        "n_failed": 0,
        "n_failed_hard": 0,
        "n_incomplete_core": 0,
        "n_filtered": 1,
        "cfg_path": str(cfg_path),
        "reason_codes": {},
        "artifacts": {},
        "tickers_all": ["AAA"],
        "run_id": "RUN-TEST",
        "canary": {},
    }

    monkeypatch.setattr(
        ru,
        "build_universe",
        lambda *a, **k: (df_uni, df_uni.copy(), {"failed": []}, dict(extra_base)),
    )
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )

    calls = {"fetch": 0, "retry": 0, "auto": [], "retry_auto": []}

    def _fake_fetch(tickers, *args, **kwargs):
        calls["fetch"] += 1
        calls["auto"].append(kwargs.get("auto_adjust"))
        idx_dt = pd.date_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {f"{tickers[0]}_close": [1.0] * len(idx_dt)}, index=idx_dt
        )
        vols = pd.DataFrame({tickers[0]: [100.0] * len(idx_dt)}, index=idx_dt)
        return prices, vols

    def _fake_retry(*args, **kwargs):
        calls["retry"] += 1
        calls["retry_auto"].append(kwargs.get("auto_adjust"))
        idx_dt = pd.date_range("2024-01-01", periods=5)
        prices = pd.DataFrame({"AAA_close": [1.0] * len(idx_dt)}, index=idx_dt)
        vols = pd.DataFrame({"AAA": [100.0] * len(idx_dt)}, index=idx_dt)
        return prices, vols, []

    monkeypatch.setattr(ru, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(ru, "_retry_missing_history", _fake_retry)

    ru.main(cfg_path)

    assert calls["fetch"] == 1
    assert calls["auto"] == [False]
    assert calls["retry"] == 1
    assert calls["retry_auto"] == [False]

    manifest = json.loads(Path(cfg["universe"]["manifest"]).read_text())
    assert manifest["extra"]["data_policy"]["adjust_dividends"] is False


def test_run_main_with_force_executes_full_pipeline_and_restores_env(
    monkeypatch, tmp_path
):
    cfg_path = _cfg_file(tmp_path, allow_incomplete_history=True)
    state = {"saw_force_inside_build": False, "fetch_calls": 0}
    base_build = _basic_build(["AAA"])

    def _fake_build(cfg, cfg_path_arg, run_id, stop_event=None):
        state["saw_force_inside_build"] = os.environ.get("UNIVERSE_FORCE") == "1"
        return base_build(cfg, cfg_path_arg, run_id, stop_event=stop_event)

    monkeypatch.setattr(ru, "build_universe", _fake_build)
    monkeypatch.setattr(
        ru,
        "artifact_targets",
        lambda **kwargs: (
            tmp_path / "p.pkl",
            tmp_path / "v.pkl",
            tmp_path / "p_unadj.pkl",
            tmp_path / "v_unadj.pkl",
        ),
    )

    def _fake_fetch(tickers, *args, **kwargs):
        state["fetch_calls"] += 1
        idx_dt = pd.date_range("2024-01-01", periods=6)
        factor = 1.0 if kwargs.get("auto_adjust", True) else 2.0
        prices = pd.DataFrame(
            {f"{tickers[0]}_close": [factor] * len(idx_dt)}, index=idx_dt
        )
        vols = pd.DataFrame({tickers[0]: [100.0] * len(idx_dt)}, index=idx_dt)
        return prices, vols

    def _fake_retry(*args, **kwargs):
        prices, vols = _fake_fetch(["AAA"], auto_adjust=kwargs.get("auto_adjust", True))
        return prices, vols, []

    monkeypatch.setattr(ru, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(ru, "_retry_missing_history", _fake_retry)

    monkeypatch.setenv("UNIVERSE_FORCE", "OLD")
    ru._run_main_with_force(cfg_path, force=True)

    assert state["saw_force_inside_build"] is True
    assert state["fetch_calls"] >= 2
    assert os.environ.get("UNIVERSE_FORCE") == "OLD"
