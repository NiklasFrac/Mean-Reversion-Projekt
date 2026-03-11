from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from universe import adv as adv_mod


def _cfg_for_adv(path: str, *, window: int = 5) -> dict[str, dict[str, object]]:
    return {
        "data": {
            "adv_path": path,
            "adv_window": window,
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-12-31",
            "download_interval": "1d",
            "adv_cache_ttl_days": 365.0,
            "adv_cache_min_coverage_ratio": 0.8,
        }
    }


def test_adv_fingerprint_normalizes_warmup_end_variants():
    tickers = ["AAA", "BBB"]
    data_cfg = {
        "download_start_date": "2024-01-01",
        "download_end_date": "2024-12-31",
        "download_interval": "1d",
    }
    window = 30
    ts = pd.Timestamp("2024-03-15 17:30:00", tz="US/Eastern")
    ts_idx = pd.DatetimeIndex([pd.Timestamp("2024-03-15")])

    fp_ts = adv_mod.adv_fingerprint(tickers, data_cfg, window, warmup_end=ts)
    fp_idx = adv_mod.adv_fingerprint(tickers, data_cfg, window, warmup_end=ts_idx)

    assert fp_ts == fp_idx


def test_compute_adv_map_drops_symbol_with_insufficient_window_coverage():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAA_close": [10, 11, 12, 13, 14, 15]}, index=idx)
    volumes = pd.DataFrame({"AAA": [100, 101, 102, 103, 104, 105]}, index=idx)

    out = adv_mod.compute_adv_map(
        ["AAA"],
        prices_df=prices,
        volumes_df=volumes,
        warmup_end=None,
        window=10,  # min_valid=7 -> symbol should be dropped
    )

    assert out.empty
    assert list(out.columns) == [
        "dollar_adv_hist",
        "price_warmup_med",
        "volume_warmup_avg",
        "adv_window",
        "adv_asof",
    ]


def test_compute_adv_map_min_valid_ratio_is_configurable():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAA_close": [10, 11, 12, 13, 14, 15]}, index=idx)
    volumes = pd.DataFrame({"AAA": [100, 101, 102, 103, 104, 105]}, index=idx)

    out_default = adv_mod.compute_adv_map(
        ["AAA"],
        prices_df=prices,
        volumes_df=volumes,
        warmup_end=None,
        window=10,  # default ratio=0.7 => min_valid=7 -> dropped
    )
    out_relaxed = adv_mod.compute_adv_map(
        ["AAA"],
        prices_df=prices,
        volumes_df=volumes,
        warmup_end=None,
        window=10,
        min_valid_ratio=0.6,  # min_valid=6 -> retained
    )

    assert out_default.empty
    assert "AAA" in out_relaxed.index
    assert float(out_relaxed.loc["AAA", "price_warmup_med"]) == pytest.approx(12.5)


def test_compute_adv_map_respects_warmup_cutoff_and_window_tail():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"AAA_close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, index=idx)
    volumes = pd.DataFrame({"AAA": [10] * 10}, index=idx)

    out = adv_mod.compute_adv_map(
        ["AAA"],
        prices_df=prices,
        volumes_df=volumes,
        warmup_end=pd.Timestamp("2024-01-06 23:00:00", tz="UTC"),
        window=3,
    )

    assert "AAA" in out.index
    # cutoff keeps 1..6; window tail is 4..6
    assert float(out.loc["AAA", "dollar_adv_hist"]) == pytest.approx(50.0)
    assert float(out.loc["AAA", "price_warmup_med"]) == pytest.approx(5.0)
    assert pd.Timestamp(out.loc["AAA", "adv_asof"]).normalize() == pd.Timestamp(
        "2024-01-06"
    )


def test_load_or_compute_adv_map_uses_valid_cache_without_recompute(
    monkeypatch, tmp_path
):
    tickers = ["AAA", "BBB"]
    adv_path = tmp_path / "adv_map.csv"
    cfg = _cfg_for_adv(str(adv_path), window=5)
    fingerprint = adv_mod.adv_fingerprint(tickers, cfg["data"], 5, warmup_end=None)

    cached = pd.DataFrame(
        {
            "dollar_adv_hist": [100.0, 200.0],
            "price_warmup_med": [10.0, 20.0],
            "volume_warmup_avg": [1000.0, 2000.0],
            "adv_window": [5.0, 5.0],
            "adv_asof": ["2024-01-05", "2024-01-05"],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    cached.to_csv(adv_path)
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": fingerprint, "tickers": len(tickers)}),
        encoding="utf-8",
    )

    def _boom(*args, **kwargs):
        raise AssertionError("compute_adv_map should not be called on valid cache hit")

    monkeypatch.setattr(adv_mod, "compute_adv_map", _boom)

    idx = pd.date_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {"AAA_close": [1, 2, 3, 4, 5, 6], "BBB_close": [2, 3, 4, 5, 6, 7]}, index=idx
    )
    volumes = pd.DataFrame({"AAA": [10] * 6, "BBB": [20] * 6}, index=idx)

    out = adv_mod.load_or_compute_adv_map(
        tickers,
        cfg,
        prices,
        volumes,
        warmup_end=None,
    )

    assert out.shape[0] == 2
    assert float(out.loc["AAA", "dollar_adv_hist"]) == 100.0
    assert pd.Timestamp(out.loc["BBB", "adv_asof"]).normalize() == pd.Timestamp(
        "2024-01-05"
    )


def test_load_or_compute_adv_map_recomputes_on_fingerprint_mismatch(
    monkeypatch, tmp_path
):
    tickers = ["AAA"]
    adv_path = tmp_path / "adv_map.csv"
    cfg = _cfg_for_adv(str(adv_path), window=5)

    stale = pd.DataFrame(
        {
            "dollar_adv_hist": [111.0],
            "price_warmup_med": [11.0],
            "volume_warmup_avg": [1100.0],
            "adv_window": [5.0],
            "adv_asof": ["2024-01-03"],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    stale.to_csv(adv_path)
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": "WRONG", "tickers": len(tickers)}),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_compute(*args, **kwargs):
        calls["n"] += 1
        return pd.DataFrame(
            {
                "dollar_adv_hist": [222.0],
                "price_warmup_med": [22.0],
                "volume_warmup_avg": [2200.0],
                "adv_window": [5.0],
                "adv_asof": [pd.Timestamp("2024-01-04")],
            },
            index=pd.Index(["AAA"], name="ticker"),
        )

    monkeypatch.setattr(adv_mod, "compute_adv_map", _fake_compute)

    idx = pd.date_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"AAA_close": [1, 2, 3, 4, 5, 6]}, index=idx)
    volumes = pd.DataFrame({"AAA": [10] * 6}, index=idx)

    out = adv_mod.load_or_compute_adv_map(
        tickers,
        cfg,
        prices,
        volumes,
        warmup_end=None,
    )

    assert calls["n"] == 1
    assert float(out.loc["AAA", "dollar_adv_hist"]) == 222.0


def test_load_or_compute_adv_map_does_not_overwrite_cache_with_empty_recompute(
    monkeypatch, tmp_path
):
    tickers = ["AAA"]
    adv_path = tmp_path / "adv_map.csv"
    cfg = _cfg_for_adv(str(adv_path), window=5)

    seeded = pd.DataFrame(
        {
            "dollar_adv_hist": [111.0],
            "price_warmup_med": [11.0],
            "volume_warmup_avg": [1100.0],
            "adv_window": [5.0],
            "adv_asof": ["2024-01-03"],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    seeded.to_csv(adv_path)
    # Mismatch forces recompute path.
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": "WRONG", "tickers": len(tickers)}),
        encoding="utf-8",
    )
    before = adv_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        adv_mod,
        "compute_adv_map",
        lambda *args, **kwargs: pd.DataFrame(
            columns=[
                "dollar_adv_hist",
                "price_warmup_med",
                "volume_warmup_avg",
                "adv_window",
                "adv_asof",
            ]
        ),
    )

    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"AAA_close": [1, 2, 3]}, index=idx)
    volumes = pd.DataFrame({"AAA": [10, 20, 30]}, index=idx)

    with pytest.raises(RuntimeError, match="no safe fallback ADV cache"):
        adv_mod.load_or_compute_adv_map(
            tickers,
            cfg,
            prices,
            volumes,
            warmup_end=None,
        )

    after = adv_path.read_text(encoding="utf-8")
    assert after == before


def test_load_or_compute_adv_map_reuses_stale_safe_cache_when_recompute_is_empty(
    monkeypatch, tmp_path
):
    tickers = ["AAA"]
    adv_path = tmp_path / "adv_map.csv"
    cfg = _cfg_for_adv(str(adv_path), window=5)
    cfg["data"]["adv_cache_ttl_days"] = 1.0
    fingerprint = adv_mod.adv_fingerprint(tickers, cfg["data"], 5, warmup_end=None)

    seeded = pd.DataFrame(
        {
            "dollar_adv_hist": [333.0],
            "price_warmup_med": [33.0],
            "volume_warmup_avg": [3300.0],
            "adv_window": [5.0],
            "adv_asof": ["2024-01-03"],
        },
        index=pd.Index(tickers, name="ticker"),
    )
    seeded.to_csv(adv_path)
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": fingerprint, "tickers": len(tickers)}),
        encoding="utf-8",
    )
    stale_ts = time.time() - (3 * 86400)
    os.utime(adv_path, (stale_ts, stale_ts))
    before = adv_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        adv_mod,
        "compute_adv_map",
        lambda *args, **kwargs: pd.DataFrame(
            columns=[
                "dollar_adv_hist",
                "price_warmup_med",
                "volume_warmup_avg",
                "adv_window",
                "adv_asof",
            ]
        ),
    )

    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"AAA_close": [1, 2, 3]}, index=idx)
    volumes = pd.DataFrame({"AAA": [10, 20, 30]}, index=idx)

    out = adv_mod.load_or_compute_adv_map(
        tickers,
        cfg,
        prices,
        volumes,
        warmup_end=None,
    )

    assert float(out.loc["AAA", "dollar_adv_hist"]) == 333.0
    after = adv_path.read_text(encoding="utf-8")
    assert after == before


def test_load_price_volume_panels_sanitizes_invalid_download_knobs(
    monkeypatch, tmp_path
):
    captured: dict[str, float | int] = {}

    def _fake_fetch(*args, **kwargs):
        (
            _tickers,
            _start,
            _end,
            _interval,
            batch_size,
            pause,
            retries,
            backoff,
            _use_threads,
        ) = args[:9]
        captured["batch_size"] = int(batch_size)
        captured["pause"] = float(pause)
        captured["retries"] = int(retries)
        captured["backoff"] = float(backoff)
        idx = pd.date_range("2024-01-01", periods=2)
        prices = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
        vols = pd.DataFrame({"AAA": [100.0, 110.0]}, index=idx)
        return prices, vols

    def _fake_retry(*args, **kwargs):
        captured["retry_retries"] = int(kwargs.get("retries"))
        captured["retry_backoff"] = float(kwargs.get("backoff"))
        prices_df = args[1]
        volumes_df = args[2]
        return prices_df, volumes_df, []

    monkeypatch.setattr(adv_mod, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(adv_mod, "_retry_missing_history", _fake_retry)

    prices, vols = adv_mod.load_price_volume_panels(
        ["AAA"],
        {
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-05",
            "download_interval": "1d",
            "download_batch": 0,
            "download_pause": -5.0,
            "download_retries": -2,
            "backoff_factor": 0.0,
            "raw_prices_cache": str(tmp_path / "raw_prices.pkl"),
            "volume_path": str(tmp_path / "raw_volume.pkl"),
        },
        auto_adjust=False,
    )

    assert not prices.empty
    assert not vols.empty
    assert captured["batch_size"] == 1
    assert captured["pause"] == 0.0
    assert captured["retries"] == 0
    assert captured["backoff"] == 2.0
    assert captured["retry_retries"] == 0
    assert captured["retry_backoff"] == 2.0


def test_load_price_volume_panels_forwards_request_timeout(monkeypatch, tmp_path):
    seen: dict[str, float | None] = {}

    def _fake_fetch(*args, **kwargs):
        seen["fetch_timeout"] = kwargs.get("request_timeout")
        idx = pd.date_range("2024-01-01", periods=2)
        prices = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
        vols = pd.DataFrame({"AAA": [100.0, 110.0]}, index=idx)
        return prices, vols

    def _fake_retry(*args, **kwargs):
        seen["retry_timeout"] = kwargs.get("request_timeout")
        prices_df = args[1]
        volumes_df = args[2]
        return prices_df, volumes_df, []

    monkeypatch.setattr(adv_mod, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(adv_mod, "_retry_missing_history", _fake_retry)

    prices, vols = adv_mod.load_price_volume_panels(
        ["AAA"],
        {
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-05",
            "download_interval": "1d",
            "raw_prices_cache": str(tmp_path / "raw_prices.pkl"),
            "volume_path": str(tmp_path / "raw_volume.pkl"),
        },
        auto_adjust=False,
        request_timeout=13.5,
    )

    assert not prices.empty
    assert not vols.empty
    assert seen["fetch_timeout"] == 13.5
    assert seen["retry_timeout"] == 13.5


def test_load_price_volume_panels_uses_atomic_pickle_writer(monkeypatch, tmp_path):
    written: list[str] = []

    def _fake_fetch(*args, **kwargs):
        _ = args, kwargs
        idx = pd.date_range("2024-01-01", periods=2)
        prices = pd.DataFrame({"AAA_close": [1.0, 1.1]}, index=idx)
        vols = pd.DataFrame({"AAA": [100.0, 110.0]}, index=idx)
        return prices, vols

    def _fake_retry(*args, **kwargs):
        _ = kwargs
        return args[1], args[2], []

    def _fake_atomic(obj, path):
        _ = obj
        written.append(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr(adv_mod, "fetch_price_volume_data", _fake_fetch)
    monkeypatch.setattr(adv_mod, "_retry_missing_history", _fake_retry)
    monkeypatch.setattr(adv_mod, "_atomic_write_pickle", _fake_atomic)

    prices_path = tmp_path / "prices.pkl"
    vols_path = tmp_path / "vols.pkl"
    prices, vols = adv_mod.load_price_volume_panels(
        ["AAA"],
        {
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-05",
            "download_interval": "1d",
            "raw_prices_cache": str(prices_path),
            "volume_path": str(vols_path),
        },
        auto_adjust=False,
    )

    assert not prices.empty
    assert not vols.empty
    assert str(prices_path) in written
    assert str(vols_path) in written
    assert prices_path.exists()
    assert vols_path.exists()


def test_load_or_compute_adv_map_accepts_string_warmup_end(tmp_path):
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"AAA_close": [10, 11, 12, 13, 14, 15]}, index=idx)
    volumes = pd.DataFrame({"AAA": [100, 101, 102, 103, 104, 105]}, index=idx)
    cfg = _cfg_for_adv(str(tmp_path / "adv_map.csv"), window=3)

    out, meta = adv_mod.load_or_compute_adv_map(
        ["AAA"],
        cfg,
        prices,
        volumes,
        warmup_end="2024-01-04",
        return_meta=True,
    )

    assert "AAA" in out.index
    assert meta["warmup_end"] == "2024-01-04T00:00:00"
