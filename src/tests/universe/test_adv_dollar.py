from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from universe import adv as adv_mod
from universe import runner_universe as ru


def test_runner_writes_dollar_adv(monkeypatch, tmp_path):
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
            "adv_window": 2,
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-01-10",
            "download_interval": "1d",
            "download_batch": 10,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    idx = pd.Index(["AAA", "BBB"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0, 20.0],
            "market_cap": [1_000_000_000.0, 2_000_000_000.0],
            "volume": [1_000_000.0, 2_000_000.0],
            "float_pct": [0.5, 0.6],
            "dividend": [True, False],
            "is_etf": [False, False],
            "shares_out": [100_000_000, 200_000_000],
        },
        index=idx,
    )

    def _fake_build(cfg_arg, cfg_path_arg, run_id, stop_event=None):
        df_funda = df_uni.copy()
        monitoring = {"failed": []}
        extra = {
            "n_tickers_total": len(df_uni),
            "n_filtered": len(df_uni),
            "tickers_all": df_uni.index.tolist(),
            "artifacts": {},
        }
        return df_uni, df_funda, monitoring, extra

    dates = pd.date_range("2024-01-01", periods=2)
    price_panel = pd.DataFrame(
        {"AAA_close": [1.0, 2.0], "BBB_close": [3.0, 5.0]}, index=dates
    )
    volume_panel = pd.DataFrame({"AAA": [10.0, 20.0], "BBB": [30.0, 40.0]}, index=dates)

    monkeypatch.setattr(ru, "build_universe", _fake_build)
    monkeypatch.setattr(
        ru, "fetch_price_volume_data", lambda *a, **k: (price_panel, volume_panel)
    )
    monkeypatch.setattr(
        ru, "_retry_missing_history", lambda *a, **k: (price_panel, volume_panel, [])
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

    # Seed ADV map as produced by build_universe; runner should not overwrite it.
    adv_seed = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "adv_dollar_median_usd": [25.0, 145.0]}
    )
    adv_seed.to_csv(cfg["data"]["adv_path"], index=False)

    before = adv_seed.copy()
    ru.main(cfg_path)

    adv_csv = Path(cfg["data"]["adv_path"])
    assert adv_csv.exists()
    adv_df = pd.read_csv(adv_csv)
    pd.testing.assert_frame_equal(
        adv_df.sort_values("ticker").reset_index(drop=True),
        before.sort_values("ticker").reset_index(drop=True),
        check_like=True,
    )


def test_adv_cache_ttl_is_decoupled_from_fundamentals_ttl(monkeypatch, tmp_path):
    tickers = ["AAA"]
    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"AAA_close": [10.0, 11.0, 12.0]}, index=idx)
    volumes = pd.DataFrame({"AAA": [100.0, 100.0, 100.0]}, index=idx)
    adv_path = tmp_path / "adv_map.csv"

    cfg = {
        "data": {
            "adv_path": str(adv_path),
            "adv_window": 30,
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-12-31",
            "download_interval": "1d",
            "fundamentals_cache_ttl_days": 2000.0,
            "adv_cache_ttl_days": 1.0,
        }
    }

    fingerprint = adv_mod.adv_fingerprint(tickers, cfg["data"], 30, warmup_end=None)
    pd.DataFrame(
        {
            "dollar_adv_hist": [123.0],
            "price_warmup_med": [10.0],
            "volume_warmup_avg": [100.0],
            "adv_window": [30.0],
            "adv_asof": ["2024-01-01"],
        },
        index=pd.Index(["AAA"], name="ticker"),
    ).to_csv(adv_path)
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": fingerprint, "tickers": 1}),
        encoding="utf-8",
    )
    stale_ts = time.time() - (3 * 86400)
    os.utime(adv_path, (stale_ts, stale_ts))

    recomputed = pd.DataFrame(
        {
            "dollar_adv_hist": [999.0],
            "price_warmup_med": [12.0],
            "volume_warmup_avg": [100.0],
            "adv_window": [30.0],
            "adv_asof": [pd.Timestamp("2024-01-03")],
        },
        index=pd.Index(["AAA"], name="ticker"),
    )
    calls = {"compute": 0}

    def _fake_compute(*args, **kwargs):
        calls["compute"] += 1
        return recomputed.copy()

    monkeypatch.setattr(adv_mod, "compute_adv_map", _fake_compute)

    out = adv_mod.load_or_compute_adv_map(
        tickers,
        cfg,
        prices,
        volumes,
        warmup_end=None,
    )

    assert calls["compute"] == 1
    assert float(out.loc["AAA", "dollar_adv_hist"]) == 999.0


def test_adv_cache_recomputes_when_coverage_ratio_too_low(monkeypatch, tmp_path):
    tickers = [f"T{i:03d}" for i in range(100)]
    idx = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({f"{tickers[0]}_close": [10.0, 11.0, 12.0]}, index=idx)
    volumes = pd.DataFrame({tickers[0]: [100.0, 100.0, 100.0]}, index=idx)
    adv_path = tmp_path / "adv_map.csv"
    window = 30

    cfg = {
        "data": {
            "adv_path": str(adv_path),
            "adv_window": window,
            "download_start_date": "2024-01-01",
            "download_end_date": "2024-12-31",
            "download_interval": "1d",
            "adv_cache_ttl_days": 365.0,
            "adv_cache_min_coverage_ratio": 0.80,
        }
    }

    fp = adv_mod.adv_fingerprint(tickers, cfg["data"], window, warmup_end=None)
    low_cov_df = pd.DataFrame(
        {
            "dollar_adv_hist": [100.0] * 10,
            "price_warmup_med": [10.0] * 10,
            "volume_warmup_avg": [100.0] * 10,
            "adv_window": [float(window)] * 10,
            "adv_asof": ["2024-01-01"] * 10,
        },
        index=pd.Index(tickers[:10], name="ticker"),
    )
    low_cov_df.to_csv(adv_path)
    adv_path.with_suffix(adv_path.suffix + ".meta.json").write_text(
        json.dumps({"fingerprint": fp, "tickers": len(tickers)}),
        encoding="utf-8",
    )

    calls = {"compute": 0}

    def _fake_compute(*args, **kwargs):
        calls["compute"] += 1
        out = pd.DataFrame(
            {
                "dollar_adv_hist": [200.0] * len(tickers),
                "price_warmup_med": [11.0] * len(tickers),
                "volume_warmup_avg": [100.0] * len(tickers),
                "adv_window": [float(window)] * len(tickers),
                "adv_asof": [pd.Timestamp("2024-01-03")] * len(tickers),
            },
            index=pd.Index(tickers, name="ticker"),
        )
        return out

    monkeypatch.setattr(adv_mod, "compute_adv_map", _fake_compute)

    out = adv_mod.load_or_compute_adv_map(
        tickers,
        cfg,
        prices,
        volumes,
        warmup_end=None,
    )

    assert calls["compute"] == 1
    assert out.shape[0] == len(tickers)
    assert float(out.loc[tickers[0], "dollar_adv_hist"]) == 200.0
