from __future__ import annotations

import json

import pandas as pd

from universe import exchange_source as ex
from universe import runner_universe as ru


def test_load_exchange_tickers_prefers_master_and_filters(monkeypatch, tmp_path):
    screener = tmp_path / "nasdaq_screener_2024.csv"
    screener.write_text(
        "Symbol,Name\nAAA,FirmA\nTESTX,FirmB\nBBB-WS,FirmC\nZZZ,^junk\n",
        encoding="utf-8",
    )

    res = ex.load_exchange_tickers(
        filters_cfg={
            "drop_prefixes": ["TEST"],
            "drop_suffixes": ["-WS"],
            "drop_contains": ["^"],
        },
        universe_cfg={"screener_glob": str(screener)},
    )

    assert res == ["AAA", "ZZZ"]


def test_load_exchange_tickers_uses_download_and_deduplicates(monkeypatch, tmp_path):
    screener = tmp_path / "nasdaq_screener_2024.csv"
    screener.write_text("Symbol\nAAA\nBBB\nAAA\n$BAD\n", encoding="utf-8")

    res = ex.load_exchange_tickers(
        filters_cfg={},
        universe_cfg={"screener_glob": str(screener)},
    )

    assert res == ["AAA", "BBB"]


def test_runner_main_smoke(monkeypatch, tmp_path):
    out_csv = tmp_path / "universe.csv"
    out_ext_csv = tmp_path / "universe_ext.csv"
    manifest = tmp_path / "manifest.json"
    adv_path = tmp_path / "adv.csv"
    prices_path = tmp_path / "prices.pkl"
    volumes_path = tmp_path / "volumes.pkl"

    cfg = {
        "universe": {
            "exchange": ["NYSE"],
            "output_tickers_csv": str(out_csv),
            "output_tickers_ext_csv": str(out_ext_csv),
            "manifest": str(manifest),
            "adv_cache": str(tmp_path / "adv_cache.pkl"),
        },
        "runtime": {
            "use_hashed_artifacts": False,
            "progress_bar": False,
            "persist_run_scoped_outputs": True,
            "run_scoped_outputs_dir": str(tmp_path / "by_run"),
        },
        "data": {
            "adv_path": str(adv_path),
            "raw_prices_cache": str(prices_path),
            "raw_prices_unadj_cache": str(tmp_path / "prices_unadj.pkl"),
            "raw_prices_unadj_warmup_cache": str(tmp_path / "prices_unadj_warmup.pkl"),
            "volume_path": str(volumes_path),
            "raw_volume_unadj_cache": str(tmp_path / "volumes_unadj.pkl"),
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
            "sector": ["Tech", "Health"],
            "industry": ["Software", "Pharma"],
            "country": ["US", "CA"],
        },
        index=idx,
    )

    def _fake_build(cfg, cfg_path_arg, run_id, stop_event=None):
        df_funda = df_uni.copy()
        monitoring = {"failed": []}
        extra = {
            "n_tickers_total": len(df_uni),
            "n_filtered": len(df_uni),
            "tickers_all": df_uni.index.tolist(),
            "artifacts": {},
        }
        return df_uni, df_funda, monitoring, extra

    dates = pd.date_range("2024-01-01", periods=3)
    price_panel = pd.DataFrame(
        {"AAA": [1.0, 1.1, 1.2], "BBB": [2.0, 2.1, 2.2]}, index=dates
    )
    volume_panel = pd.DataFrame(
        {"AAA": [10.0, 12.0, 11.0], "BBB": [20.0, 21.0, 19.0]}, index=dates
    )

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

    ru.main(cfg_path)

    assert out_csv.exists()
    assert out_ext_csv.exists()
    assert manifest.exists()
    run_dirs = list((tmp_path / "by_run").glob("*"))
    assert run_dirs, "run-scoped directory should have been created"
