from __future__ import annotations

import datetime as dt
import json
import os

import pandas as pd

from universe import outputs
from universe.storage import resolve_artifact_paths


def test_atomic_write_text_retries_replace(monkeypatch, tmp_path):
    target = tmp_path / "out.txt"
    calls = {"n": 0}
    real_replace = os.replace

    def _fake_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(outputs.os, "replace", _fake_replace)
    outputs.atomic_write_text(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"
    assert calls["n"] == 3


def test_write_manifest_serializes_non_json_native_values(tmp_path):
    manifest = tmp_path / "manifest.json"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("universe: {}\n", encoding="utf-8")

    outputs.write_manifest(
        manifest_path=manifest,
        cfg_path=cfg,
        cfg_hash="ABC123",
        run_id="RUN-1",
        n_initial=2,
        n_final=1,
        monitoring={"failed": []},
        extra={
            "data_policy": {"download_start_date": dt.date(2026, 1, 2)},
            "cache_path": tmp_path / "cache.pkl",
            "symbols": {"AAA", "BBB"},
        },
        schema_version="1.0.0-test",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["extra"]["data_policy"]["download_start_date"] == "2026-01-02"
    assert payload["extra"]["cache_path"].endswith("cache.pkl")
    assert sorted(payload["extra"]["symbols"]) == ["AAA", "BBB"]


def test_persist_universe_run_artifacts_parses_string_bool_for_run_scoped_flag(
    tmp_path,
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n",
        encoding="utf-8",
    )
    out_tickers = tmp_path / "tickers.csv"
    out_manifest = tmp_path / "manifest.json"
    out_tickers.write_text("ticker\nAAA\n", encoding="utf-8")
    out_manifest.write_text("{}", encoding="utf-8")

    paths = resolve_artifact_paths(
        universe_cfg={},
        data_cfg={
            "raw_prices_cache": str(tmp_path / "raw_prices.pkl"),
            "volume_path": str(tmp_path / "raw_volume.pkl"),
        },
        runtime_cfg={"run_scoped_outputs_dir": str(tmp_path / "by_run")},
    )

    outputs.persist_universe_run_artifacts(
        cfg_path=cfg_path,
        cfg_hash="HASH1234",
        run_id="RUN-OUT",
        universe_cfg={},
        runtime_cfg={"persist_run_scoped_outputs": "false"},
        data_cfg={},
        out_tickers=out_tickers,
        out_tickers_ext=None,
        out_manifest=out_manifest,
        adv_csv=None,
        adv_csv_filtered=None,
        tickers_final=["AAA"],
        df_fundamentals=pd.DataFrame(index=pd.Index(["AAA"], name="ticker")),
        df_universe=pd.DataFrame(index=pd.Index(["AAA"], name="ticker")),
        monitoring={"failed": []},
        stats={},
        artifact_paths=paths,
    )

    assert not (tmp_path / "by_run").exists()


def test_persist_universe_run_artifacts_writes_report_into_run_scoped_root(
    tmp_path,
):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "universe:\n  output_tickers_csv: out.csv\n  manifest: man.json\n",
        encoding="utf-8",
    )
    out_tickers = tmp_path / "tickers.csv"
    out_manifest = tmp_path / "manifest.json"
    out_tickers.write_text("ticker\nAAA\n", encoding="utf-8")
    out_manifest.write_text("{}", encoding="utf-8")

    paths = resolve_artifact_paths(
        universe_cfg={},
        data_cfg={
            "raw_prices_cache": str(tmp_path / "raw_prices.pkl"),
            "volume_path": str(tmp_path / "raw_volume.pkl"),
        },
        runtime_cfg={"run_scoped_outputs_dir": str(tmp_path / "by_run")},
    )

    outputs.persist_universe_run_artifacts(
        cfg_path=cfg_path,
        cfg_hash="HASH1234",
        run_id="RUN-OUT",
        universe_cfg={},
        runtime_cfg={"persist_run_scoped_outputs": "true"},
        data_cfg={},
        out_tickers=out_tickers,
        out_tickers_ext=None,
        out_manifest=out_manifest,
        adv_csv=None,
        adv_csv_filtered=None,
        tickers_final=["AAA"],
        df_fundamentals=pd.DataFrame(index=pd.Index(["AAA"], name="ticker")),
        df_universe=pd.DataFrame(index=pd.Index(["AAA"], name="ticker")),
        monitoring={"failed": []},
        stats={},
        artifact_paths=paths,
    )

    run_root = tmp_path / "by_run" / "RUN-OUT_HASH1234"
    report_path = run_root / "report.md"
    assert report_path.exists()
    assert "Universe Run Report" in report_path.read_text(encoding="utf-8")
