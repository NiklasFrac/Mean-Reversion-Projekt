from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


from processing.config_loader import load_config
from processing.io_atomic import atomic_write_parquet, file_hash, make_manifest


def test_atomic_write_parquet_handles_exception(monkeypatch, tmp_path: Path):
    # Monkeypatch DataFrame.to_parquet -> wirft Exception -> darf nicht crashen
    called = {"ok": False}

    def _boom(*args, **kwargs):
        called["ok"] = True
        raise RuntimeError("no parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom, raising=True)

    df = pd.DataFrame({"A": [1, 2, 3]})
    out = tmp_path / "x.parquet"
    with pytest.raises(RuntimeError):
        atomic_write_parquet(df, out, compression="zstd")
    assert called["ok"] is True
    # Datei nicht erzeugt
    assert not out.exists()


def test_file_hash_and_manifest(tmp_path: Path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    # Erwarteter SHA1
    expected = hashlib.sha1(b"hello").hexdigest()
    assert file_hash(f) == expected

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    inputs = {"raw_prices": f, "raw_volume": None}
    man = make_manifest(cfg_path=cfg, inputs=inputs, extra={"k": 1})
    assert man["cfg_path"].endswith("cfg.yaml")
    assert man["inputs"]["raw_prices"]["sha1"] == expected
    # None-Path wird erlaubt und sauber serialisiert
    assert man["inputs"]["raw_volume"]["path"] is None


def test_load_config_env_override(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "x.yaml"
    cfg.write_text("alpha: 123\n", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg))
    got = load_config()
    assert got["alpha"] == 123
