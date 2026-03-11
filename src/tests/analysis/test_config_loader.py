# src/tests/analysis/test_config_loader.py
import pytest

from analysis.config_io import load_config


def test_load_config_env_var(tmp_path, monkeypatch):
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text("k: { v: 1 }\n", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfgp))
    cfg = load_config(None)
    assert cfg["k"]["v"] == 1


def test_load_config_file_not_found(tmp_path, monkeypatch):
    # Leeres Arbeitsverzeichnis ohne configs -> Fehler
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    with pytest.raises(FileNotFoundError):
        load_config(None)
