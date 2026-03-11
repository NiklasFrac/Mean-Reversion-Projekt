from __future__ import annotations

from pathlib import Path


from processing.config_loader import load_config


def test_load_config_prefers_env_path(tmp_path: Path, monkeypatch):
    # Projekt-Dateien, die es NICHT sein sollen
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "config.yaml").write_text("a: 1\n", encoding="utf-8")

    # Eigentliche gewünschte Config via ENV
    env_cfg = tmp_path / "override.yaml"
    env_cfg.write_text("answer: 42\n", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(env_cfg))

    # Arbeitsverzeichnis auf tmp_path setzen, damit Kandidatenliste dort sucht
    monkeypatch.chdir(tmp_path)

    cfg = load_config(None)
    assert cfg.get("answer") == 42


def test_load_config_explicit_path_beats_env(tmp_path: Path, monkeypatch):
    explicit_cfg = tmp_path / "explicit.yaml"
    explicit_cfg.write_text("answer: 7\n", encoding="utf-8")

    env_cfg = tmp_path / "override.yaml"
    env_cfg.write_text("answer: 42\n", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(env_cfg))

    cfg = load_config(explicit_cfg)
    assert cfg.get("answer") == 7
