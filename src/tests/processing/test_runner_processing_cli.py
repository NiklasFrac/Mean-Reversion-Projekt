from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def test_cli_success_calls_pipeline_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # fake config file
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("data: {}\n", encoding="utf-8")

    # fake processing.pipeline module with main()
    fake_mod = types.ModuleType("processing.pipeline")
    called: dict[str, Path] = {}

    def _main(p: Path) -> None:
        called["cfg"] = Path(p)

    fake_mod.main = _main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "processing.pipeline", fake_mod)

    # Import erst NACH dem Stub, damit der from-Import den Stub findet
    from processing.runner_processing import cli  # noqa: WPS433

    # patch argv and execute
    monkeypatch.setattr(sys, "argv", ["runner_processing.py", "--cfg", str(cfg)])
    cli()
    assert called["cfg"] == cfg


def test_cli_missing_config_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "nope.yaml"

    from processing.runner_processing import cli  # noqa: WPS433

    monkeypatch.setattr(sys, "argv", ["runner_processing.py", "--cfg", str(missing)])
    with pytest.raises(SystemExit) as ei:
        cli()
    assert ei.value.code == 2
