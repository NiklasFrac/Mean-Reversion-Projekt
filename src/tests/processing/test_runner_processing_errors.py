from __future__ import annotations

import sys

import pytest

from processing import runner_processing


def test_cli_exits_when_config_missing(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner_processing", "--cfg", "does_not_exist.yaml"]
    )
    with pytest.raises(SystemExit) as exc:
        runner_processing.cli()
    assert exc.value.code == 2
