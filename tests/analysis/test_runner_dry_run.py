# src/tests/analysis/test_runner_dry_run.py
import json
import sys

from analysis.runner_analysis import main as runner_main


def test_runner_dry_run(tmp_path, monkeypatch, capsys):
    cfg = {
        "data": {
            "prices_path": str(tmp_path / "p.pkl"),
            "pairs_path": str(tmp_path / "out.pkl"),
        }
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("LINES", "40")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    sys.argv = ["runner_analysis.py", "--cfg", str(p), "--dry-run"]
    rc = runner_main()
    assert rc == 0
