from __future__ import annotations

import os
import time
from pathlib import Path


from processing.raw_loader import _discover


def test__discover_picks_most_recent(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # three files, different mtimes
    f1 = tmp_path / "raw_prices.1.pkl"
    f2 = tmp_path / "raw_prices.2.pkl"
    f3 = tmp_path / "raw_prices.3.pkl"
    for f in (f1, f2, f3):
        f.write_bytes(b"x")
        time.sleep(0.01)  # sicheres mtime Delta

    # mtime anheben: f2 neuester
    os.utime(f2, (time.time() + 10, time.time() + 10))

    got = _discover("raw_prices.*.pkl")
    assert got is not None
    assert got.name == "raw_prices.2.pkl"
