from __future__ import annotations

from pathlib import Path

import pandas as pd

import processing.io_atomic as ioa


def test_atomic_writes_trigger_fsync(monkeypatch, tmp_path: Path):
    fsync_calls: list[int] = []

    def _fsync(fd: int) -> None:
        fsync_calls.append(int(fd))

    monkeypatch.setattr(ioa.os, "fsync", _fsync, raising=True)

    # json
    ioa.atomic_write_json({"a": 1}, tmp_path / "x.json")
    # pickle
    ioa.atomic_write_pickle({"a": 1}, tmp_path / "x.pkl")

    # parquet (stub writer to avoid optional engine dependency in tests)
    def _fake_to_parquet(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PARQ")
        return None

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=True)
    ioa.atomic_write_parquet(pd.DataFrame({"A": [1, 2]}), tmp_path / "x.parquet")

    # At least one fsync per artifact write path (in practice more).
    assert len(fsync_calls) >= 3
