from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


from processing.io_atomic import atomic_write_parquet


def test_atomic_write_parquet_logs_warning_on_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
):
    df = pd.DataFrame({"x": [1, 2, 3]})
    target = tmp_path / "out.parquet"

    def boom(*args, **kwargs):
        raise RuntimeError("to_parquet failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom, raising=True)
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        atomic_write_parquet(df, target, compression="zstd")
    # no file created, but WARNING logged
    assert not target.exists()
    assert any("Parquet atomic write failed" in rec.message for rec in caplog.records)
