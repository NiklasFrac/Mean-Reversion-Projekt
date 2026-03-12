from __future__ import annotations

from pathlib import Path

import pandas as pd


from processing.io_atomic import atomic_write_parquet


def test_atomic_write_parquet_happy_path(monkeypatch, tmp_path: Path):
    # fake to_parquet that simply writes bytes (without pyarrow/fastparquet)
    def fake_to_parquet(self, path, *args, **kwargs):
        p = Path(path)
        p.write_bytes(b"PARQ")
        return None

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)

    df = pd.DataFrame({"A": [1, 2, 3]})
    out = tmp_path / "filled.parquet"
    atomic_write_parquet(df, out, compression="zstd")
    assert out.exists() and out.read_bytes() == b"PARQ"
