from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.reporting import artifacts


def test_hash_helpers(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    h_df = artifacts.hash_dataframe_sha256(df)
    assert len(h_df) == 64

    h_json = artifacts.hash_json_sha256({"x": np.float64(1.5), "y": np.int64(2)})
    assert len(h_json) == 64

    pq = tmp_path / "data.parquet"
    df.to_parquet(pq)
    h_pq = artifacts.hash_parquet_sha256(pq)
    assert len(h_pq) == 64


def test_artifact_store_load_or_build(tmp_path: Path) -> None:
    store = artifacts.ArtifactStore(root=tmp_path)
    serializer = artifacts.DataFrameCSVSerializer()
    cfg = {"alpha": 1}
    calls = {"n": 0}

    def builder() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"x": [1, 2, 3]})

    out1 = artifacts.load_or_build(
        stage=artifacts.Stage.REPORTS,
        config=cfg,
        builder=builder,
        serializer=serializer,
        store=store,
    )
    out2 = artifacts.load_or_build(
        stage=artifacts.Stage.REPORTS,
        config=cfg,
        builder=builder,
        serializer=serializer,
        store=store,
    )
    assert calls["n"] == 1
    assert out1.equals(out2)


def test_canon_df_and_json_defaults(tmp_path: Path) -> None:
    df = pd.DataFrame({"b": [1.23456789123], "a": [1.0]}, index=[2])
    canon = artifacts._canon_df(df)
    assert list(canon.columns) == ["a", "b"]
    assert float(canon.loc["2", "b"]) == round(1.23456789123, 10)

    @dataclass
    class D:
        x: int

    payload = {
        "f": np.float64(1.2),
        "i": np.int64(2),
        "p": tmp_path,
        "t": pd.Timestamp("2024-01-02"),
        "na": pd.NaT,
        "arr": np.array([1, 2]),
        "dc": D(3),
    }
    s = artifacts.to_json(payload)
    assert isinstance(s, str)


def test_serializers_and_store_exists(tmp_path: Path) -> None:
    js = artifacts.JSONSerializer()
    out_path = tmp_path / "obj.json"
    js.save({"a": 1}, out_path)
    assert js.load(out_path) == {"a": 1}

    store = artifacts.ArtifactStore(root=tmp_path)
    assert store.exists(artifacts.Stage.REPORTS, "nope", ".csv") is False
