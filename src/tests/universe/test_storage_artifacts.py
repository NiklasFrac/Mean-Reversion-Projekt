from __future__ import annotations

import pandas as pd

from universe import storage
from universe.artifact_defaults import DEFAULT_CHECKPOINT_PATH


def test_artifact_targets_resolve_mirrors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "raw_prices_cache": str(tmp_path / "canonical_prices.pkl"),
        "volume_path": str(tmp_path / "canonical_vols.pkl"),
    }

    prices_path, vols_path, mirror_prices, mirror_vols = storage.artifact_targets(
        hashed=True,
        data_cfg=cfg,
        price_bytes=b"price-bytes",
        vol_bytes=b"vol-bytes",
    )

    assert prices_path.parent == tmp_path
    assert vols_path.parent == tmp_path
    assert mirror_prices == tmp_path / "canonical_prices.pkl"
    assert mirror_vols == tmp_path / "canonical_vols.pkl"
    assert "canonical_prices." in prices_path.name
    assert "canonical_vols." in vols_path.name


def test_artifact_targets_non_hashed_uses_canonical_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "raw_prices_cache": str(tmp_path / "canonical_prices.pkl"),
        "volume_path": str(tmp_path / "canonical_vols.pkl"),
    }
    prices_path, vols_path, mirror_prices, mirror_vols = storage.artifact_targets(
        hashed=False,
        data_cfg=cfg,
    )
    assert prices_path == tmp_path / "canonical_prices.pkl"
    assert vols_path == tmp_path / "canonical_vols.pkl"
    assert mirror_prices == tmp_path / "canonical_prices.pkl"
    assert mirror_vols == tmp_path / "canonical_vols.pkl"


def test_write_event_artifact_writes_hashed_and_canonical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2024-01-01"]))
    canonical = tmp_path / "canonical.pkl"

    hashed = storage.write_event_artifact(
        name="raw_dividends",
        df=df,
        canonical_path=canonical,
        hashed=True,
    )

    assert hashed is not None
    assert hashed.exists()
    assert canonical.exists()
    assert hashed.name.startswith("raw_dividends.")


def test_ensure_updated_at_column_adds_missing():
    df = pd.DataFrame({"price": [1.0]}, index=pd.Index(["AAA"], name="ticker"))
    res = storage.ensure_updated_at_column(df.copy())
    assert "updated_at" in res.columns
    assert res["updated_at"].isna().all()


def test_load_fundamentals_store_reads_parquet_then_pickle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"price": [1.0]}, index=pd.Index(["AAA"], name="ticker"))
    parquet_path = tmp_path / "fundamentals.parquet"
    df.to_parquet(parquet_path)
    loaded = storage.load_fundamentals_store(parquet_path)
    assert loaded.index.tolist() == ["AAA"]
    # Ensure fallback to pickle when parquet missing
    pickle_path = parquet_path.with_suffix(".pkl")
    df.to_pickle(pickle_path)
    loaded_pickle = storage.load_fundamentals_store(parquet_path)
    assert loaded_pickle.index.tolist() == ["AAA"]


def test_save_fundamentals_store_falls_back_to_pickle(tmp_path, monkeypatch):
    df = pd.DataFrame({"price": [1.0]}, index=pd.Index(["AAA"], name="ticker"))
    target = tmp_path / "funda.parquet"

    def boom_parquet(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom_parquet)

    used_path = storage.save_fundamentals_store(df.copy(), target)
    assert used_path.suffix == ".pkl"
    assert used_path.exists()


def test_build_event_frame_normalizes_timezones():
    idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
    event_map = {"AAA": pd.Series([1, 2], index=idx)}
    frame = storage.build_event_frame(event_map)
    assert frame.index.tz is None
    assert frame.loc["2024-01-01", "AAA"] == 1


def test_resolve_artifact_paths_respects_explicit_checkpoint_disable():
    disabled_none = storage.resolve_artifact_paths(
        runtime_cfg={"checkpoint_path": None}
    )
    assert disabled_none.checkpoint_path is None

    disabled_literal = storage.resolve_artifact_paths(
        runtime_cfg={"checkpoint_path": "null"}
    )
    assert disabled_literal.checkpoint_path is None

    defaulted = storage.resolve_artifact_paths(runtime_cfg={})
    assert str(defaulted.checkpoint_path).replace("\\", "/") == DEFAULT_CHECKPOINT_PATH


def test_resolve_artifact_paths_adv_path_none_uses_default_paths():
    paths = storage.resolve_artifact_paths(data_cfg={"adv_path": None})
    assert str(paths.adv_csv).replace("\\", "/") == "runs/data/adv_map_usd.csv"
    assert (
        str(paths.adv_csv_filtered).replace("\\", "/")
        == "runs/data/adv_map_usd_filtered.csv"
    )
