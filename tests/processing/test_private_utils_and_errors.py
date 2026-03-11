from __future__ import annotations

import importlib
import pickle
from pathlib import Path

import pandas as pd
import pytest

import processing.io_atomic as io_mod
import processing.pipeline as pipe_mod
import processing.raw_loader as raw_mod


def test__tmp_path_for_appends_tmp_suffix(tmp_path: Path):
    io = importlib.reload(io_mod)
    p = tmp_path / "x.parquet"
    got = io._tmp_path_for(p)
    assert got.parent == p.parent
    assert got.name.startswith("x.parquet.tmp.")


def test__git_functions_fail_safe(monkeypatch):
    # Modul frisch laden, damit evtl. globale Patches aus anderen Tests verschwinden
    io = importlib.reload(io_mod)

    def boom(*args, **kwargs):
        raise RuntimeError("no git")

    monkeypatch.setattr(io.subprocess, "check_output", boom, raising=True)
    assert io._git_commit() is None
    assert io._git_dirty() is None


def test__pip_freeze_fail_safe(monkeypatch):
    io = importlib.reload(io_mod)

    def boom(*args, **kwargs):
        raise RuntimeError("no pip")

    monkeypatch.setattr(io.subprocess, "check_output", boom, raising=True)
    path, sha = io._pip_freeze()
    assert path == "" and sha is None


def test__load_any_prices_raises_on_unsupported(tmp_path: Path):
    raw = raw_mod
    p = tmp_path / "weird.pkl"
    with p.open("wb") as f:
        pickle.dump([1, 2, 3], f)  # kein DataFrame, kein dict -> TypeError erwartet
    with pytest.raises(TypeError):
        raw._load_any_prices(p)


def test__load_any_prices_dict_pickle(tmp_path: Path):
    raw = raw_mod
    p = tmp_path / "ok.pkl"
    payload = {"A": [1.0, 2.0], "B": pd.Series([3.0, 4.0])}
    with p.open("wb") as f:
        pickle.dump(payload, f)
    df = raw._load_any_prices(p)
    assert list(df.columns) == ["A", "B"]
    assert df.shape == (2, 2)


def test__discover_handles_exception(monkeypatch):
    raw = raw_mod

    def bad_glob(*args, **kwargs):
        raise RuntimeError("glob failed")

    # Path.glob wirft -> _discover soll None zurückgeben
    from pathlib import Path as _P

    monkeypatch.setattr(_P, "glob", bad_glob, raising=True)
    assert raw._discover("*.x") is None


def test__normalize_mlflow_tracking_uri_windows_file_schemes():
    pipe = pipe_mod
    assert (
        pipe._normalize_mlflow_tracking_uri(r"file://C:\tmp\mlruns", is_windows=True)
        == r"file:C:\tmp\mlruns"
    )
    assert (
        pipe._normalize_mlflow_tracking_uri(r"file:///C:/tmp/mlruns", is_windows=True)
        == "file:C:/tmp/mlruns"
    )
    assert (
        pipe._normalize_mlflow_tracking_uri(r"C:\tmp\mlruns", is_windows=True)
        == r"file:C:\tmp\mlruns"
    )


def test__default_mlflow_tracking_uri_uses_sqlite(tmp_path: Path, monkeypatch):
    pipe = pipe_mod
    monkeypatch.chdir(tmp_path)
    uri = pipe._default_mlflow_tracking_uri()
    assert uri.startswith("sqlite:///")
    assert "runs/metadata/mlflow.db" in uri


def test__default_mlflow_tracking_uri_anchors_to_project_root_from_src(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pipe = pipe_mod
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path / "src")
    uri = pipe._default_mlflow_tracking_uri()
    assert (
        uri == f"sqlite:///{(tmp_path / 'runs' / 'metadata' / 'mlflow.db').as_posix()}"
    )


def test__resolve_mlflow_tracking_uri_resolves_repo_relative_sqlite_path(
    tmp_path: Path,
):
    pipe = pipe_mod
    uri = pipe._resolve_mlflow_tracking_uri(
        "sqlite:///runs/metadata/mlflow.db",
        project_root=tmp_path,
    )
    assert (
        uri == f"sqlite:///{(tmp_path / 'runs' / 'metadata' / 'mlflow.db').as_posix()}"
    )


def test__get_tradable_mask_cached_reuses_mask(monkeypatch):
    pipe = pipe_mod
    idx = pd.date_range("2024-01-01", periods=5, tz="America/New_York")
    calls = {"n": 0}

    def _fake_build(index, *, calendar_code="XNYS", rth_only=True):
        calls["n"] += 1
        return pd.Series(True, index=index)

    monkeypatch.setattr(pipe, "build_tradable_mask", _fake_build, raising=True)

    cache: dict[
        tuple[str, bool, int, str | None, str | None, str],
        tuple[pd.DatetimeIndex, pd.Series],
    ] = {}
    m1 = pipe._get_tradable_mask_cached(
        index=idx, calendar_code="XNYS", rth_only=True, cache=cache
    )
    m2 = pipe._get_tradable_mask_cached(
        index=idx, calendar_code="XNYS", rth_only=True, cache=cache
    )

    assert calls["n"] == 1
    assert m1.equals(m2)
