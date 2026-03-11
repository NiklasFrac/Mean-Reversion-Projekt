import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.config_io import deep_merge, dict_hash_sha256, file_sha256
from analysis.pipeline import main
from analysis.threading_control import set_thread_limits


@pytest.mark.unit
def test_deep_merge_merges_recursively():
    a = {"a": 1, "b": {"x": 1, "y": 2}}
    b = {"b": {"y": 9, "z": 3}, "c": 7}
    out = deep_merge(a, b)
    assert out == {"a": 1, "b": {"x": 1, "y": 9, "z": 3}, "c": 7}


@pytest.mark.unit
def test_set_thread_limits_sets_env(monkeypatch):
    for k in (
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        monkeypatch.delenv(k, raising=False)
    set_thread_limits(blas_threads=2, numexpr_threads=3)
    import os

    assert os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"
    assert os.environ.get("OMP_NUM_THREADS") == "2"
    assert os.environ.get("NUMEXPR_MAX_THREADS") == "3"


@pytest.mark.unit
def test_file_and_dict_hash(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 and h1 == h2

    d1 = {"x": 1, "y": [1, 2]}
    d2 = {"y": [1, 2], "x": 1}  # andere Reihenfolge → gleicher Hash
    assert dict_hash_sha256(d1) == dict_hash_sha256(d2)


@pytest.mark.unit
def test_main_param_guards_raise(monkeypatch, tmp_path: Path):
    # Minimal-Konfig + kleine rolling_window, um Guard zu triggern
    prices = pd.DataFrame(
        {"A": np.exp(np.cumsum(np.zeros(60)))},
        index=pd.date_range("2020-01-01", periods=60, tz="UTC"),
    )
    prices_path = tmp_path / "p.pkl"
    prices.to_pickle(prices_path)

    pairs_out = tmp_path / "pairs.pkl"
    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_out)},
        "data_analysis": {
            "rolling_window": 20,
            "rolling_step": 1,
            # Trigger a hard guard (too few resamples).
            "bootstrap": {"n_resamples": 10},
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }
    cfg_path = tmp_path / "cfg_guard.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg_path))

    with pytest.raises(ValueError):
        main(cfg_path=None, quick=False, overrides=None)
