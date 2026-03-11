import builtins
import os
import subprocess
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.data_analysis as da


def test_init_prometheus_import_error(monkeypatch):
    monkeypatch.setattr(da, "_PROM", {}, raising=False)
    monkeypatch.setattr(da, "_PROM_REG", None, raising=False)
    monkeypatch.setattr(da, "_PROM_RUN_ID", None, raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("missing prometheus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = {"monitoring": {"prometheus": {"enabled": True}}}
    da.init_prometheus(cfg)
    assert da._PROM == {} and da._PROM_REG is None and da._PROM_RUN_ID is None


def test_init_prometheus_port_busy_and_hash_fallback(monkeypatch):
    class DummyMetric:
        def observe(self, *args, **kwargs):
            return self

        def set(self, *args, **kwargs):
            return self

        def labels(self, **kwargs):
            return self

    fake_module = types.SimpleNamespace(
        CollectorRegistry=lambda: object(),
        Gauge=lambda *a, **k: DummyMetric(),
        Histogram=lambda *a, **k: DummyMetric(),
        PlatformCollector=lambda *a, **k: None,
        ProcessCollector=lambda *a, **k: None,
        start_http_server=lambda *a, **k: (_ for _ in ()).throw(OSError("busy")),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", fake_module)
    monkeypatch.setattr(
        da,
        "dict_hash_sha256",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("hashfail")),
    )
    monkeypatch.setattr(da, "_PROM", {}, raising=False)
    monkeypatch.setattr(da, "_PROM_REG", None, raising=False)
    monkeypatch.setattr(da, "_PROM_RUN_ID", None, raising=False)

    cfg = {"monitoring": {"prometheus": {"enabled": True, "port": 9999}}}
    da.init_prometheus(cfg)
    assert da._PROM == {} and da._PROM_REG is None and da._PROM_RUN_ID is None


def test_prom_observe_handles_metric_failure(monkeypatch):
    class Broken:
        def observe(self, *args, **kwargs):
            raise RuntimeError("boom")

        def labels(self, **kwargs):
            return self

        def set(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(da, "_PROM", {"runtime": Broken()}, raising=False)
    monkeypatch.setattr(da, "_PROM_RUN_ID", "run1", raising=False)
    da.prom_observe({"stage": 0.1}, 3)  # should swallow exceptions


def test_fsync_and_file_hash_missing_path(tmp_path: Path):
    p = tmp_path / "file.txt"
    p.write_text("content", encoding="utf-8")
    da._fsync_dir(p)
    assert da.file_sha256(tmp_path / "missing.txt") is None


def test_load_config_handles_directory_path(monkeypatch, tmp_path: Path):
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(da.Path, "exists", lambda self: self == cfg_dir)
    with pytest.raises(FileNotFoundError):
        da.load_config(cfg_dir)


def test_ensure_utc_index_integer_and_tz_conversion():
    df_int = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 2, 3])
    with pytest.raises(ValueError):
        da.ensure_utc_index(df_int)

    idx = pd.date_range("2023-01-01", periods=2, tz="Europe/Berlin")
    df = pd.DataFrame({"a": [1, 2]}, index=idx)
    out = da.ensure_utc_index(df)
    assert out.index.tz and str(out.index.tz) in {"Europe/Berlin", "CET"}


def test_dedup_columns_and_canon_pair():
    df = pd.DataFrame([[1, 2, 3]], columns=[1, "1", 2])
    deduped = da._dedup_str_columns(df)
    assert list(deduped.columns) == ["1", "2"]
    assert da._canon_pair("Z", "A") == ("A", "Z")


def test_select_price_columns_aggregates_duplicate_multiindex():
    cols = pd.MultiIndex.from_tuples([("AAA", "close"), ("AAA", "close")])
    df = pd.DataFrame([[1, 2], [3, 4]], columns=cols)
    out = da.select_price_columns(df)
    assert list(out.columns) == ["AAA"]
    assert list(out.iloc[0]) == [2] and list(out.iloc[1]) == [4]


def test_load_filled_data_unknown_and_bad_csv(tmp_path: Path):
    bad = tmp_path / "prices.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        da.load_filled_data(bad)

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("a,b\nfoo,1\nbar,2\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            da.load_filled_data(bad_csv)
    assert not any("Could not infer format" in str(item.message) for item in caught)


def test_compute_log_returns_all_policy_keeps_partial_nan_rows():
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "A": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "B": [1.0, np.nan, 1.1, np.nan, 1.2, 1.3],
            "C": [1.0, np.nan, np.nan, np.nan, np.nan, 1.1],
        },
        index=idx,
    )
    logret = da.compute_log_returns(
        df,
        min_positive_frac=0.0,
        max_nan_frac_cols=0.9,
        drop_policy_rows="all",
    )
    assert "C" not in logret.columns
    assert "B" in logret.columns
    assert bool(logret["B"].isna().any())


def test_pairs_and_clustering_edge_cases():
    assert da.list_high_pairs_vectorized(pd.DataFrame()) == []

    corr = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], columns=["X", "Y"], index=["X", "Y"])
    assert da.pairs_df_from_corr(corr, threshold=0.9).empty

    with pytest.raises(ValueError):
        da.hierarchical_clustering_from_corr(pd.DataFrame())

    bad = pd.DataFrame(
        [[1.0, np.nan], [np.nan, 1.0]], columns=["A", "B"], index=["A", "B"]
    )
    with pytest.raises(ValueError):
        da.hierarchical_clustering_from_corr(bad)


def test_rolling_pair_metrics_fast_zero_valid():
    idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    vals = np.arange(8, dtype=float)
    logrets = pd.DataFrame({"A": vals, "B": vals[::-1]}, index=idx)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary, window_cors = da.rolling_pair_metrics_fast(
            logrets,
            candidate_pairs=[("B", "A"), ("A", "B")],
            window=4,
            step=4,
            thr1=0.5,
            thr2=0.6,
        )
    assert "A-B" in window_cors
    assert summary.loc[0, "pair"] == "A-B"
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_stationary_bootstrap_indices_edges():
    rng = np.random.default_rng(0)
    assert da._stationary_bootstrap_indices(0, 5, rng).size == 0
    rng2 = np.random.default_rng(1)
    res = da._stationary_bootstrap_indices(5, 1, rng2)
    assert res.size == 5


def test_benjamini_yekutieli_behaviour():
    assert da.benjamini_yekutieli([]) == []
    res = da.benjamini_yekutieli([0.01, 0.04, 0.8], alpha=0.2)
    assert any(res)


def test_bootstrap_worker_and_batch_defaults():
    idx, pair, boot = da._bootstrap_worker(0, "A-B", None, 0.5, 0.0, 5, 7, None, 0.95)
    assert idx == 0 and pair == "A-B" and boot["pval"] == 1.0

    tasks = [(0, "A-B", np.array([0.1, 0.2]))]
    results = da._bootstrap_batch_worker(tasks, 0.1, 0.0, 3, 9, None, 0.95)
    assert results and results[0][0] == 0 and results[0][1] == "A-B"


def test_set_thread_limits_fallback(monkeypatch):
    monkeypatch.setattr(da, "_TP_CONTROLLER", None, raising=False)
    fake_ne = types.SimpleNamespace(
        set_num_threads=lambda n: (_ for _ in ()).throw(RuntimeError("fail"))
    )
    monkeypatch.setattr(da, "_ne", fake_ne, raising=False)
    for var in (
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)
    da.set_thread_limits(blas_threads=1, numexpr_threads=2)
    assert os.environ.get("MKL_NUM_THREADS") == "1"
    assert os.environ.get("OMP_NUM_THREADS") == "1"


def test_get_git_sha_env_fallback(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no git")),
    )
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    assert da.get_git_sha() == "abc123"


def test_load_filled_data_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        da.load_filled_data(tmp_path / "missing.pkl")


def test_compute_helpers_empty_inputs():
    assert da.compute_shrink_corr(pd.DataFrame()).empty
    assert da.compute_log_returns(pd.DataFrame()).empty


def test_pairs_df_from_corr_empty_branch():
    empty = da.pairs_df_from_corr(pd.DataFrame())
    assert list(empty.columns) == ["left", "right", "corr"]


def test_benjamini_yekutieli_all_false():
    out = da.benjamini_yekutieli([0.9, 0.8, 0.7], alpha=0.05)
    assert out == [False, False, False]


def test_save_results_meta_failure(monkeypatch, tmp_path: Path):
    df = pd.DataFrame({"pair": ["A-B"], "mean_corr": [0.1]})
    out_path = tmp_path / "pairs.pkl"
    monkeypatch.setattr(
        da,
        "_atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("meta fail")),
    )
    da.save_results(df, out_path, {"meta": True})
    assert out_path.exists()


def test_worker_init_handles_errors(monkeypatch):
    monkeypatch.setattr(
        da,
        "set_thread_limits",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    da._worker_init_for_mp(1, 1)
