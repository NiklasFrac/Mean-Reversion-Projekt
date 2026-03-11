import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import analysis.pipeline as pipeline
import analysis.threading_control as tc


def test_mp_bootstrap_parallel_path(monkeypatch, tmp_path: Path):
    # Build correlated prices to yield >1 candidate pair.
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    base = rng.normal(0, 0.002, size=n).cumsum()
    a = np.exp(base) * 100
    b = np.exp(base + 0.0005) * 101
    c = np.exp(base + rng.normal(0, 0.0008, size=n)) * 99
    prices = pd.DataFrame({"AAA": a, "BBB": b, "CCC": c}, index=idx)
    prices_path = tmp_path / "prices.pkl"
    prices.to_pickle(prices_path)

    pairs_path = tmp_path / "pairs.pkl"
    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_path)},
        "data_analysis": {
            "use_multiprocessing": True,
            "n_jobs": 2,
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.5,
            "pct_thr1": 0.4,
            "pct_thr2": 0.5,
            "pair_pct_threshold": 0.4,
            "pair_mean_corr_min": 0.4,
            "fdr_alpha": 0.05,
            "bootstrap": {
                "n_resamples": 60,
                "null_mean_corr": 0.4,
                "block_size": 5,
                "batch_size": 2,
            },
            "returns_cleaning": {
                "min_positive_frac": 0.9,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Fake ProcessPoolExecutor to stay in-process but exercise the parallel code path.
    submit_calls = []
    init_calls = []

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, *_, initializer=None, initargs=None, **__):
            self.initializer = initializer
            self.initargs = initargs or ()

        def __enter__(self):
            if self.initializer:
                init_calls.append(self.initializer)
                self.initializer(*self.initargs)
            return self

        def __exit__(self, *args):
            return False

        def submit(self, func, *args, **kwargs):
            submit_calls.append(func)
            return DummyFuture(func(*args, **kwargs))

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", lambda futs: futs)
    monkeypatch.setattr(pipeline, "load_config", lambda path=None: cfg)

    out = pipeline.main(cfg_path=cfg_path, quick=False, overrides=None)
    assert isinstance(out, pd.DataFrame)
    assert len(submit_calls) >= 1  # parallel path used
    assert init_calls and init_calls[0].__name__ == "_worker_init_for_mp"
    assert pairs_path.exists()


def test_thread_limiter_env_and_context(monkeypatch):
    enter_calls = []

    class FakeCtx:
        def __enter__(self):
            enter_calls.append("enter")

        def __exit__(self, exc_type, exc, tb):
            enter_calls.append("exit")

    class FakeController:
        def limit(self, limits):
            self.limits = limits
            return FakeCtx()

    fake_ne = SimpleNamespace(
        set_num_threads=lambda n: os.environ.setdefault("NE_THREADS", str(n))
    )

    monkeypatch.setattr(tc, "_TP_CONTROLLER", FakeController())
    monkeypatch.setattr(tc, "_TP_LIMIT_CTX", None, raising=False)
    monkeypatch.setattr(tc, "_ne", fake_ne)

    tc.set_thread_limits(blas_threads=2, numexpr_threads=3)
    assert os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"
    assert os.environ.get("OMP_NUM_THREADS") == "2"
    assert os.environ.get("NUMEXPR_MAX_THREADS") == "3"
    assert enter_calls == ["enter"]  # context entered once

    # Idempotent: second call should not re-enter limit context
    tc.set_thread_limits(blas_threads=2, numexpr_threads=3)
    assert enter_calls == ["enter"]


def test_mp_bootstrap_batch_failure_raises(monkeypatch, tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 140
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    base = rng.normal(0, 0.0015, size=n).cumsum()
    prices = pd.DataFrame(
        {
            "AAA": np.exp(base) * 100,
            "BBB": np.exp(base + 0.0002) * 101,
            "CCC": np.exp(base - 0.0002) * 99,
            "DDD": np.exp(base + rng.normal(0, 0.0004, size=n)) * 98,
        },
        index=idx,
    )
    prices_path = tmp_path / "prices.pkl"
    prices.to_pickle(prices_path)

    cfg = {
        "data": {
            "prices_path": str(prices_path),
            "pairs_path": str(tmp_path / "pairs.pkl"),
        },
        "data_analysis": {
            "use_multiprocessing": True,
            "n_jobs": 2,
            "rolling_window": 50,
            "rolling_step": 10,
            "pair_corr_threshold": 0.3,
            "pct_thr1": 0.3,
            "pct_thr2": 0.4,
            "pair_pct_threshold": 0.3,
            "pair_mean_corr_min": 0.2,
            "fdr_alpha": 0.1,
            "bootstrap": {
                "n_resamples": 60,
                "null_mean_corr": 0.2,
                "block_size": 5,
                "batch_size": 2,
            },
            "returns_cleaning": {
                "min_positive_frac": 0.9,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    class BrokenFuture:
        def result(self):
            raise RuntimeError("worker boom")

    class FakeExecutor:
        def __init__(self, *_, **__):
            return

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, _func, *args, **kwargs):
            return BrokenFuture()

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline, "as_completed", lambda futs: futs)
    monkeypatch.setattr(pipeline, "load_config", lambda path=None: cfg)

    with pytest.raises(RuntimeError, match="Bootstrap batch failed"):
        pipeline.main(cfg_path=tmp_path / "cfg.yaml", quick=False, overrides=None)
