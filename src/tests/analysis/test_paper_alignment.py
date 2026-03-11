import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.pipeline as pipeline
from analysis.utils import _pct_label


def _write_prices(path: Path, *, tz: str = "UTC", n: int = 180) -> None:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz=tz)
    base = rng.normal(0.0, 0.01, size=n).cumsum()
    df = pd.DataFrame(
        {
            "A": 100.0 * np.exp(base),
            "B": 99.0 * np.exp(base + rng.normal(0.0, 0.001, size=n)),
        },
        index=idx,
    )
    df.to_pickle(path)


def test_k_windows_zero_pairs_are_excluded_before_bootstrap(
    monkeypatch, tmp_path: Path
):
    prices_path = tmp_path / "prices.pkl"
    _write_prices(prices_path)
    out_path = tmp_path / "pairs.pkl"

    thr1, thr2 = 0.7, 0.8
    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)

    monkeypatch.setattr(
        pipeline,
        "list_high_pairs_vectorized",
        lambda *_a, **_k: [("A", "B", 0.9), ("C", "D", 0.9)],
    )

    def fake_rolling(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "pair": ["A-B", "C-D"],
                "mean_corr_raw": [0.81, 0.95],
                f"{lbl1}_raw": [88.0, 99.0],
                f"{lbl2}_raw": [76.0, 99.0],
                "mean_corr": [0.81, 0.95],
                lbl1: [88.0, 99.0],
                lbl2: [76.0, 99.0],
            }
        )
        window_cors = {
            "A-B": np.array([0.79, 0.83, 0.81], dtype=float),
            "C-D": np.array([np.nan, np.nan, np.nan], dtype=float),
        }
        return summary, window_cors

    monkeypatch.setattr(pipeline, "rolling_pair_metrics_fast", fake_rolling)

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 50,
            "pair_corr_threshold": 0.3,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "pair_pct_threshold1": 0.0,
            "pair_pct_threshold2": 0.0,
            "max_candidates": 100,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.7, "block_size": 5},
            "fdr_alpha": 0.2,
            "n_jobs": 1,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    run_pairs = Path(meta["outputs"]["run_scoped_pairs_path"])
    boot = pd.read_parquet(run_pairs.parent / "bootstrap_results.parquet")

    assert meta["stats"]["n_candidates"] == 1
    assert boot.shape[0] == 1
    assert boot["pair"].tolist() == ["A-B"]


def test_pair_mean_corr_min_is_ignored_for_paper_aligned_selection(
    monkeypatch, tmp_path: Path
):
    prices_path = tmp_path / "prices.pkl"
    _write_prices(prices_path)
    out_path = tmp_path / "pairs.pkl"

    thr1, thr2 = 0.7, 0.8
    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)

    monkeypatch.setattr(
        pipeline,
        "list_high_pairs_vectorized",
        lambda *_a, **_k: [("A", "B", 0.9)],
    )

    def fake_rolling(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "pair": ["A-B"],
                "mean_corr_raw": [0.20],  # deliberately below pair_mean_corr_min
                f"{lbl1}_raw": [95.0],
                f"{lbl2}_raw": [90.0],
                "mean_corr": [0.20],
                lbl1: [95.0],
                lbl2: [90.0],
            }
        )
        window_cors = {"A-B": np.array([0.2, 0.21, 0.19], dtype=float)}
        return summary, window_cors

    monkeypatch.setattr(pipeline, "rolling_pair_metrics_fast", fake_rolling)
    monkeypatch.setattr(
        pipeline,
        "bootstrap_pair_stats",
        lambda *_a, **_k: {
            "mean_corr": 0.20,
            "mean_corr_ci": (0.15, 0.25),
            "pct_above_thr1": 95.0,
            "pct1_ci": (90.0, 100.0),
            "pval": 0.0,
            "k_windows": 3,
            "n_resamples_eff": 60,
            "block_size_eff": 5,
            "ci_level": 0.95,
        },
    )

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 50,
            "pair_corr_threshold": 0.3,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "pair_pct_threshold1": 0.0,
            "pair_pct_threshold2": 0.0,
            "pair_mean_corr_min": 0.95,  # must be ignored
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.7, "block_size": 5},
            "fdr_alpha": 0.2,
            "n_jobs": 1,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    selected = pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    assert not selected.empty
    assert selected["pair"].tolist() == ["A-B"]


def test_disable_max_candidates_switch_evaluates_all_candidates(
    monkeypatch, tmp_path: Path
):
    prices_path = tmp_path / "prices.pkl"
    _write_prices(prices_path)
    out_path = tmp_path / "pairs.pkl"

    thr1, thr2 = 0.7, 0.8
    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)

    monkeypatch.setattr(
        pipeline,
        "list_high_pairs_vectorized",
        lambda *_a, **_k: [("A", "B", 0.9), ("C", "D", 0.9), ("E", "F", 0.9)],
    )

    def fake_rolling(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "pair": ["A-B", "C-D", "E-F"],
                "mean_corr_raw": [0.82, 0.81, 0.80],
                f"{lbl1}_raw": [90.0, 88.0, 86.0],
                f"{lbl2}_raw": [85.0, 83.0, 81.0],
                "mean_corr": [0.82, 0.81, 0.80],
                lbl1: [90.0, 88.0, 86.0],
                lbl2: [85.0, 83.0, 81.0],
            }
        )
        window_cors = {
            "A-B": np.array([0.80, 0.82, 0.84], dtype=float),
            "C-D": np.array([0.79, 0.81, 0.83], dtype=float),
            "E-F": np.array([0.78, 0.80, 0.82], dtype=float),
        }
        return summary, window_cors

    monkeypatch.setattr(pipeline, "rolling_pair_metrics_fast", fake_rolling)
    monkeypatch.setattr(
        pipeline,
        "bootstrap_pair_stats",
        lambda *_a, **_k: {
            "mean_corr": 0.8,
            "mean_corr_ci": (0.75, 0.85),
            "pct_above_thr1": 90.0,
            "pct1_ci": (85.0, 95.0),
            "pval": 0.0,
            "k_windows": 3,
            "n_resamples_eff": 60,
            "block_size_eff": 5,
            "ci_level": 0.95,
        },
    )

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 50,
            "pair_corr_threshold": 0.3,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "pair_pct_threshold1": 0.0,
            "pair_pct_threshold2": 0.0,
            "max_candidates": 1,
            "disable_max_candidates": True,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.7, "block_size": 5},
            "fdr_alpha": 0.2,
            "n_jobs": 1,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    selected = pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    run_pairs = Path(meta["outputs"]["run_scoped_pairs_path"])
    boot = pd.read_parquet(run_pairs.parent / "bootstrap_results.parquet")

    assert meta["data_analysis"]["max_candidates_disabled"] is True
    assert meta["data_analysis"]["max_candidates"] is None
    assert meta["stats"]["n_candidates"] == 3
    assert boot.shape[0] == 3
    assert set(boot["pair"].tolist()) == {"A-B", "C-D", "E-F"}
    assert selected.shape[0] == 3


def test_pipeline_preserves_input_timezone_domain(tmp_path: Path):
    prices_path = tmp_path / "prices_ny.pkl"
    _write_prices(prices_path, tz="America/New_York")
    out_path = tmp_path / "pairs.pkl"

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "train_start_local": "2024-01-15",
            "train_cutoff_local": "2024-06-01",
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.2,
            "pct_thr1": 0.2,
            "pct_thr2": 0.2,
            "pair_pct_threshold1": 0.0,
            "pair_pct_threshold2": 0.0,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.1, "block_size": 5},
            "fdr_alpha": 0.5,
            "n_jobs": 1,
            "rng_seed": 3,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["time"]["tz_policy"] == "preserve_input_index_tz"
    assert meta["time"]["index_tz"] in {"America/New_York", "EST5EDT"}
    assert isinstance(meta["time"]["train_start_effective"], str)
    assert not meta["time"]["train_start_effective"].endswith("Z")


def test_selection_can_skip_bootstrap_and_significance(monkeypatch, tmp_path: Path):
    prices_path = tmp_path / "prices.pkl"
    _write_prices(prices_path)
    out_path = tmp_path / "pairs.pkl"

    thr1, thr2 = 0.7, 0.8
    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)

    monkeypatch.setattr(
        pipeline,
        "list_high_pairs_vectorized",
        lambda *_a, **_k: [("A", "B", 0.9), ("C", "D", 0.9)],
    )

    def fake_rolling(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "pair": ["A-B", "C-D"],
                "mean_corr_raw": [0.82, 0.79],
                f"{lbl1}_raw": [90.0, 50.0],
                f"{lbl2}_raw": [80.0, 20.0],
                "mean_corr": [0.82, 0.79],
                lbl1: [90.0, 50.0],
                lbl2: [80.0, 20.0],
            }
        )
        window_cors = {
            "A-B": np.array([0.80, 0.82, 0.84], dtype=float),
            "C-D": np.array([0.78, 0.79, 0.80], dtype=float),
        }
        return summary, window_cors

    monkeypatch.setattr(pipeline, "rolling_pair_metrics_fast", fake_rolling)

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 50,
            "pair_corr_threshold": 0.3,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "pair_pct_threshold1": 0.6,
            "pair_pct_threshold2": 0.3,
            "enable_bootstrap": False,
            "enable_hypothesis_test": False,
            "enable_fdr": False,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.7, "block_size": 5},
            "fdr_alpha": 0.2,
            "n_jobs": 1,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    selected = pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    run_pairs = Path(meta["outputs"]["run_scoped_pairs_path"])
    boot = pd.read_parquet(run_pairs.parent / "bootstrap_results.parquet")

    assert selected["pair"].tolist() == ["A-B"]
    assert meta["data_analysis"]["enable_bootstrap"] is False
    assert meta["data_analysis"]["enable_hypothesis_test"] is False
    assert meta["data_analysis"]["enable_fdr"] is False
    assert meta["data_analysis"]["significance_mode"] == "disabled"
    assert boot["selected"].tolist() == [True, False]
    assert boot["pval"].isna().all()
    assert (boot["n_resamples_eff"] == 0).all()


def test_selection_can_use_unadjusted_pvalues_without_fdr(monkeypatch, tmp_path: Path):
    prices_path = tmp_path / "prices.pkl"
    _write_prices(prices_path)
    out_path = tmp_path / "pairs.pkl"

    thr1, thr2 = 0.7, 0.8
    lbl1, lbl2 = _pct_label(thr1), _pct_label(thr2)

    monkeypatch.setattr(
        pipeline,
        "list_high_pairs_vectorized",
        lambda *_a, **_k: [("A", "B", 0.9), ("C", "D", 0.9)],
    )

    def fake_rolling(*_args, **_kwargs):
        summary = pd.DataFrame(
            {
                "pair": ["A-B", "C-D"],
                "mean_corr_raw": [0.81, 0.80],
                f"{lbl1}_raw": [95.0, 95.0],
                f"{lbl2}_raw": [90.0, 90.0],
                "mean_corr": [0.81, 0.80],
                lbl1: [95.0, 95.0],
                lbl2: [90.0, 90.0],
            }
        )
        window_cors = {
            "A-B": np.array([0.80, 0.82, 0.84], dtype=float),
            "C-D": np.array([0.79, 0.81, 0.83], dtype=float),
        }
        return summary, window_cors

    monkeypatch.setattr(pipeline, "rolling_pair_metrics_fast", fake_rolling)

    pvals = {"A-B": 0.01, "C-D": 0.40}

    def fake_bootstrap(cors, **_kwargs):
        pair_name = "A-B" if float(np.asarray(cors, dtype=float)[0]) >= 0.80 else "C-D"
        return {
            "mean_corr": float(np.nanmean(cors)),
            "mean_corr_ci": (0.75, 0.85),
            "pct_above_thr1": 95.0,
            "pct1_ci": (90.0, 100.0),
            "pval": pvals[pair_name],
            "k_windows": 3,
            "n_resamples_eff": 60,
            "block_size_eff": 5,
            "ci_level": 0.95,
        }

    monkeypatch.setattr(pipeline, "bootstrap_pair_stats", fake_bootstrap)

    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(out_path)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 50,
            "pair_corr_threshold": 0.3,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "pair_pct_threshold1": 0.0,
            "pair_pct_threshold2": 0.0,
            "enable_bootstrap": True,
            "enable_hypothesis_test": True,
            "enable_fdr": False,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.7, "block_size": 5},
            "fdr_alpha": 0.05,
            "n_jobs": 1,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    selected = pipeline.main(cfg_path=None, quick=False, overrides=cfg)
    meta = json.loads(out_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    run_pairs = Path(meta["outputs"]["run_scoped_pairs_path"])
    boot = pd.read_parquet(run_pairs.parent / "bootstrap_results.parquet")

    assert selected["pair"].tolist() == ["A-B"]
    assert meta["data_analysis"]["significance_mode"] == "pvalue"
    assert boot["significance_reject"].tolist() == [True, False]
    assert boot["fdr_reject"].tolist() == [True, False]
