import json
from pathlib import Path

from analysis.pipeline import main


def test_end_to_end(tmp_path, tmp_prices_file, monkeypatch):
    cfg = {
        "data": {
            "prices_path": str(tmp_prices_file),
            "pairs_path": str(tmp_path / "filtered_pairs.pkl"),
        },
        "data_analysis": {
            "rolling_window": 252,
            "rolling_step": 21,
            "pair_corr_threshold": 0.30,
            "pct_thr1": 0.25,
            "pct_thr2": 0.30,
            "pair_pct_threshold": 0.60,
            "pair_mean_corr_min": 0.25,
            "bootstrap": {"n_resamples": 100, "null_mean_corr": 0.60},
            "fdr_alpha": 0.1,
            "n_jobs": 1,
            "rng_seed": 123,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.02,
                "drop_policy_rows": "any",
            },
        },
        "perf": {"blas_threads": 1, "numexpr_threads": 1},
    }
    out = main(cfg_path=None, quick=False, overrides=cfg)
    assert out is not None
    # Artefakte prüfen
    meta_path = Path(cfg["data"]["pairs_path"]).with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["analysis_schema_version"] == 1
    assert meta["metrics"]["n_windows"] >= 10
