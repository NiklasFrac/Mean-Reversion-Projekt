from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import main


def _make_synthetic_prices(n=300, seed=123):
    rng = np.random.default_rng(seed)
    # A,B stark korreliert; C,D moderat; E,F unkorreliert
    base = np.cumsum(rng.normal(0.0005, 0.01, size=n))
    ab_noise = rng.normal(0, 0.002, size=n)
    cd_noise = rng.normal(0, 0.004, size=n)
    ef1 = np.cumsum(rng.normal(0.0005, 0.015, size=n))
    ef2 = np.cumsum(rng.normal(0.0005, 0.015, size=n))
    A = 100 * np.exp(base + ab_noise)
    B = 100 * np.exp(base + ab_noise * 1.05)
    C = 80 * np.exp(base * 0.6 + cd_noise)
    D = 90 * np.exp(base * 0.6 + cd_noise * 1.1)
    E = 50 * np.exp(ef1)
    F = 60 * np.exp(ef2)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}, index=idx)
    return df


def test_end_to_end_selects_pairs(tmp_path: Path):
    prices = _make_synthetic_prices()
    pkl = tmp_path / "filled.pkl"
    prices.to_pickle(pkl)

    out = tmp_path / "pairs.pkl"
    cfg = {
        "data": {"prices_path": str(pkl), "pairs_path": str(out)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.7,
            "pct_thr1": 0.6,
            "pct_thr2": 0.7,
            "pair_pct_threshold": 0.6,
            "pair_mean_corr_min": 0.6,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.6, "block_size": 10},
            "fdr_alpha": 0.1,
            "n_jobs": 1,
            "rng_seed": 123,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
        "perf": {"blas_threads": 1, "numexpr_threads": 1},
    }

    df_sel = main(cfg_path=None, quick=False, overrides=cfg)
    assert not df_sel.empty
    # Erwartung: A-B irgendwo dabei
    assert any(x in df_sel["pair"].tolist() for x in ("A-B", "B-A"))

    # Artefakte vorhanden
    assert (
        out.exists()
        and out.with_suffix(".csv").exists()
        and out.with_suffix(".meta.json").exists()
    )
