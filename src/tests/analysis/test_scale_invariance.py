import json
from pathlib import Path

import numpy as np
import pandas as pd

# robust import, works for both layouts
from analysis.pipeline import main


def _make_prices(seed: int, n: int = 120):
    rng = np.random.default_rng(seed)
    # geometric random walk for A
    ret_a = 0.0004 + 0.01 * rng.standard_normal(n)
    px_a = 100.0 * np.exp(np.cumsum(ret_a))
    # B = perfekt skaliert zu A (identisch bis auf Faktor)
    px_b = 2.0 * px_a
    # C = independent random walk (should not correlate strongly)
    ret_c = 0.0004 + 0.02 * rng.standard_normal(n)
    px_c = 50.0 * np.exp(np.cumsum(ret_c))
    idx = pd.bdate_range("2022-01-03", periods=n, tz=None)
    return pd.DataFrame({"A": px_a, "B": px_b, "C": px_c}, index=idx)


def _run_once(tmp_path: Path, df: pd.DataFrame, name: str):
    prices_path = tmp_path / f"{name}.pkl"
    df.to_pickle(prices_path)
    out_path = tmp_path / f"pairs_{name}.pkl"

    cfg = {
        "data": {
            "prices_path": str(prices_path),
            "pairs_path": str(out_path),
        },
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.45,  # shrinkage corr is attenuated vs. empirical corr
            "pct_thr1": 0.35,
            "pct_thr2": 0.45,
            "pair_pct_threshold": 0.9,
            "pair_mean_corr_min": 0.35,
            "bootstrap": {"n_resamples": 50, "block_size": 10},
            "fdr_alpha": 0.10,
            "n_jobs": 1,
            "rng_seed": 7,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "perf": {"blas_threads": 1, "numexpr_threads": 1},
    }

    df_selected = main(cfg_path=None, quick=False, overrides=cfg)
    pairs = set(df_selected.get("pair", pd.Series([], dtype=str)).astype(str).tolist())
    meta = json.loads(out_path.with_suffix(".meta.json").read_text())
    return pairs, meta


def test_scale_invariance(tmp_path: Path):
    # Run 1: Basis
    df1 = _make_prices(seed=2024, n=120)
    pairs1, meta1 = _run_once(tmp_path, df1, "base")

    # Run 2: Skaliere A und B separat, C bleibt
    df2 = df1.copy()
    df2["A"] = df2["A"] * 3.7
    df2["B"] = df2["B"] * 0.51
    pairs2, meta2 = _run_once(tmp_path, df2, "scaled")

    # expectation: pair (A,B) is selected in both runs (scaling does not change log returns)
    assert any(p in pairs1 for p in ("A-B", "B-A"))
    assert any(p in pairs2 for p in ("A-B", "B-A"))

    # optional: invariance (same pair set) - tolerant check
    # (with deterministic seed and identical input it should be identical.)
    assert pairs1 == pairs2

    # meta present (observability)
    for m in (meta1, meta2):
        assert "env" in m and "timings" in m and "metrics" in m
