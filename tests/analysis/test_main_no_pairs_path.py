import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis.data_analysis as da


def test_main_writes_empty_and_meta_when_no_pairs(tmp_path: Path):
    n = 120
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n, tz="UTC")
    ret = rng.standard_normal((n, 3)) * 0.01
    px = 100 * np.exp(ret.cumsum(axis=0))
    df = pd.DataFrame(px, index=idx, columns=list("ABC"))
    pkl = tmp_path / "filled.pkl"
    df.to_pickle(pkl)
    out = tmp_path / "pairs.pkl"

    cfg = {
        "data": {"prices_path": str(pkl), "pairs_path": str(out)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.99,
            "pct_thr1": 0.95,
            "pct_thr2": 0.99,
            "pair_pct_threshold": 0.95,
            "pair_mean_corr_min": 0.95,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.95, "block_size": 10},
            "fdr_alpha": 0.05,
            "n_jobs": 1,
            "rng_seed": 123,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False, "keep_alive_sec": 0}},
    }

    res = da.main(cfg_path=None, quick=False, overrides=cfg)
    assert res.empty

    meta_path = out.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["stats"]["n_selected"] == 0
    assert "timings" in meta
