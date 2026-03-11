import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline import main


def test_main_cutoff_and_no_pairs_writes_empty(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=120, tz="UTC")
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 120),
            "B": np.linspace(200, 210, 120),
            "C": np.linspace(300, 315, 120),
        },
        index=idx,
    )
    pkl = tmp_path / "filled.pkl"
    prices.to_pickle(pkl)
    out = tmp_path / "pairs.pkl"

    cfg = {
        "data": {"prices_path": str(pkl), "pairs_path": str(out)},
        "data_analysis": {
            "train_cutoff_utc": "2020-04-15T00:00:00Z",  # legacy fallback hits cutoff branch
            "pair_corr_threshold": 0.999,
            "pct_thr1": 0.99,
            "pct_thr2": 0.99,
            "pair_pct_threshold": 0.99,
            "pair_mean_corr_min": 0.99,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.99, "block_size": 10},
            "rolling_window": 60,
            "rolling_step": 10,
            "rng_seed": 1,
            "returns_cleaning": {
                "min_positive_frac": 0.95,
                "max_nan_frac_cols": 0.05,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False}},
    }

    df = main(cfg_path=None, quick=False, overrides=cfg)
    assert df.empty
    assert out.exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["stats"]["n_selected"] == 0
