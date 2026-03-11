import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.atomic_io import save_results
from analysis.preprocess import load_filled_data


@pytest.mark.unit
def test_load_filled_data_from_pickle_and_csv(tmp_path: Path):
    idx = pd.date_range("2022-01-03", periods=10, tz=None)
    df = pd.DataFrame(
        {"A": np.linspace(100, 101, 10), "B": np.linspace(50, 49, 10)}, index=idx
    )

    # pickle
    pkl = tmp_path / "p.pkl"
    with pkl.open("wb") as f:
        pickle.dump(df, f)
    out_pkl = load_filled_data(pkl)
    assert out_pkl.index.tz is None
    assert list(out_pkl.columns) == ["A", "B"]

    # csv
    csv = tmp_path / "p.csv"
    df.to_csv(csv)
    out_csv = load_filled_data(csv)
    assert out_csv.index.tz is None and list(out_csv.columns) == ["A", "B"]


@pytest.mark.unit
def test_save_results_writes_all_files(tmp_path: Path):
    df_sel = pd.DataFrame([{"pair": "A-B", "mean_corr": 0.75, "pval": 0.01}])
    out_p = tmp_path / "pairs.pkl"
    meta = {
        "thresholds": {"corr": 0.7},
        "rolling": {"window": 60, "step": 10},
        "bootstrap": {"n_resamples": 60, "seed": 1, "block_size": 5},
        "inputs": [{"prices_path": "X", "sha256": None, "rows": 10, "cols": 2}],
        "metrics": {
            "n_windows": 3,
            "dropped_rows_after_returns": 0,
            "dropped_cols_after_cleaning": 0,
        },
        "env": {"python": "3.12"},
        "timings": {"load_prices": 0.01},
        "timestamp": 1.0,
    }
    save_results(df_sel, out_p, meta)

    assert out_p.exists()
    assert out_p.with_suffix(".csv").exists()
    mj = out_p.with_suffix(".meta.json")
    assert mj.exists()
    loaded = json.loads(mj.read_text(encoding="utf-8"))
    for k in (
        "thresholds",
        "rolling",
        "bootstrap",
        "inputs",
        "metrics",
        "env",
        "timings",
        "timestamp",
    ):
        assert k in loaded
