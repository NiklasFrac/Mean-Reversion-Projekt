import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.config_io import load_config
from analysis.pipeline import main


@pytest.mark.unit
def test_load_config_raises_when_not_found(monkeypatch, tmp_path: Path):
    # kein BACKTEST_CONFIG, und wir arbeiten in leerem tmp cwd
    monkeypatch.delenv("BACKTEST_ANALYSIS_CONFIG", raising=False)
    monkeypatch.delenv("BACKTEST_CONFIG", raising=False)
    monkeypatch.delenv("STRAT_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_config(None)


@pytest.mark.integration
def test_main_quick_smoke_artifacts(tmp_path: Path, monkeypatch):
    # 1) synthetic Preise schreiben
    n = 300
    rng = np.random.default_rng(123)
    e = rng.standard_normal((n, 2))
    A = e[:, 0]
    B = 0.85 * A + 0.5 * e[:, 1]
    C = rng.standard_normal(n)
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(0.001 + 0.01 * A)),
            "B": 50 * np.exp(np.cumsum(0.001 + 0.01 * B)),
            "C": 30 * np.exp(np.cumsum(0.001 + 0.01 * C)),
        },
        index=pd.date_range("2020-01-01", periods=n, tz="UTC"),
    )
    prices_path = tmp_path / "prices.pkl"
    prices.to_pickle(prices_path)

    pairs_out = tmp_path / "pairs.pkl"

    # 2) Minimal-YAML auf Disk + Env setzen
    cfg = {
        "data": {"prices_path": str(prices_path), "pairs_path": str(pairs_out)},
        "data_analysis": {
            "rolling_window": 60,
            "rolling_step": 10,
            "pair_corr_threshold": 0.3,
            "pct_thr1": 0.3,
            "pct_thr2": 0.4,
            "pair_pct_threshold": 0.3,
            "pair_mean_corr_min": 0.2,
            "bootstrap": {"n_resamples": 60, "null_mean_corr": 0.2, "block_size": 5},
            "fdr_alpha": 0.2,
            "n_jobs": 1,
            "rng_seed": 7,
            "returns_cleaning": {
                "min_positive_frac": 0.99,
                "max_nan_frac_cols": 0.02,
                "drop_policy_rows": "any",
            },
        },
        "monitoring": {"prometheus": {"enabled": False, "keep_alive_sec": 0}},
        "perf": {"blas_threads": 1, "numexpr_threads": 1},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.delenv("BACKTEST_ANALYSIS_CONFIG", raising=False)
    monkeypatch.setenv("BACKTEST_CONFIG", str(cfg_path))

    # 3) Run (quick) + Artefakte prüfen
    df_sel = main(cfg_path=None, quick=True, overrides=None)
    assert df_sel is not None
    assert pairs_out.exists()
    meta_path = pairs_out.with_suffix(".meta.json")
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Minimal-Keys, unabhängig davon ob Selektion leer wäre
    assert "stats" in meta and "timings" in meta and "timestamp" in meta
