import numpy as np
import pandas as pd
import pytest

from analysis.bootstrap_fdr import (
    _stationary_bootstrap_indices,
    benjamini_hochberg,
    bootstrap_pair_stats,
)
from analysis.rolling import rolling_pair_metrics_fast


def _make_returns_with_corr(n=300, seed=123):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    A = e[:, 0]
    B = 0.8 * A + 0.6 * e[:, 1]  # stark korreliert
    C = rng.standard_normal(n)  # unkorreliert
    return pd.DataFrame({"A": A, "B": B, "C": C})


@pytest.mark.unit
def test_rolling_pair_metrics_fast_nonempty_and_pct_order():
    logret = _make_returns_with_corr()
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    df_sum, per_win = rolling_pair_metrics_fast(
        logret, pairs, window=60, step=20, thr1=0.4, thr2=0.6
    )
    assert not df_sum.empty
    assert "A-B" in set(df_sum["pair"])
    # A-B sollte höhere mean_corr als A-C/B-C haben
    ab = float(df_sum.loc[df_sum["pair"] == "A-B", "mean_corr"].iloc[0])
    rest = float(df_sum.loc[df_sum["pair"] != "A-B", "mean_corr"].mean())
    assert ab > rest
    # window_cors liefert pro Paar Arrays
    assert "A-B" in per_win and per_win["A-B"].ndim == 1


@pytest.mark.unit
def test_stationary_bootstrap_indices_shape_and_range():
    m = 100
    idx = _stationary_bootstrap_indices(m, block_size=7, rng=np.random.default_rng(42))
    assert idx.shape == (m,)
    assert int(idx.min()) >= 0 and int(idx.max()) < m


@pytest.mark.unit
def test_bootstrap_pair_stats_keys_and_bounds():
    cors = np.array([0.3, 0.5, 0.7, 0.8, 0.6])
    out = bootstrap_pair_stats(
        cors, thr1=0.5, null_mean=0.2, n_resamples=200, rng_seed=7, block_size=3
    )
    required = {"mean_corr", "mean_corr_ci", "pct_above_thr1", "pct1_ci", "pval"}
    assert required.issubset(set(out.keys()))
    assert 0.0 <= out["pval"] <= 1.0
    lo, hi = out["mean_corr_ci"]
    assert lo <= out["mean_corr"] <= hi


@pytest.mark.unit
def test_benjamini_hochberg_basic():
    pvals = [0.001, 0.010, 0.020, 0.200]
    dec = benjamini_hochberg(pvals, alpha=0.05)
    # n=4 → Schwellen [0.0125, 0.025, 0.0375, 0.05] → erste drei True
    assert dec == [True, True, True, False]
