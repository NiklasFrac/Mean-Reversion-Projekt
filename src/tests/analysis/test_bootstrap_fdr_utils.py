# src/tests/analysis/test_bootstrap_fdr_utils.py
import numpy as np

from analysis.bootstrap_fdr import benjamini_hochberg, bootstrap_pair_stats


def test_bootstrap_paths_iid_and_block():
    cors = np.array([0.2, 0.5, 0.8, 0.6, 0.7])
    s1 = bootstrap_pair_stats(
        cors, thr1=0.6, null_mean=0.4, n_resamples=200, rng_seed=1, block_size=None
    )
    s2 = bootstrap_pair_stats(
        cors, thr1=0.6, null_mean=0.4, n_resamples=200, rng_seed=1, block_size=3
    )
    for s in (s1, s2):
        assert 0.0 <= s["pval"] <= 1.0
        assert s["mean_corr_ci"][0] <= s["mean_corr_ci"][1]


def test_bootstrap_empty_returns_empty():
    s = bootstrap_pair_stats(
        np.array([np.nan, np.nan]), thr1=0.6, null_mean=0.5, n_resamples=50, rng_seed=0
    )
    assert s == {}


def test_benjamini_hochberg_cases():
    assert benjamini_hochberg([0.9, 0.8, 0.7], alpha=0.05) == [False, False, False]
    dec = benjamini_hochberg([0.001, 0.01, 0.03, 0.2], alpha=0.1)
    assert any(dec)
