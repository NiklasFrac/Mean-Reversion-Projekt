import numpy as np

import analysis.data_analysis as da


def test_bootstrap_handles_blocksize_and_nans():
    cors = np.array([0.6, np.nan, 0.7, 0.8, np.nan])
    out1 = da.bootstrap_pair_stats(
        cors, thr1=0.7, null_mean=0.5, n_resamples=60, rng_seed=123, block_size=1
    )
    out2 = da.bootstrap_pair_stats(
        cors, thr1=0.7, null_mean=0.5, n_resamples=60, rng_seed=123, block_size=5
    )
    for out in (out1, out2):
        assert set(out).issuperset(
            {"mean_corr", "mean_corr_ci", "pct_above_thr1", "pct1_ci", "pval"}
        )
        assert 0.0 <= out["pval"] <= 1.0


def test_benjamini_hochberg_no_rejections():
    pvals = [0.9, 0.8, 0.7]
    dec = da.benjamini_hochberg(pvals, alpha=0.05)
    assert dec == [False, False, False]
