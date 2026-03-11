import numpy as np
import pytest

from analysis.bootstrap_fdr import (
    _stationary_bootstrap_indices,
    benjamini_hochberg,
    bootstrap_pair_stats,
)
from analysis.utils import parse_pair


def test_parse_pair_variants():
    assert parse_pair("A-B") == ("A", "B")
    assert parse_pair("A/B") == ("A", "B")
    assert parse_pair("A, B") == ("A", "B")
    assert parse_pair("A B") == ("A", "B")
    assert parse_pair(["A", "B"]) == ("A", "B")
    with pytest.raises(ValueError):
        parse_pair("InvalidSingleToken")


def test_benjamini_hochberg_scenarios():
    assert benjamini_hochberg([0.2, 0.3], alpha=0.05) == [False, False]
    out = benjamini_hochberg([0.001, 0.04, 0.2], alpha=0.1)
    assert out.count(True) >= 1
    assert benjamini_hochberg([], alpha=0.5) == []


def test_bootstrap_pair_stats_block_and_iid():
    cors = np.array([0.6, 0.7, 0.8, 0.9])
    iid = bootstrap_pair_stats(
        cors, thr1=0.7, null_mean=0.5, n_resamples=50, rng_seed=1, block_size=None
    )
    blk = bootstrap_pair_stats(
        cors, thr1=0.7, null_mean=0.5, n_resamples=50, rng_seed=1, block_size=3
    )
    assert 0 <= iid["pval"] <= 1 and 0 <= blk["pval"] <= 1
    assert iid["pct_above_thr1"] >= 0 and blk["pct_above_thr1"] >= 0


def test_stationary_bootstrap_indices_length():
    idx = _stationary_bootstrap_indices(
        m=10, block_size=4, rng=np.random.default_rng(0)
    )
    assert len(idx) == 10 and idx.min() >= 0 and idx.max() < 10
