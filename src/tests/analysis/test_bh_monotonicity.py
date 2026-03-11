from analysis.bootstrap_fdr import benjamini_hochberg


def test_bh_alpha_monotonicity():
    p = [0.001, 0.01, 0.02, 0.04, 0.2, 0.5, 0.9]
    d1 = benjamini_hochberg(p, alpha=0.05)
    d2 = benjamini_hochberg(p, alpha=0.10)
    assert sum(d2) >= sum(d1)
