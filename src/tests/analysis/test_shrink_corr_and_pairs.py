from analysis.numerics import compute_log_returns, compute_shrink_corr
from analysis.pairs import list_high_pairs_vectorized


def test_pairs_exist_on_synthetic(synthetic_prices_wide):
    logret = compute_log_returns(synthetic_prices_wide)
    corr = compute_shrink_corr(logret)
    pairs = list_high_pairs_vectorized(corr, threshold=0.30)
    assert len(pairs) >= 3  # wir haben 5 korrelierte Paare injiziert
