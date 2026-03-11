from analysis.numerics import compute_log_returns, compute_shrink_corr
from analysis.pairs import list_high_pairs_vectorized
from analysis.rolling import rolling_pair_metrics_fast


def test_rolling_fast_metrics(synthetic_prices_wide):
    logret = compute_log_returns(synthetic_prices_wide)
    corr = compute_shrink_corr(logret)
    pairs = [
        (a, b) for a, b, _ in list_high_pairs_vectorized(corr, threshold=0.30)[:20]
    ]
    df_summary, window_cors = rolling_pair_metrics_fast(
        logret, pairs, window=252, step=21, thr1=0.25, thr2=0.30
    )
    assert not df_summary.empty
    assert all(len(v) > 0 for v in window_cors.values())
