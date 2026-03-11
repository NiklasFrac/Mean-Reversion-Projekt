import pytest

# Legacy build-test: older projects had a top-level `bo.py` with `_eval_signal`.
# This codebase uses `backtest.optimize.paper_bo` instead; keep the file from
# breaking `pytest` runs but skip it to avoid false negatives.
pytest.skip(
    "legacy build-test (bo._eval_signal) – module removed in current codebase",
    allow_module_level=True,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bo import _eval_signal  # type: ignore  # noqa: E402,F401


def make_synthetic_pair(n=200, sigma_spread=0.5):
    t = pd.date_range("2020-01-01", periods=n, freq="D")
    x = np.cumsum(np.random.normal(0, 1e-3, size=n)) + 50.0
    # mean-reverting spread (AR(1) with negative phi)
    phi = -0.3
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = phi * spread[i - 1] + np.random.normal(0, sigma_spread)
    y = x + spread
    return pd.Series(y, index=t), pd.Series(x, index=t)


def test_eval_signal_returns_numeric():
    # build pairs_data compatible with bo._eval_signal usage
    pairs_data = {}
    for i in range(3):
        y, x = make_synthetic_pair()
        pairs_data[f"PAIR{i}"] = {"t1_price": y, "t2_price": x}
    # price_data not used by _eval_signal implementation here, pass empty dict
    price_data = {}
    # build a minimal cfg
    cfg = {
        "costs": {"per_trade": 1.0, "slippage": 0.0005},
        "backtest": {"initial_capital": 100000},
        "strategy": {"z_window": 30},
        "spread_zscore": {},
    }
    # call _eval_signal — choose sensible params
    score = _eval_signal(1.0, 0.3, 2.0, 10, 0, pairs_data, price_data, cfg)
    assert isinstance(score, float)
    assert not np.isnan(score)
