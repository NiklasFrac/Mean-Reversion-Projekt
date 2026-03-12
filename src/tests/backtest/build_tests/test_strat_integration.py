import numpy as np
import pandas as pd

from backtest.strat.baseline import BaselineZScoreStrategy


def make_mean_reverting_pair(n=250, phi=-0.4, sigma=0.5):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    x = np.cumsum(np.random.normal(0, 0.001, size=n)) + 50.0
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + np.random.normal(0, sigma)
    y = x + spread
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


def test_baseline_strategy_runs_and_outputs_expected_structure():
    # build small universe of 3 synthetic pairs
    pairs = {}
    for i in range(3):
        y, x = make_mean_reverting_pair()
        pairs[f"PAIR{i}"] = {"t1_price": y, "t2_price": x}
    cfg = {
        "backtest": {
            "initial_capital": 100000,
            "risk_per_trade": 0.01,
            "splits": {
                "train": {"start": "2020-01-01", "end": "2020-06-30"},
                "test": {"start": "2020-07-01", "end": "2020-09-30"},
            },
        },
        "signal": {"entry_z": 1.0, "exit_z": 0.2, "stop_z": 3.0, "max_hold_days": 10},
        "costs": {"per_trade": 0.0, "slippage": 0.0},
        "strategy": {"name": "baseline"},
        "spread_zscore": {"z_window": 30},
        "pair_prefilter": {},
    }
    strat = BaselineZScoreStrategy(cfg, borrow_ctx=None)
    out = strat(pairs)
    assert isinstance(out, dict)
    # if intents are created, the strategy should return the intent/state bundle
    for pair, meta in out.items():
        intents = meta.get("intents")
        state = meta.get("state")
        assert intents is None or hasattr(intents, "columns")
        assert state is None or isinstance(state, dict)
