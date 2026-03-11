from backtest.strat.baseline import BaselineZScoreStrategy

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
    "costs": {"per_trade": 1.0, "slippage": 0.0005},
    "strategy": {"name": "baseline"},
    "spread_zscore": {"z_window": 30},
    "pair_prefilter": {},
}
s = BaselineZScoreStrategy(cfg, borrow_ctx=None)
print("Strategy inst: ", s)
