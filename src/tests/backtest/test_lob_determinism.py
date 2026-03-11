from types import SimpleNamespace

import pandas as pd

from backtest.simulators.lob import annotate_with_lob


def _cfg(seed: int) -> SimpleNamespace:
    lob_cfg = {
        "tick": 0.01,
        "levels": 3,
        "size_per_level": 100,
        "min_spread_ticks": 1,
        "steps_per_day": 2,
        "lam": 1.0,
        "max_add": 10,
        "bias_top": 0.7,
        "cancel_prob": 0.1,
        "max_cancel": 5,
        "fill_model": {
            "enabled": True,
            "beta_kappa_base": 50.0,
            "allow_reject": True,
            "reject_below": 0.001,
            "min_fill_if_filled": 0.05,
        },
        "order_flow": {"mode": "taker"},
    }
    return SimpleNamespace(
        exec_lob=lob_cfg,
        raw_yaml={"seed": seed, "execution": {"lob": lob_cfg}},
    )


def test_lob_determinism_with_seed() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B", tz="America/New_York")
    prices = {
        "AAA": pd.Series(100.0, index=idx),
        "BBB": pd.Series(50.0, index=idx),
    }

    trades = pd.DataFrame(
        {
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "entry_date": [idx[2]],
            "exit_date": [idx[6]],
            "signal": [1],
            "size": [10],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [101.0],
            "exit_price_x": [49.5],
            "y_units": [10],
            "x_units": [-10],
            "gross_pnl": [0.0],
        }
    )

    cfg = _cfg(seed=42)
    out1 = annotate_with_lob(trades, prices, cfg)
    out2 = annotate_with_lob(trades, prices, cfg)

    cols = [
        "exec_entry_vwap_y",
        "exec_entry_vwap_x",
        "exec_exit_vwap_y",
        "exec_exit_vwap_x",
        "exec_fill_frac",
        "lob_gross_pnl",
    ]
    pd.testing.assert_frame_equal(out1[cols], out2[cols], check_dtype=False)
