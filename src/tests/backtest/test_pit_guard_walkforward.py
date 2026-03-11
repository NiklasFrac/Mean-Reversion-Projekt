from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.pit_guard import (
    PitGuardConfig,
    assert_no_future_dependency,
    assert_no_future_dependency_walkforward,
)


def _make_prices(n_days: int = 140) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-02", periods=n_days, tz="America/New_York")
    s = pd.Series(100 + np.linspace(0, 5, n_days), index=idx, name="AAA")
    return pd.DataFrame({"AAA": s})


def _dummy_runner(
    start_idx,
    prices,
    prices_panel,
    pairs,
    cfg,
    adv_map,
    borrow_ctx,
    availability_long,
    availability_scope,
):
    splits = cfg.get("backtest", {}).get("splits", {})
    test = splits.get("test", {})
    e0 = pd.Timestamp(test.get("start"))
    e1 = pd.Timestamp(test.get("end"))
    idx = prices.index
    tz = idx.tz
    if tz is not None:
        e0 = e0.tz_localize(tz) if e0.tz is None else e0.tz_convert(tz)
        e1 = e1.tz_localize(tz) if e1.tz is None else e1.tz_convert(tz)
    window = prices.loc[(idx >= e0) & (idx <= e1)]
    if window.empty:
        raise AssertionError("empty window in dummy runner")

    s = window.iloc[:, 0]
    returns = s.pct_change().fillna(0.0)
    equity = (100.0 + returns.cumsum()).rename("equity")
    stats = pd.DataFrame({"equity": equity}, index=window.index)
    trades = pd.DataFrame(
        {
            "pair": ["AAA-BBB"],
            "entry_date": [window.index[1]],
            "exit_date": [window.index[-1]],
            "gross_pnl": [1.0],
            "net_pnl": [1.0],
        }
    )
    info = {"n_pairs": 1, "n_trades": len(trades)}
    artifacts = {"trades_te": trades}
    return stats, info, stats, info, artifacts


def _leaky_runner(
    start_idx,
    prices,
    prices_panel,
    pairs,
    cfg,
    adv_map,
    borrow_ctx,
    availability_long,
    availability_scope,
):
    stats, info, stats2, info2, artifacts = _dummy_runner(
        start_idx,
        prices,
        prices_panel,
        pairs,
        cfg,
        adv_map,
        borrow_ctx,
        availability_long,
        availability_scope,
    )
    leak = float(prices.iloc[-1, 0])
    stats = stats.copy()
    stats["equity"] = stats["equity"] + leak
    return stats, info, stats, info, artifacts


def _base_cfg() -> dict:
    return {
        "seed": 7,
        "data": {"calendar_name": "XNYS"},
        "backtest": {
            "splits": {
                "train": {"start": "2020-01-02", "end": "2020-03-31"},
                "test": {"start": "2020-04-01", "end": "2020-05-29"},
            },
        },
    }


def test_pit_guard_single_pass():
    prices = _make_prices(120)
    cfg = _base_cfg()
    pg = PitGuardConfig(noise_sigma=0.05)
    assert_no_future_dependency(
        prices=prices,
        prices_panel=None,
        pairs={},
        cfg=cfg,
        runner=_dummy_runner,
        pg=pg,
    )


def test_pit_guard_single_detects_leak():
    prices = _make_prices(120)
    cfg = _base_cfg()
    pg = PitGuardConfig(noise_sigma=0.05)
    with pytest.raises(AssertionError):
        assert_no_future_dependency(
            prices=prices,
            prices_panel=None,
            pairs={},
            cfg=cfg,
            runner=_leaky_runner,
            pg=pg,
        )


def test_pit_guard_walkforward_pass():
    prices = _make_prices(160)
    cfg = {
        "seed": 11,
        "data": {"calendar_name": "XNYS"},
        "backtest": {
            "range": {"start": "2020-01-02", "end": "2020-08-31"},
            "walkforward": {
                "enabled": True,
                "train_mode": "rolling",
                "initial_train_months": 2,
                "test_months": 1,
                "step_months": 1,
            },
        },
    }
    pg = PitGuardConfig(noise_sigma=0.05, max_windows=2)
    res = assert_no_future_dependency_walkforward(
        prices=prices,
        prices_panel=None,
        pairs={},
        cfg=cfg,
        runner=_dummy_runner,
        pg=pg,
    )
    assert len(res) == 2
