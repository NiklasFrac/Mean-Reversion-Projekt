from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from backtest.borrow.accrual import compute_borrow_cost_for_trade_row
from backtest.borrow.context import build_borrow_context
from backtest.config.cfg import make_config_from_yaml
from backtest.simulators.engine import _apply_risk_gating
from backtest.simulators.lob import annotate_with_lob

CFG_PATH = Path("runs/configs/config_backtest.yaml")


def _load_cfg() -> dict:
    if not CFG_PATH.exists():
        pytest.skip(f"Missing config: {CFG_PATH}")
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        pytest.skip("Config is not a dict")
    return cfg


def _make_lob_cfg(*, levels: int, size_per_level: int) -> object:
    yaml_cfg = {
        "seed": 42,
        "backtest": {
            "splits": {
                "train": {"start": "2020-01-01", "end": "2020-01-10"},
                "test": {"start": "2020-01-11", "end": "2020-01-20"},
            }
        },
        "execution": {
            "mode": "lob",
            "lob": {
                "tick": 0.01,
                "levels": int(levels),
                "size_per_level": int(size_per_level),
                "min_spread_ticks": 1,
                "steps_per_day": 1,
                "lam": 1.0,
                "max_add": 1,
                "bias_top": 0.7,
                "cancel_prob": 0.1,
                "max_cancel": 1,
                "liq_model": {"enabled": False},
                "fill_model": {"enabled": False},
                "post_costs": {
                    "per_trade": 0.0,
                    "maker_bps": -0.1,
                    "taker_bps": 0.3,
                    "min_fee": 0.0,
                },
            },
        },
    }
    return make_config_from_yaml(yaml_cfg)


def _make_trade_df(*, size: int) -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"])
    return pd.DataFrame(
        {
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "entry_date": [dates[0]],
            "exit_date": [dates[1]],
            "signal": [1],
            "size": [int(size)],
            "beta_entry": [1.0],
            "entry_price_y": [100.0],
            "entry_price_x": [100.0],
            "exit_price_y": [100.5],
            "exit_price_x": [99.5],
        }
    )


def _make_price_map() -> dict[str, pd.Series]:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    s = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
    return {"AAA": s, "BBB": s.copy()}


def test_lob_execution_rejects_only_when_unfilled() -> None:
    price_map = _make_price_map()

    cfg_small = _make_lob_cfg(levels=1, size_per_level=10)
    trades_small = _make_trade_df(size=5)
    out_small = annotate_with_lob(trades_small, price_map, cfg_small)
    assert bool(out_small["exec_rejected"].iloc[0]) is False
    assert float(out_small["slippage_cost"].iloc[0]) <= 0.0
    assert float(out_small["impact_cost"].iloc[0]) <= 0.0

    cfg_large = _make_lob_cfg(levels=1, size_per_level=10)
    trades_large = _make_trade_df(size=100)
    out_large = annotate_with_lob(trades_large, price_map, cfg_large)
    assert bool(out_large["exec_rejected"].iloc[0]) is False
    assert str(out_large["exec_entry_status"].iloc[0]) == "blocked"
    assert pd.isna(out_large["entry_date"].iloc[0])


def test_borrow_cost_negative_for_short() -> None:
    cfg = {
        "borrow": {
            "enabled": True,
            "accrual_mode": "entry_notional",
            "day_count": "calendar_days",
            "include_exit_day": True,
            "min_days": 1,
            "day_basis": 360,
            "default_rate_annual": 0.10,
        }
    }
    borrow_ctx = build_borrow_context(cfg)
    assert borrow_ctx is not None

    calendar = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    row = pd.Series(
        {
            "entry_date": "2020-01-01",
            "exit_date": "2020-01-06",
            "y_symbol": "AAA",
            "x_symbol": "BBB",
            "y_units": -10,
            "x_units": 10,
            "entry_price_y": 100.0,
            "entry_price_x": 100.0,
            "notional_y": -1000.0,
            "notional_x": 1000.0,
            "gross_notional": 2000.0,
        }
    )
    cost = compute_borrow_cost_for_trade_row(
        row, calendar=calendar, price_data={}, borrow_ctx=borrow_ctx
    )
    assert float(cost) < 0.0

    cfg_off = {"borrow": {"enabled": False}}
    borrow_ctx_off = build_borrow_context(cfg_off)
    cost_off = compute_borrow_cost_for_trade_row(
        row, calendar=calendar, price_data={}, borrow_ctx=borrow_ctx_off
    )
    assert float(cost_off) == 0.0


def test_risk_gating_allows_small_trades() -> None:
    cfg = _load_cfg()
    risk_cfg = cfg.get("risk", {}) if isinstance(cfg.get("risk"), dict) else {}

    df = pd.DataFrame(
        {
            "pair": ["AAA-BBB", "CCC-DDD"],
            "y_symbol": ["AAA", "CCC"],
            "x_symbol": ["BBB", "DDD"],
            "entry_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "exit_date": pd.to_datetime(["2020-01-05", "2020-01-06"]),
            "notional_y": [1000.0, 1200.0],
            "notional_x": [-1000.0, -1200.0],
            "gross_pnl": [10.0, -5.0],
            "net_pnl": [9.0, -6.0],
        }
    )
    e0 = pd.Timestamp("2020-01-02")
    e1 = pd.Timestamp("2020-01-10")
    out_df, rep = _apply_risk_gating(
        df, e0=e0, e1=e1, initial_capital=100000.0, risk_cfg=risk_cfg, price_data={}
    )
    assert int(rep.get("blocked", 0)) == 0
    assert int(rep.get("accepted", 0)) == len(df)
    assert len(out_df) == len(df)
