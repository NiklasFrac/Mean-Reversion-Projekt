from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.simulators import engine
from backtest.simulators.engine_trades import (
    _clip_trades_to_eval_window,
    _normalize_trades,
)


def test_backtest_portfolio_exec_rejects_and_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    price_data = {
        "AAA": pd.Series(np.linspace(100.0, 105.0, len(idx)), index=idx),
        "BBB": pd.Series(np.linspace(50.0, 55.0, len(idx)), index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[2], idx[3]],
            "exit_date": [idx[-1] + pd.Timedelta(days=2), idx[3]],
            "pair": ["AAA-BBB", "AAA-BBB"],
            "y_symbol": ["AAA", "AAA"],
            "x_symbol": ["BBB", "BBB"],
            "notional_y": [1000.0, 1000.0],
            "notional_x": [-1000.0, -1000.0],
            "entry_price_y": [
                float(price_data["AAA"].iloc[2]),
                float(price_data["AAA"].iloc[3]),
            ],
            "entry_price_x": [
                float(price_data["BBB"].iloc[2]),
                float(price_data["BBB"].iloc[3]),
            ],
            "signal": [1, -1],
            "gross_pnl": [10.0, -5.0],
        }
    )
    portfolio = {"AAA-BBB": {"trades": trades}}
    cfg = {
        "backtest": {
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[1].date())},
                "test": {"start": str(idx[2].date()), "end": str(idx[4].date())},
            },
        },
        "execution": {"mode": "lob"},
        "risk": {"enabled": True, "max_trade_pct": 10.0},
        "borrow": {"enabled": False},
    }

    def fake_annotate(trades_df, *_args, **_kwargs):
        out = trades_df.copy()
        out["exec_rejected"] = [False, True]
        out["exec_reject_reason"] = ["", "no_liquidity"]
        return out

    monkeypatch.setattr(engine, "annotate_with_lob", fake_annotate)

    stats, trades_out = engine.backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=price_data,
        yaml_cfg=cfg,
        market_data_panel=None,
        adv_map=None,
    )
    assert stats.attrs.get("exec_rejected_count") == 1
    assert "hard_exit" in trades_out.columns
    assert trades_out["hard_exit"].any()
    assert "net_pnl" in trades_out.columns


def test_normalize_trades_alias_and_exit_infer() -> None:
    df = pd.DataFrame(
        {
            "start_dt": ["2024-01-02", "2024-01-03"],
            "close": ["2024-01-05", "2024-01-06"],
            "gross_pnl": [1.0, 2.0],
        }
    )
    norm = _normalize_trades("PAIR-1", df)
    assert norm is not None
    assert "entry_date" in norm.columns
    assert "exit_date" in norm.columns
    assert norm["pair"].iloc[0] == "PAIR-1"


def test_clip_trades_to_eval_window_recomputes_gross_pnl() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[-1] + pd.Timedelta(days=2)],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
        }
    )
    price_data = {
        "AAA": pd.Series([100.0, 105.0, 110.0, 115.0, 120.0], index=idx),
        "BBB": pd.Series([50.0, 52.0, 55.0, 57.0, 60.0], index=idx),
    }
    out, rep = _clip_trades_to_eval_window(
        df, e0=idx[0], e1=idx[-1], price_data=price_data
    )
    assert rep["hard_exits"] == 1
    assert float(out.loc[0, "gross_pnl"]) == 100.0
    assert float(out.loc[0, "exit_price_y"]) == 120.0
    assert float(out.loc[0, "exit_price_x"]) == 60.0


def test_perf_run_budget_ignore() -> None:
    out = engine._perf_run("noop", lambda: 7)
    assert out == 7


def test_daily_pnl_cost_timing_entry_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2024-01-02", periods=3)
    price_data = {
        "AAA": pd.Series([100.0, 100.0, 100.0], index=idx),
        "BBB": pd.Series([50.0, 50.0, 50.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[0]],
            "exit_date": [idx[2]],
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "y_units": [1],
            "x_units": [-1],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [100.0],
            "exit_price_x": [50.0],
            "gross_pnl": [0.0],
            "fees": [-10.0],
            "slippage_cost": [-4.0],
            "impact_cost": [-2.0],
        }
    )
    portfolio = {"AAA-BBB": {"trades": trades}}
    cfg = {
        "seed": 1,
        "backtest": {
            "initial_capital": 1000.0,
            "splits": {
                "train": {
                    "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                    "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                },
                "test": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            },
        },
        "execution": {"mode": "lob"},
        "risk": {"enabled": False},
        "borrow": {"enabled": False},
    }

    monkeypatch.setattr(engine, "annotate_with_lob", lambda df, *_args, **_kwargs: df)

    stats, _trades_out = engine.backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=price_data,
        yaml_cfg=cfg,
        market_data_panel=None,
        adv_map=None,
    )
    equity = stats["equity"]
    assert float(equity.iloc[0]) == pytest.approx(992.0)
    assert float(equity.iloc[1]) == pytest.approx(992.0)
    assert float(equity.iloc[2]) == pytest.approx(984.0)


def test_backtest_portfolio_light_mode_dispatches_only_light(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=4)
    price_data = {
        "AAA": pd.Series([100.0, 101.0, 102.0, 103.0], index=idx),
        "BBB": pd.Series([50.0, 51.0, 52.0, 53.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[0]],
            "exit_date": [idx[2]],
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [102.0],
            "exit_price_x": [52.0],
            "gross_pnl": [0.0],
        }
    )
    portfolio = {"AAA-BBB": {"trades": trades}}
    cfg = {
        "backtest": {
            "splits": {
                "train": {
                    "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                    "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                },
                "test": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            },
        },
        "execution": {"mode": "light", "light": {"fees": {"per_trade": 0.0}}},
        "borrow": {"enabled": False},
    }

    def _lob_boom(*_args, **_kwargs):
        raise AssertionError("LOB path must not run in light mode")

    def _light_annotate(trades_df, *_args, **_kwargs):
        out = trades_df.copy()
        out["fees"] = -1.0
        out["exec_rejected"] = False
        out["exec_mode_used"] = "light"
        return out

    monkeypatch.setattr(engine, "annotate_with_lob", _lob_boom)
    monkeypatch.setattr(engine, "annotate_with_light", _light_annotate)
    monkeypatch.setattr(engine, "_LIGHT_OK", True)

    stats, trades_out = engine.backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=price_data,
        yaml_cfg=cfg,
        market_data_panel=None,
        adv_map=None,
    )

    assert stats.attrs.get("exec_mode") == "light"
    assert stats.attrs.get("exec_light_enabled") is True
    assert stats.attrs.get("exec_lob_enabled") is False
    assert str(trades_out["exec_mode_used"].iloc[0]) == "light"


def test_backtest_portfolio_light_mode_disabled_skips_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=4)
    price_data = {
        "AAA": pd.Series([100.0, 101.0, 102.0, 103.0], index=idx),
        "BBB": pd.Series([50.0, 51.0, 52.0, 53.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[0]],
            "exit_date": [idx[2]],
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [102.0],
            "exit_price_x": [52.0],
            "gross_pnl": [0.0],
        }
    )
    portfolio = {"AAA-BBB": {"trades": trades}}
    cfg = {
        "backtest": {
            "splits": {
                "train": {
                    "start": str((idx[0] - pd.Timedelta(days=1)).date()),
                    "end": str((idx[0] - pd.Timedelta(days=1)).date()),
                },
                "test": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            },
        },
        "execution": {"mode": "light", "light": {"enabled": False}},
        "borrow": {"enabled": False},
    }

    def _light_boom(*_args, **_kwargs):
        raise AssertionError("disabled light overlay must not run")

    monkeypatch.setattr(engine, "annotate_with_light", _light_boom)
    monkeypatch.setattr(engine, "_LIGHT_OK", True)

    stats, trades_out = engine.backtest_portfolio_with_yaml_cfg(
        portfolio=portfolio,
        price_data=price_data,
        yaml_cfg=cfg,
        market_data_panel=None,
        adv_map=None,
    )

    assert stats.attrs.get("exec_light_enabled") is False
    assert "exec_mode_used" not in trades_out.columns
