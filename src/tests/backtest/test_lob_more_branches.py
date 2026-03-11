from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.simulators import lob


def test_trade_units_zero_signal_or_size() -> None:
    row = pd.Series({"signal": 0, "size": 10})
    assert lob._trade_units_from_row(row) == (0, 0)

    row2 = pd.Series({"signal": 1, "size": 0})
    assert lob._trade_units_from_row(row2) == (0, 0)


def test_mid_at_with_missing_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=1, freq="D")
    series_nan = pd.Series([np.nan], index=idx)
    assert lob._mid_at(series_nan, idx[0]) is None
    assert lob._mid_at(None, idx[0]) is None


def test_exec_one_leg_zero_units_and_missing_mids() -> None:
    params = lob._mk_book_params(SimpleNamespace(exec_lob={}, raw_yaml={}))
    entry_ts = pd.Timestamp("2024-01-02")
    exit_ts = pd.Timestamp("2024-01-03")

    out_zero = lob._exec_one_leg(
        symbol="AAA",
        units=0,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_mid=None,
        exit_mid=None,
        price_series=None,
        book_params=params,
        liq_model=None,
        shard_id=0,
    )
    assert out_zero["entry_vwap"] is None
    assert out_zero["exit_vwap"] is None
    assert out_zero["entry_requested_size"] == 0
    assert out_zero["exit_requested_size"] == 0

    out_missing = lob._exec_one_leg(
        symbol="AAA",
        units=5,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_mid=None,
        exit_mid=None,
        price_series=None,
        book_params=params,
        liq_model=None,
        shard_id=1,
    )
    assert out_missing["entry_vwap"] is None
    assert out_missing["exit_vwap"] is None
    assert out_missing["entry_requested_size"] == 5
    assert out_missing["entry_unfilled_size"] == 5
    assert out_missing["exit_requested_size"] == 5
    assert out_missing["exit_unfilled_size"] == 5


def test_exec_one_leg_maker_flow() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    series = pd.Series([100.0, 101.0], index=idx)
    params = lob._mk_book_params(SimpleNamespace(exec_lob={}, raw_yaml={}))
    rep = lob._exec_one_leg(
        symbol="AAA",
        units=10,
        entry_ts=idx[0],
        exit_ts=idx[1],
        entry_mid=100.0,
        exit_mid=101.0,
        price_series=series,
        book_params=params,
        liq_model=None,
        shard_id=0,
        order_flow_entry={"mode": "maker", "maker_price": "mid"},
        order_flow_exit={"mode": "taker"},
    )
    assert rep["entry_liquidity"] == "maker"
    assert rep["exit_liquidity"] == "taker"


def test_exec_one_leg_maker_not_touched_blocks() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    series = pd.Series([100.0, 101.0], index=idx)
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [100.0, 101.0],
            ("AAA", "high"): [99.0, 100.0],
            ("AAA", "low"): [98.0, 99.0],
            ("AAA", "volume"): [1000.0, 1000.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    liq_model = lob.LiquidityModel(
        panel, cfg=lob.LiquidityModelCfg(enabled=True), adv_map_usd=None
    )
    params = lob._mk_book_params(SimpleNamespace(exec_lob={}, raw_yaml={}))

    rep = lob._exec_one_leg(
        symbol="AAA",
        units=10,
        entry_ts=idx[0],
        exit_ts=idx[1],
        entry_mid=100.0,
        exit_mid=101.0,
        price_series=series,
        book_params=params,
        liq_model=liq_model,
        shard_id=0,
        order_flow_entry={
            "mode": "maker",
            "maker_price": "mid",
            "fallback_to_taker": False,
        },
        order_flow_exit={"mode": "taker"},
    )
    assert rep["entry_liquidity"] == "blocked"
    assert rep["entry_filled_size"] == 0


def test_annotate_with_lob_fill_model_scaling(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="America/New_York")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 11.0], index=idx),
        "BBB": pd.Series([20.0, 20.5, 21.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
        }
    )
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [10.0, 10.5, 11.0],
            ("AAA", "volume"): [1000, 1000, 1000],
            ("BBB", "close"): [20.0, 20.5, 21.0],
            ("BBB", "volume"): [1000, 1000, 1000],
        },
        index=idx,
    )
    cfg_obj = SimpleNamespace(exec_lob={"fill_model": {"enabled": True}}, raw_yaml={})

    def _fake_fill_fraction(**_kwargs):
        return 0.5, {"expected": 0.5}

    monkeypatch.setattr(lob, "sample_package_fill_fraction", _fake_fill_fraction)

    out = lob.annotate_with_lob(trades, price_data, cfg_obj, market_data_panel=panel)
    assert "exec_fill_frac" in out.columns
    assert out["exec_fill_frac"].iloc[0] == pytest.approx(0.5)
    assert "exec_rejected" in out.columns


def test_annotate_with_lob_fill_model_scaling_preserves_explicit_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="America/New_York")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 11.0], index=idx),
        "BBB": pd.Series([20.0, 20.5, 21.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "y_units": [10],
            "x_units": [-10],
        }
    )
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [10.0, 10.5, 11.0],
            ("AAA", "volume"): [1000, 1000, 1000],
            ("BBB", "close"): [20.0, 20.5, 21.0],
            ("BBB", "volume"): [1000, 1000, 1000],
        },
        index=idx,
    )
    cfg_obj = SimpleNamespace(
        exec_lob={"fill_model": {"enabled": True}, "liq_model": {"enabled": True}},
        raw_yaml={},
    )

    def _fake_fill_fraction(**_kwargs):
        return 0.5, {"expected": 0.5}

    monkeypatch.setattr(lob, "sample_package_fill_fraction", _fake_fill_fraction)
    monkeypatch.setattr(
        lob,
        "_seeded_rng",
        lambda *_args, **_kwargs: type("R", (), {"random": lambda self: 0.0})(),
    )

    out = lob.annotate_with_lob(trades, price_data, cfg_obj, market_data_panel=panel)
    assert int(out.loc[0, "y_units"]) == 10
    assert int(out.loc[0, "x_units"]) == -10
    assert str(out.loc[0, "exec_entry_status"]) in {"filled", "delayed"}
    assert bool(out.loc[0, "exec_rejected"]) is False


def test_lob_volume_zero_blocks_entry() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 10.6], index=idx),
        "BBB": pd.Series([20.0, 20.5, 20.6], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
        }
    )
    panel = pd.DataFrame(
        {
            ("AAA", "close"): [10.0, 10.5, 10.6],
            ("AAA", "volume"): [1000.0, 0.0, 1000.0],
            ("BBB", "close"): [20.0, 20.5, 20.6],
            ("BBB", "volume"): [1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    cfg_obj = SimpleNamespace(exec_lob={"fill_model": {"enabled": False}}, raw_yaml={})

    out = lob.annotate_with_lob(trades, price_data, cfg_obj, market_data_panel=panel)
    assert bool(out.loc[0, "exec_rejected"]) is False
    assert str(out.loc[0, "exec_entry_status"]) == "delayed"
    assert int(out.loc[0, "y_units"]) == 10
    assert int(out.loc[0, "x_units"]) == -10


def test_lob_rejects_when_leg_prices_missing() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 10.6], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
            "entry_price_y": [10.5],
            "exit_price_y": [10.6],
        }
    )
    cfg_obj = SimpleNamespace(exec_lob={"fill_model": {"enabled": False}}, raw_yaml={})

    out = lob.annotate_with_lob(trades, price_data, cfg_obj, market_data_panel=None)
    assert bool(out.loc[0, "exec_rejected"]) is True
    assert str(out.loc[0, "exec_reject_reason"]) == "missing_price_series"


def test_lob_rejects_when_exit_unfilled() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 10.6], index=idx),
        "BBB": pd.Series([20.0, 20.5, 20.6], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
        }
    )
    cfg_obj = SimpleNamespace(
        exec_lob={
            "fill_model": {"enabled": False},
            "liq_model": {"enabled": False},
            "order_flow": {
                "entry": {"mode": "taker"},
                "exit": {
                    "mode": "maker",
                    "fallback_to_taker": False,
                    "maker_price": "mid",
                    "maker_touch_prob": 0.0,
                },
            },
        },
        raw_yaml={},
    )

    out = lob.annotate_with_lob(trades, price_data, cfg_obj, market_data_panel=None)
    assert bool(out.loc[0, "exec_rejected"]) is False
    assert str(out.loc[0, "exec_exit_status"]) == "forced"
    assert bool(out.loc[0, "exec_forced_exit"]) is True


def test_resolve_exec_lob_cfg_prefers_direct_then_execution_lob() -> None:
    cfg_direct = SimpleNamespace(
        exec_lob={"tick": 0.01},
        raw_yaml={"execution": {"lob": {"tick": 0.03}}},
    )
    assert lob._resolve_exec_lob_cfg(cfg_direct)["tick"] == 0.01

    cfg_raw = SimpleNamespace(
        exec_lob={},
        raw_yaml={"execution": {"exec_lob": {"tick": 0.02}, "lob": {"tick": 0.03}}},
    )
    assert lob._resolve_exec_lob_cfg(cfg_raw)["tick"] == 0.03


def test_annotate_with_lob_order_flow_reads_execution_lob_only() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    price_data = {
        "AAA": pd.Series([10.0, 10.5, 11.0], index=idx),
        "BBB": pd.Series([20.0, 20.5, 21.0], index=idx),
    }
    trades = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[2]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
        }
    )
    cfg_obj = SimpleNamespace(
        exec_lob={},
        raw_yaml={
            "execution": {
                "lob": {
                    "fill_model": {"enabled": False},
                    "liq_model": {"enabled": False},
                    "order_flow": {
                        "mode": "maker",
                        "maker_price": "mid",
                        "entry": {"mode": "maker"},
                        "exit": {"mode": "taker"},
                    },
                }
            }
        },
    )

    out = lob.annotate_with_lob(
        trades, price_data, cfg_obj, market_data_panel=None, calendar=idx
    )
    assert str(out.loc[0, "liquidity_entry_y"]) == "maker"
    assert str(out.loc[0, "liquidity_exit_y"]) == "taker"
