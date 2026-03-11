import pandas as pd
import pytest

from backtest.config.cfg import make_config_from_yaml
from backtest.simulators import engine
from backtest.simulators.engine_pipeline import _resolve_eval_window
from backtest.simulators.engine_trades import (
    _clip_trades_to_eval_window,
    _normalize_trades,
)
from backtest.simulators.engine_tz import _coerce_like_index, _ensure_calendar_tz


def _make_short_heuristic_trade(**overrides) -> pd.DataFrame:
    row = {
        "pair": "AAA-BBB",
        "entry_date": pd.Timestamp("2024-01-02"),
        "exit_date": pd.Timestamp("2024-01-05"),
        "y_symbol": "AAA",
        "x_symbol": "BBB",
        "notional_y": -1000.0,
        "notional_x": 1000.0,
        "y_units": -100.0,
        "x_units": 100.0,
        "entry_price_y": 10.0,
        "entry_price_x": 10.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engine_time_helpers() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s = pd.Series(idx)
    s_ny = engine._to_ex_tz_series(s, "America/New_York", naive_is_utc=False)
    assert s_ny.dt.tz is not None

    ts = engine._to_ex_tz_timestamp(
        pd.Timestamp("2024-01-02"), "America/New_York", naive_is_utc=False
    )
    assert ts.tz is not None

    cal = _ensure_calendar_tz(pd.DatetimeIndex(idx), "America/New_York")
    assert cal.tz is not None

    coerced = _coerce_like_index(pd.Series(idx), cal)
    assert isinstance(coerced, pd.Series)

    price = pd.Series([1.0, 2.0, 3.0], index=idx)
    asof = engine._asof_price_for_ts(price, pd.Timestamp("2024-01-02"))
    assert asof == 2.0


def test_clip_trades_to_eval_window_hard_exit() -> None:
    idx = pd.bdate_range("2024-01-02", periods=6)
    df = pd.DataFrame(
        {
            "entry_date": [idx[1]],
            "exit_date": [idx[-1]],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "signal": [1],
            "size": [10],
            "entry_price_y": [100.0],
            "entry_price_x": [100.0],
        }
    )
    price_data = {
        "AAA": pd.Series(range(6), index=idx),
        "BBB": pd.Series(range(6), index=idx),
    }
    out, rep = _clip_trades_to_eval_window(
        df,
        e0=idx[0],
        e1=idx[3],
        price_data=price_data,
    )
    assert rep["hard_exits"] == 1
    assert pd.Timestamp(out.loc[0, "exit_date"]) == idx[3]
    assert bool(out.loc[0, "hard_exit"]) is True


def test_clip_trades_to_eval_window_drops_entries_outside_eval_and_handles_missing_columns() -> (
    None
):
    idx = pd.bdate_range("2024-01-02", periods=4)
    df = pd.DataFrame(
        {
            "entry_date": [idx[0]],
            "exit_date": [idx[-1]],
        }
    )
    out, rep = _clip_trades_to_eval_window(
        df,
        e0=idx[1],
        e1=idx[2],
        price_data={},
    )
    assert rep["dropped"] == 1
    assert out.empty

    df_missing = pd.DataFrame({"entry_date": [idx[0]]})
    out2, rep2 = _clip_trades_to_eval_window(
        df_missing,
        e0=idx[0],
        e1=idx[2],
        price_data={},
    )
    assert rep2["dropped"] == 0
    assert out2.equals(df_missing)


def test_normalize_and_finalize_trades() -> None:
    df = pd.DataFrame(
        {
            "entry_date": ["2024-01-02"],
            "exit_date": ["2024-01-05"],
            "gross_pnl": [10.0],
        }
    )
    norm = _normalize_trades("AAA-BBB", df)
    assert norm is not None

    norm["entry_date"] = pd.to_datetime(norm["entry_date"])
    norm["exit_date"] = pd.to_datetime(norm["exit_date"])
    norm = engine._recompute_holding_days_inplace(norm)
    assert "holding_days" in norm.columns

    cal = pd.bdate_range("2024-01-02", periods=5)
    finalized = engine._finalize_costs_and_net(
        norm, calendar=cal, price_data={}, borrow_ctx=None
    )
    assert "net_pnl" in finalized.columns


def test_engine_row_helpers_and_risk_gating() -> None:
    row = pd.Series(
        {
            "signal": 1,
            "notional_y": 1000.0,
            "notional_x": 900.0,
            "y_symbol": "AAA",
            "x_symbol": "BBB",
        }
    )
    syms, notionals, gross = engine._infer_leg_payload(row)
    assert syms == ("AAA", "BBB")
    assert gross > 0

    df = pd.DataFrame(
        {
            "pair": ["AAA-BBB", "CCC-DDD"],
            "entry_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "exit_date": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-11")],
            "y_symbol": ["AAA", "CCC"],
            "x_symbol": ["BBB", "DDD"],
            "notional_y": [50000.0, 50000.0],
            "notional_x": [-50000.0, -50000.0],
            "gross_pnl": [10.0, -5.0],
        }
    )
    risk_cfg = {"max_trade_pct": 0.01}
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-02"),
        e1=pd.Timestamp("2024-01-31"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    assert rep["blocked"] >= 0
    if engine.RiskManager is None:
        assert "risk_blocked" not in out.columns
    else:
        assert "risk_blocked" in out.columns


def test_apply_risk_gating_uses_mtm_equity() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5)
    price_data = {
        "AAA": pd.Series([100.0, 100.0, 300.0, 300.0, 300.0], index=idx),
        "BBB": pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx),
    }
    df = pd.DataFrame(
        {
            "pair": ["AAA-BBB", "AAA-BBB"],
            "entry_date": [idx[0], idx[2]],
            "exit_date": [idx[4], idx[3]],
            "y_symbol": ["AAA", "AAA"],
            "x_symbol": ["BBB", "BBB"],
            "notional_y": [300.0, 400.0],
            "notional_x": [-300.0, -400.0],
            "y_units": [3, 4],
            "x_units": [-3, -4],
            "entry_price_y": [100.0, 100.0],
            "entry_price_x": [100.0, 100.0],
        }
    )
    risk_cfg = {
        "max_trade_pct": 0.6,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=idx[0],
        e1=idx[-1],
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data=price_data,
    )
    assert rep["blocked"] == 0
    assert len(out) == 2


def test_apply_risk_gating_blocks_per_name_concentration() -> None:
    df = pd.DataFrame(
        {
            "pair": ["PRE", "NEW"],
            "entry_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")],
            "exit_date": [pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-05")],
            "y_symbol": ["AAA", "AAA"],
            "x_symbol": ["BBB", "DDD"],
            "notional_y": [200.0, 100.0],
            "notional_x": [-200.0, -100.0],
            "net_pnl": [0.0, 0.0],
        }
    )
    risk_cfg = {
        "max_trade_pct": 1.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "max_per_name_pct": 0.25,
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-02"),
        e1=pd.Timestamp("2024-01-06"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 2
    else:
        assert rep["blocked"] == 1
        assert len(out) == 1
        assert out.iloc[0]["pair"] == "PRE"


def test_short_availability_heuristic_blocks_on_price() -> None:
    df = _make_short_heuristic_trade(entry_price_y=4.0)
    risk_cfg = {
        "max_trade_pct": 10.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "short_availability_heuristic": {
            "enabled": True,
            "min_price": 5.0,
            "block_on_missing": True,
        },
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 1
    else:
        assert rep["blocked"] == 1
        assert out.empty

    risk_cfg["short_availability_heuristic"]["min_price"] = 0.0
    out2, rep2 = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    assert rep2["blocked"] == 0
    assert len(out2) == 1


def test_short_availability_heuristic_blocks_on_adv_threshold() -> None:
    df = _make_short_heuristic_trade()
    risk_cfg = {
        "max_trade_pct": 10.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "short_availability_heuristic": {
            "enabled": True,
            "min_adv_usd": 2_000_000.0,
            "block_on_missing": True,
        },
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
        adv_map={"AAA": 1_000_000.0, "BBB": 5_000_000.0},
    )
    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 1
    else:
        assert rep["blocked"] == 1
        assert out.empty


def test_short_availability_heuristic_blocks_on_missing_adv() -> None:
    df = _make_short_heuristic_trade()
    risk_cfg = {
        "max_trade_pct": 10.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "short_availability_heuristic": {
            "enabled": True,
            "min_adv_usd": 2_000_000.0,
            "block_on_missing": True,
        },
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 1
    else:
        assert rep["blocked"] == 1
        assert out.empty


def test_short_availability_heuristic_allows_missing_adv_when_disabled() -> None:
    df = _make_short_heuristic_trade()
    risk_cfg = {
        "max_trade_pct": 10.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "short_availability_heuristic": {
            "enabled": True,
            "min_adv_usd": 2_000_000.0,
            "block_on_missing": False,
        },
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
    )
    assert rep["blocked"] == 0
    assert len(out) == 1


def test_short_availability_heuristic_blocks_on_missing_price() -> None:
    df = _make_short_heuristic_trade(entry_price_y=None)
    risk_cfg = {
        "max_trade_pct": 10.0,
        "max_gross_exposure": 10.0,
        "max_net_exposure": 10.0,
        "short_availability_heuristic": {
            "enabled": True,
            "min_price": 5.0,
            "block_on_missing": True,
        },
    }
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-10"),
        initial_capital=1000.0,
        risk_cfg=risk_cfg,
        price_data={},
        adv_map={"AAA": 5_000_000.0, "BBB": 5_000_000.0},
    )
    if engine.RiskManager is None:
        assert rep["blocked"] == 0
        assert len(out) == 1
    else:
        assert rep["blocked"] == 1
        assert out.empty


def test_misc_helpers_and_eval_window() -> None:
    base = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=[0, 1])
    out = pd.DataFrame({"x": [10, 20]}, index=[0, 1])
    restored = engine._restore_essential_columns(out, base)
    assert "a" in restored.columns

    res = engine._perf_run("noop", lambda: 5)
    assert res == 5

    cfg = make_config_from_yaml(
        {
            "backtest": {
                "splits": {
                    "train": {"start": "2024-01-02", "end": "2024-01-05"},
                    "test": {"start": "2024-01-06", "end": "2024-01-10"},
                }
            }
        }
    )
    cal = pd.bdate_range("2024-01-02", periods=10)
    e0, e1 = _resolve_eval_window(cal, cfg)
    assert e0 <= e1


def test_make_config_from_yaml_rejects_legacy_walkforward_and_window_end_keys() -> None:
    with pytest.raises(ValueError):
        make_config_from_yaml(
            {
                "backtest": {
                    "window_end_policy": "force_exit",
                    "splits": {
                        "train": {"start": "2024-01-02", "end": "2024-01-05"},
                        "test": {"start": "2024-01-06", "end": "2024-01-10"},
                    },
                }
            }
        )
    with pytest.raises(ValueError):
        make_config_from_yaml(
            {
                "backtest": {
                    "walkforward": {"carry_open_trades": True, "stateful_sizing": True},
                    "splits": {
                        "train": {"start": "2024-01-02", "end": "2024-01-05"},
                        "test": {"start": "2024-01-06", "end": "2024-01-10"},
                    },
                }
            }
        )


def test_make_config_from_yaml_rejects_legacy_lob_schema_keys() -> None:
    splits = {
        "train": {"start": "2024-01-02", "end": "2024-01-05"},
        "test": {"start": "2024-01-06", "end": "2024-01-10"},
    }

    with pytest.raises(ValueError, match="execution.exec_lob"):
        make_config_from_yaml(
            {
                "backtest": {"splits": splits},
                "execution": {"mode": "lob", "exec_lob": {"tick": 0.01}},
            }
        )

    with pytest.raises(ValueError, match="execution.lob.policy"):
        make_config_from_yaml(
            {
                "backtest": {"splits": splits},
                "execution": {"mode": "lob", "lob": {"policy": {"post_only": True}}},
            }
        )

    with pytest.raises(ValueError, match="execution.lob.post_costs.borrow_bps"):
        make_config_from_yaml(
            {
                "backtest": {"splits": splits},
                "execution": {
                    "mode": "lob",
                    "lob": {"post_costs": {"borrow_bps": 0.0}},
                },
            }
        )


def test_engine_time_helpers_extra() -> None:
    s_empty = pd.Series([], dtype="datetime64[ns]")
    out_empty = engine._to_ex_tz_series(s_empty, "America/New_York", naive_is_utc=True)
    assert out_empty.empty

    s = pd.Series(pd.date_range("2024-01-01", periods=2, freq="D"))
    out = engine._to_ex_tz_series(s, "America/New_York", naive_is_utc=True)
    assert str(out.dt.tz) == "America/New_York"

    ts = engine._to_ex_tz_timestamp(pd.NaT, "America/New_York", naive_is_utc=False)
    assert pd.isna(ts)

    cal = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2, tz="UTC"))
    cal2 = _ensure_calendar_tz(cal, "America/New_York")
    assert str(cal2.tz) == "America/New_York"

    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    s_tz = pd.Series(pd.date_range("2024-01-01 12:00", periods=2, freq="D", tz="UTC"))
    coerced = _coerce_like_index(s_tz, idx)
    assert coerced.dt.hour.eq(0).all()


def test_engine_asof_price_edge_cases() -> None:
    s_bad = pd.Series([1.0, 2.0], index=[0, 1])
    assert engine._asof_price_for_ts(s_bad, pd.Timestamp("2024-01-01")) is None

    idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    assert engine._asof_price_for_ts(s, pd.Timestamp("2024-01-02")) == 2.0
    assert engine._asof_price_for_ts(s, pd.Timestamp("2023-12-31")) is None
    assert (
        engine._asof_price_for_ts(pd.Series(dtype=float), pd.Timestamp("2024-01-01"))
        is None
    )


def test_normalize_trades_variants() -> None:
    assert _normalize_trades("AAA-BBB", None) is None
    assert _normalize_trades("AAA-BBB", object()) is None

    df_alias = pd.DataFrame(
        {
            "entry_ts": [pd.Timestamp("2024-01-01")],
            "exit_ts": [pd.Timestamp("2024-01-02")],
        }
    )
    norm = _normalize_trades("AAA-BBB", df_alias)
    assert norm is not None
    assert "entry_date" in norm.columns
    assert "exit_date" in norm.columns

    df_idx = pd.DataFrame(
        {"exit_ts": [pd.Timestamp("2024-01-02")]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")]),
    )
    norm_idx = _normalize_trades("AAA-BBB", df_idx)
    assert norm_idx is not None
    assert "entry_date" in norm_idx.columns

    df_bad = pd.DataFrame({"entry_date": [pd.NaT], "exit_date": [pd.NaT]})
    assert _normalize_trades("AAA-BBB", df_bad) is None

    df_missing = pd.DataFrame({"foo": [1]})
    assert _normalize_trades("AAA-BBB", df_missing) is None


def test_recompute_holding_days_with_nat() -> None:
    df = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-01"), pd.NaT],
            "exit_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
        }
    )
    out = engine._recompute_holding_days_inplace(df)
    assert int(out.loc[0, "holding_days"]) >= 1


def test_finalize_costs_with_borrow_error(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-01")],
            "exit_date": [pd.Timestamp("2024-01-02")],
            "gross_pnl": [10.0],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
        }
    )
    cal = pd.bdate_range("2024-01-01", periods=2)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "backtest.simulators.engine_trades.compute_borrow_cost_for_trade_row", _boom
    )

    class BorrowCtx:
        enabled = True

    out = engine._finalize_costs_and_net(
        df, calendar=cal, price_data={}, borrow_ctx=BorrowCtx()
    )
    assert "borrow_cost" in out.columns
    assert "net_pnl" in out.columns


def test_perf_run_logs_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyRes:
        def __init__(self) -> None:
            self.value = "ok"
            self.runtime_sec = 0.01

    def _fake_measure(fn):
        return DummyRes()

    monkeypatch.setattr(engine, "measure_runtime", _fake_measure)
    assert engine._perf_run("dummy", lambda: "ok") == "ok"


def test_infer_leg_payload_gross_fallback() -> None:
    row = pd.Series({"gross_notional": 100.0, "signal": -1})
    syms, notionals, gross = engine._infer_leg_payload(row)
    assert gross == 100.0
    assert notionals[0] < 0 and notionals[1] > 0


def test_infer_leg_payload_from_size_and_price() -> None:
    row = pd.Series({"size": 10, "entry_price": 2.0, "signal": 1})
    _syms, _notionals, gross = engine._infer_leg_payload(row)
    assert gross == 20.0


def test_engine_row_helpers_error_paths() -> None:
    class BadFloat:
        def __float__(self):
            raise TypeError("bad float")

    row = pd.Series({"bad": BadFloat()})
    assert engine._get_first_present(row, ("bad",)) is None


def test_clip_trades_to_eval_window_empty_df() -> None:
    empty = pd.DataFrame()
    out, rep = _clip_trades_to_eval_window(
        empty,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-02"),
        price_data={},
    )
    assert rep["dropped"] == 0
    assert out.empty


def test_apply_risk_gating_missing_time_cols() -> None:
    df = pd.DataFrame({"pair": ["AAA-BBB"], "entry_date": [pd.Timestamp("2024-01-01")]})
    out, rep = engine._apply_risk_gating(
        df,
        e0=pd.Timestamp("2024-01-01"),
        e1=pd.Timestamp("2024-01-02"),
        initial_capital=1000.0,
        risk_cfg={},
        price_data={},
    )
    assert rep["blocked"] == 0
    assert out.equals(df)


def test_apply_borrow_event_enforcement_clip_exit() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-01")],
            "exit_date": [pd.Timestamp("2024-01-10")],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "notional_y": [100.0],
            "notional_x": [-100.0],
        }
    )
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "clip_exit",
                "recall_grace_days": 1,
                "buyin_penalty_bps": 10.0,
            }
        }
    }

    class BorrowCtx:
        def events_for_range(self, _syms, _start, _end):
            return pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")],
                    "symbol": ["AAA", "AAA"],
                    "type": ["recall_notice", "buy_in_effective"],
                }
            )

        availability_long = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02", tz="America/New_York")],
                "symbol": ["AAA"],
                "available": [0],
            }
        )

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=BorrowCtx()
    )
    assert rep["changed_exits"] == 1
    assert rep["buyin_penalties"] == 1
    assert bool(out.loc[out.index[0], "hard_exit"]) is True
    assert out.loc[out.index[0], "buyin_penalty_cost"] < 0


def test_apply_execution_hooks_strict_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-03")],
            "pair": ["AAA-BBB"],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
        }
    )
    cfg = make_config_from_yaml(
        {
            "_bo_require_execution_hooks": True,
            "execution": {"mode": "lob", "lob": {"enabled": True}},
            "backtest": {
                "splits": {
                    "train": {"start": "2024-01-01", "end": "2024-01-02"},
                    "test": {"start": "2024-01-03", "end": "2024-01-04"},
                }
            },
        }
    )
    monkeypatch.setattr(engine, "_LOB_OK", True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("lob blew up")

    monkeypatch.setattr(engine, "annotate_with_lob", _boom)

    with pytest.raises(RuntimeError, match="strict mode"):
        engine._apply_execution_hooks(
            trades,
            base_cols=trades[
                ["entry_date", "exit_date", "pair", "y_symbol", "x_symbol"]
            ].copy(),
            price_data={},
            cfg=cfg,
            calendar=pd.bdate_range("2024-01-02", periods=2),
            e0=pd.Timestamp("2024-01-02"),
            market_data_panel=None,
            adv_map=None,
        )
