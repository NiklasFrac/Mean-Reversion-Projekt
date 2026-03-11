from types import SimpleNamespace

import pandas as pd

from backtest.simulators import engine


def _make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_date": ["2024-01-02", "2024-01-02"],
            "exit_date": ["2024-01-10", "2024-01-10"],
            "y_symbol": ["AAA", "CCC"],
            "x_symbol": ["BBB", "DDD"],
            "notional_y": [1_000.0, 1_000.0],
            "notional_x": [1_000.0, 1_000.0],
            "entry_price_y": [10.0, 10.0],
            "entry_price_x": [20.0, 20.0],
        }
    )


def test_borrow_enforcement_clip_exit() -> None:
    trades = _make_trades()
    tz = "America/New_York"
    events = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-04", tz=tz),
                pd.Timestamp("2024-01-03", tz=tz),
            ],
            "symbol": ["AAA", "CCC"],
            "type": ["buyin", "recall_notice"],
        }
    )
    avail = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05", tz=tz)],
            "symbol": ["BBB"],
            "available": [0],
        }
    )
    borrow_ctx = SimpleNamespace(borrow_events=events, availability_long=avail)

    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "clip_exit",
                "recall_grace_days": 1,
                "buyin_penalty_bps": 10.0,
            },
            "ftd_block_threshold": 0.0,
        }
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert rep["changed_exits"] > 0
    assert bool(out.loc[out["y_symbol"] == "AAA", "hard_exit"].iloc[0]) is True
    assert out.loc[out["y_symbol"] == "AAA", "exit_date"].iloc[0] == pd.Timestamp(
        "2024-01-04", tz=tz
    )
    assert out.loc[out["y_symbol"] == "AAA", "buyin_penalty_cost"].iloc[0] < 0.0
    assert (
        out.loc[out["y_symbol"] == "CCC", "hard_exit_reason"].iloc[0] == "recall_grace"
    )


def test_borrow_enforcement_penalty_only() -> None:
    trades = _make_trades()
    tz = "America/New_York"
    events = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-04", tz=tz)],
            "symbol": ["AAA"],
            "type": ["buy_in_effective"],
        }
    )
    borrow_ctx = SimpleNamespace(borrow_events=events)
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "penalty_only",
                "buyin_penalty_bps": 5.0,
            }
        }
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert rep["buyin_penalties"] >= 1
    assert out.loc[out["y_symbol"] == "AAA", "exit_date"].iloc[0] == pd.Timestamp(
        "2024-01-10", tz=tz
    )


def test_borrow_enforcement_availability_synth_buyin() -> None:
    trades = _make_trades()
    tz = "America/New_York"
    avail = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05", tz=tz)],
            "symbol": ["BBB"],
            "available": [0],
        }
    )
    borrow_ctx = SimpleNamespace(availability_long=avail)
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "penalty_only",
                "buyin_penalty_bps": 10.0,
            },
            "ftd_block_threshold": 0.0,
        }
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert rep["buyin_penalties"] >= 1
    assert out["buyin_penalty_cost"].min() < 0.0


def test_borrow_enforcement_missing_dates_and_availability_strings() -> None:
    tz = "America/New_York"
    trades = pd.DataFrame(
        {
            "entry_ts": [pd.Timestamp("2024-01-02", tz=tz), pd.NaT],
            "exit_ts": [
                pd.Timestamp("2024-01-06", tz=tz),
                pd.Timestamp("2024-01-07", tz=tz),
            ],
            "pair_key": ["AAA/BBB", "AAA/BBB"],
            "notional_y": [1_000.0, 1_000.0],
            "notional_x": [1_000.0, 1_000.0],
        }
    )
    events = pd.DataFrame(
        {
            "day": [
                pd.Timestamp("2024-01-03", tz=tz),
                pd.Timestamp("2024-01-04", tz=tz),
            ],
            "ticker": ["AAA", "AAA"],
            "event": ["recall_notice", ""],
            "available": [0, 0],
        }
    )
    avail = pd.DataFrame(
        {
            "day": [pd.Timestamp("2024-01-04", tz=tz)],
            "ticker": ["BBB"],
            "available": ["no"],
        }
    )
    borrow_ctx = SimpleNamespace(borrow_events=events, availability_long=avail)
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "clip_exit",
                "recall_grace_days": 1,
                "buyin_penalty_bps": 10.0,
            },
            "ftd_block_threshold": "nan",
        }
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert rep["changed_exits"] >= 1
    assert out["hard_exit"].any()
    assert out["buyin_penalty_cost"].min() < 0.0


def test_borrow_enforcement_penalty_only_recall_grace() -> None:
    tz = "America/New_York"
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-02", tz=tz)],
            "exit_date": [pd.Timestamp("2024-01-10", tz=tz)],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "notional_y": [1_000.0],
            "notional_x": [1_000.0],
        }
    )
    events = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05", tz=tz)],
            "symbol": ["AAA"],
            "type": ["recall_notice"],
        }
    )
    borrow_ctx = SimpleNamespace(borrow_events=events)
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "penalty_only",
                "recall_grace_days": 1,
            }
        }
    }

    out, _rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert bool(out.loc[out.index[0], "hard_exit"]) is True
    assert out.loc[out.index[0], "hard_exit_reason"] == "recall_grace"


def test_borrow_enforcement_gross_fallback_from_qty_prices() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-06")],
            "symbol": ["AAA"],
            "qty_y": [10],
            "qty_x": [5],
            "entry_price_y": [10.0],
            "entry_price_x": [20.0],
        }
    )
    events = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03")],
            "symbol": ["AAA"],
            "type": ["buy_in_effective"],
        }
    )
    borrow_ctx = SimpleNamespace(borrow_events=events)
    raw_yaml = {
        "borrow": {
            "enforcement": {
                "enabled": True,
                "mode": "penalty_only",
                "buyin_penalty_bps": 10.0,
            }
        }
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=borrow_ctx
    )
    assert rep["buyin_penalties"] == 1
    assert out["buyin_penalty_cost"].iloc[0] < 0.0


def test_borrow_enforcement_handles_bad_threshold_and_empty_events() -> None:
    class BadFloat:
        def __float__(self):
            raise TypeError("bad float")

    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-03")],
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
        }
    )

    class BorrowCtx:
        def events_for_range(self, *_args, **_kwargs):
            return pd.DataFrame()

    raw_yaml = {
        "borrow": {"enforcement": {"enabled": True}, "ftd_block_threshold": BadFloat()},
    }

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=BorrowCtx()
    )
    assert rep["changed_exits"] == 0
    assert list(out.columns) == list(trades.columns)
    assert out["entry_date"].dt.tz is not None


def test_borrow_enforcement_skips_when_no_symbols() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": [pd.Timestamp("2024-01-02")],
            "exit_date": [pd.Timestamp("2024-01-03")],
        }
    )
    raw_yaml = {"borrow": {"enforcement": {"enabled": True}}}

    out, rep = engine._apply_borrow_event_enforcement(
        trades, raw_yaml=raw_yaml, borrow_ctx=None
    )
    assert rep["changed_exits"] == 0
    assert list(out.columns) == list(trades.columns)
    assert out["entry_date"].dt.tz is not None
