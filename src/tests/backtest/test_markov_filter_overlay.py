from __future__ import annotations

import pandas as pd
import pytest

from backtest.strat import baseline
from backtest.strat.baseline import BaselineZScoreStrategy
from backtest.strat.markov_filter import build_markov_entry_filter


def test_positions_from_z_entry_gate_blocks_only_new_entries() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-2.5, 0.0, -2.5, -1.0], index=idx)
    entry_gate = pd.Series([False, True, True, True], index=idx, dtype=bool)

    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
        entry_gate=entry_gate,
    )

    assert pos.loc[idx[0]] == 0
    assert pos.loc[idx[2]] == 1


def test_positions_from_z_entry_gate_does_not_affect_exits() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    z = pd.Series([-2.5, -1.0, 0.0], index=idx)
    entry_gate = pd.Series([True, False, False], index=idx, dtype=bool)

    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
        entry_gate=entry_gate,
    )

    assert pos.loc[idx[0]] == 1
    assert pos.loc[idx[1]] == 1
    assert pos.loc[idx[2]] == 0


def test_markov_entry_filter_allows_high_reversion_state() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    z = pd.Series(
        [2.5, 0.0, 2.6, 0.0, 2.7, 0.0, 2.8, 0.0, 2.9, 0.0, 2.9, 0.0], index=idx
    )
    cfg = {
        "markov_filter": {
            "enabled": True,
            "horizon_days": 1,
            "min_revert_prob": 0.8,
            "min_train_observations": 4,
            "min_state_observations": 2,
        }
    }

    out = build_markov_entry_filter(
        cfg,
        z=z,
        train_index=idx[:8],
        eval_index=idx[8:],
        entry_z=2.0,
        exit_z=0.5,
    )

    assert out.diagnostics["active"] is True
    assert out.hit_prob.loc[idx[8]] == pytest.approx(1.0)
    assert bool(out.entry_gate.loc[idx[8]]) is True


def test_markov_entry_filter_blocks_low_reversion_state() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    z = pd.Series([2.5, 2.6, 2.7, 2.8, 2.9, 2.7, 2.6, 2.5, 2.8, 2.7], index=idx)
    cfg = {
        "markov_filter": {
            "enabled": True,
            "horizon_days": 1,
            "min_revert_prob": 0.1,
            "min_train_observations": 4,
            "min_state_observations": 2,
        }
    }

    out = build_markov_entry_filter(
        cfg,
        z=z,
        train_index=idx[:8],
        eval_index=idx[8:],
        entry_z=2.0,
        exit_z=0.5,
    )

    assert out.diagnostics["active"] is True
    assert out.hit_prob.loc[idx[8]] == pytest.approx(0.0)
    assert bool(out.entry_gate.loc[idx[8]]) is False


def test_markov_entry_filter_passes_through_unseen_state() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    z = pd.Series([-2.5, 0.0, -2.6, 0.0, -2.7, 0.0, 2.5, 2.6], index=idx)
    cfg = {
        "markov_filter": {
            "enabled": True,
            "horizon_days": 1,
            "min_revert_prob": 0.9,
            "min_train_observations": 4,
            "min_state_observations": 2,
        }
    }

    out = build_markov_entry_filter(
        cfg,
        z=z,
        train_index=idx[:6],
        eval_index=idx[6:],
        entry_z=2.0,
        exit_z=0.5,
    )

    assert out.diagnostics["active"] is True
    assert bool(out.entry_gate.loc[idx[6]]) is True
    assert out.diagnostics["state_visit_counts"]["extreme_positive"] == 0


def test_baseline_strategy_markov_overlay_can_remove_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=20)
    prices = pd.DataFrame(
        {
            "y": pd.Series([100.0 + i for i in range(len(idx))], index=idx),
            "x": pd.Series([90.0 + i for i in range(len(idx))], index=idx),
        }
    )
    z = pd.Series(
        [
            0.0,
            2.5,
            2.6,
            2.7,
            2.8,
            2.6,
            2.5,
            2.7,
            2.8,
            0.0,
            2.5,
            2.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        index=idx,
    )

    monkeypatch.setattr(
        baseline, "_estimate_beta_ols_with_intercept", lambda *_a, **_k: 1.0
    )
    monkeypatch.setattr(
        baseline,
        "_rolling_zscore_stats_on_allowed_dates",
        lambda _spread, *, full_index=None, **_k: (
            z.reindex(full_index if full_index is not None else z.index),
            pd.Series(
                0.0,
                index=full_index if full_index is not None else z.index,
                dtype=float,
            ),
            pd.Series(
                1.0,
                index=full_index if full_index is not None else z.index,
                dtype=float,
            ),
        ),
    )

    captured_disabled: dict[str, pd.Series] = {}
    captured_enabled: dict[str, pd.Series] = {}

    def _capture_into(store: dict[str, pd.Series]):
        def _fake_from_signals(**kwargs: object) -> pd.DataFrame:
            store["signals"] = kwargs["signals"].copy()  # type: ignore[index]
            return pd.DataFrame()

        return staticmethod(_fake_from_signals)

    cfg_base = {
        "backtest": {
            "initial_capital": 100000.0,
            "risk_per_trade": 0.01,
            "execution_lag_bars": 1,
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[9].date())},
                "test": {"start": str(idx[10].date()), "end": str(idx[-1].date())},
            },
        },
        "strategy": {"name": "baseline"},
        "signal": {
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 3.0,
            "max_hold_days": 10,
            "volatility_window": 5,
        },
        "spread_zscore": {"z_window": 5, "z_min_periods": 2},
        "risk": {"enabled": False},
        "execution": {"max_participation": 0.0},
    }

    monkeypatch.setattr(
        baseline.TradeBuilder, "from_signals", _capture_into(captured_disabled)
    )
    BaselineZScoreStrategy(cfg_base, borrow_ctx=None)(
        {"AAA-BBB": {"prices": prices, "meta": {"t1": "AAA", "t2": "BBB"}}}
    )

    cfg_markov = {
        **cfg_base,
        "markov_filter": {
            "enabled": True,
            "horizon_days": 1,
            "min_revert_prob": 0.5,
            "min_train_observations": 6,
            "min_state_observations": 2,
        },
    }
    monkeypatch.setattr(
        baseline.TradeBuilder, "from_signals", _capture_into(captured_enabled)
    )
    BaselineZScoreStrategy(cfg_markov, borrow_ctx=None)(
        {"AAA-BBB": {"prices": prices, "meta": {"t1": "AAA", "t2": "BBB"}}}
    )

    sig_disabled = captured_disabled["signals"]
    sig_enabled = captured_enabled["signals"]

    assert int(sig_disabled.loc[idx[10]]) == -1
    assert int(sig_disabled.loc[idx[11]]) == -1
    assert int(sig_enabled.loc[idx[10]]) == 0
    assert int(sig_enabled.loc[idx[11]]) == 0
