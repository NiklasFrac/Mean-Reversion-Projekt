from __future__ import annotations

import pandas as pd
import pytest

from backtest.config.cfg import make_config_from_yaml
from backtest.simulators import lob
from backtest.simulators.liquidity_model import LiquidityModel, LiquidityModelCfg


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            ("AAA", "open"): [100.0, 100.5, 106.0, 106.5, 107.0, 108.0],
            ("AAA", "high"): [101.0, 101.0, 112.0, 107.0, 108.0, 109.0],
            ("AAA", "low"): [99.5, 100.0, 95.0, 106.0, 106.5, 107.5],
            ("AAA", "close"): [100.0, 100.5, 107.5, 106.8, 107.2, 108.2],
            ("AAA", "volume"): [1_000.0, 1_100.0, 8_000.0, 1_000.0, 1_000.0, 1_000.0],
        },
        index=idx,
    )
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    return panel


def test_stress_state_uses_ohlcv_and_is_past_only() -> None:
    panel = _panel()
    cfg = LiquidityModelCfg(
        enabled=True,
        asof_shift=1,
        vol_window=2,
        adv_window=2,
        min_periods_frac=0.5,
        stress_enabled=True,
        stress_intensity=1.0,
    )
    model = LiquidityModel(panel, cfg=cfg, adv_map_usd=None)
    ts = pd.Timestamp("2024-01-03")

    state_before = model.stress_state("AAA", ts)
    assert state_before["gap_bps"] > 0.0
    assert state_before["range_bps"] > 0.0
    assert state_before["volume_rel"] >= 1.0
    assert state_before["stress_regime"] in {"elevated", "stress", "panic"}

    panel_future = panel.copy()
    panel_future.loc[pd.Timestamp("2024-01-06"), ("AAA", "high")] = 500.0
    model_future = LiquidityModel(panel_future, cfg=cfg, adv_map_usd=None)
    state_after = model_future.stress_state("AAA", ts)

    assert state_after == state_before


def test_book_params_apply_stress_modifiers() -> None:
    panel = _panel()
    cfg = LiquidityModelCfg(
        enabled=True,
        asof_shift=1,
        vol_window=2,
        adv_window=2,
        min_periods_frac=0.5,
        stress_enabled=True,
        stress_intensity=2.0,
    )
    model = LiquidityModel(panel, cfg=cfg, adv_map_usd=None)
    base = {
        "tick": 0.01,
        "levels": 3,
        "size_per_level": 1_000,
        "min_spread_ticks": 1,
        "steps_per_day": 1,
        "lam": 1.0,
        "max_add": 50,
        "bias_top": 0.7,
        "cancel_prob": 0.1,
        "max_cancel": 25,
    }

    params = model.book_params("AAA", pd.Timestamp("2024-01-03"), base=base)

    assert params["_stress_regime"] == "panic"
    assert params["_stress_aggr_prob"] == pytest.approx(0.45)
    assert params["_stress_aggr_max_frac"] == pytest.approx(1.0)
    assert params["cancel_prob"] >= 0.35
    assert params["_liq_spread_ticks"] >= 4


def test_override_pnl_is_rejected() -> None:
    yaml_cfg = {
        "backtest": {
            "splits": {
                "train": {"start": "2024-01-01", "end": "2024-01-02"},
                "test": {"start": "2024-01-03", "end": "2024-01-04"},
            }
        },
        "execution": {"mode": "lob", "override_pnl": True, "lob": {}},
    }
    with pytest.raises(ValueError, match="override_pnl"):
        make_config_from_yaml(yaml_cfg)


def test_exit_stress_ladders_are_monotone() -> None:
    assert lob._effective_exit_grace_days(2, "normal") == 2
    assert lob._effective_exit_grace_days(2, "elevated") == 2
    assert lob._effective_exit_grace_days(2, "stress") == 1
    assert lob._effective_exit_grace_days(2, "panic") == 0

    assert lob._panic_cross_bps_for_regime(50.0, "normal") == pytest.approx(50.0)
    assert lob._panic_cross_bps_for_regime(50.0, "elevated") == pytest.approx(75.0)
    assert lob._panic_cross_bps_for_regime(50.0, "stress") == pytest.approx(125.0)
    assert lob._panic_cross_bps_for_regime(50.0, "panic") == pytest.approx(250.0)
