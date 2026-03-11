# tests/property/test_config_properties.py
from __future__ import annotations

from hypothesis import given, settings

from .legacy_config_schema import BacktestConfig
from .strategies_config import valid_cfgs


@settings(deadline=None, max_examples=200)
@given(valid_cfgs())
def test_roundtrip_idempotent(cfg: BacktestConfig):
    d = cfg.model_dump(mode="json")
    cfg2 = BacktestConfig.model_validate(d)
    assert cfg2 == cfg


@settings(deadline=None, max_examples=200)
@given(valid_cfgs())
def test_calendar_invariants(cfg: BacktestConfig):
    assert cfg.calendar.lag_bars >= 0


@settings(deadline=None, max_examples=200)
@given(valid_cfgs())
def test_execution_guards(cfg: BacktestConfig):
    if cfg.execution.mode == "lob":
        assert cfg.execution.tick > 0
        assert cfg.execution.levels >= 1
        assert cfg.execution.min_spread_ticks >= 1
        assert cfg.execution.steps_per_day >= 1
    if cfg.execution.mode == "light":
        fees = cfg.execution.light.fees
        assert fees.per_trade >= 0
        assert fees.bps >= 0
        assert fees.per_share >= 0
        assert fees.min_fee >= 0
        assert fees.max_fee >= 0
