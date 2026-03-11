from __future__ import annotations

from backtest.risk_policy import (
    build_risk_policy,
    cap_units_by_participation,
    cap_units_by_trade_notional,
    size_units_from_risk_budget,
)


def test_build_risk_policy_supports_caps_and_sections() -> None:
    policy = build_risk_policy(
        risk_cfg={
            "caps": {
                "max_gross": 3.0,
                "max_net": 1.2,
                "per_trade": 0.25,
                "per_name": 0.15,
            },
            "max_open_positions": 4,
            "short_availability_heuristic": {"enabled": True, "min_price": 5.0},
        },
        backtest_cfg={"risk_per_trade": 0.02},
        execution_cfg={"max_participation": 0.07},
    )
    assert policy.exposure.max_gross_exposure == 3.0
    assert policy.exposure.max_net_exposure == 1.2
    assert policy.sizing.max_trade_pct == 0.25
    assert policy.sizing.risk_per_trade == 0.02
    assert policy.sizing.max_participation == 0.07
    assert policy.exposure.max_per_name_pct == 0.15
    assert policy.exposure.max_open_positions == 4
    assert policy.short_heuristic.enabled is True


def test_sizing_helpers_apply_caps_consistently() -> None:
    base = size_units_from_risk_budget(
        capital=100_000.0,
        risk_per_trade=0.02,
        per_unit_risk=50.0,
        min_units_if_positive=True,
    )
    assert base == 40

    notional_capped = cap_units_by_trade_notional(
        units=base,
        capital=100_000.0,
        max_trade_pct=0.01,
        per_unit_notional=2_000.0,
    )
    assert notional_capped == 0

    liq_capped = cap_units_by_participation(
        units=20,
        max_participation=0.1,
        adv_sum_usd=50_000.0,
        per_unit_notional=5_000.0,
        require_gt_one_capacity=True,
        min_units_if_positive=True,
    )
    assert liq_capped == 0
