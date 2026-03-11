import numpy as np
import pandas as pd
import pytest

from backtest.simulators import costs


def test_exec_params_and_slippage_models() -> None:
    params_power = costs.exec_params_from_cfg(
        {
            "execution": {
                "heuristic": {
                    "impact_model": "power",
                    "power_alpha": 0.5,
                    "power_coefficient": 0.2,
                    "impact_min_bps": 5.0,
                    "max_slippage": 0.01,
                }
            }
        }
    )
    assert params_power.impact_model == "power"

    slip_power = costs.calc_adv_slippage(
        1_000.0, adv=10_000.0, vol=0.02, params=params_power
    )
    assert 0.0 <= slip_power <= params_power.max_slippage

    params_linear = costs.exec_params_from_cfg(
        {
            "execution": {
                "heuristic": {
                    "impact_model": "linear",
                    "adv_impact_k": 0.5,
                    "max_slippage": 0.1,
                }
            }
        }
    )
    slip_linear = costs.calc_adv_slippage(
        1_000.0, adv=10_000.0, vol=0.02, params=params_linear
    )
    assert slip_linear >= params_linear.base_slippage

    slip_base = costs.calc_adv_slippage(
        1_000.0, adv=None, vol=None, params=params_linear
    )
    assert slip_base == pytest.approx(params_linear.base_slippage)

    p_full = costs.partial_fill_probability(0.0, adv=None, params=params_linear)
    assert p_full == 1.0

    p_low_adv = costs.partial_fill_probability(1_000.0, adv=100.0, params=params_linear)
    assert params_linear.min_fill_prob <= p_low_adv <= 1.0


def test_simulate_execution_and_trade_costs() -> None:
    res = costs.simulate_execution(
        size=10,
        price_y=10.0,
        price_x=20.0,
        per_trade_fixed=0.1,
        adv=10_000.0,
        vol=0.02,
        params={"impact_model": "sqrt"},
    )
    assert 0 <= res.filled_size <= res.requested_size
    assert res.total_cost == pytest.approx(res.total_slippage_cost + res.total_fee_cost)
    assert 0.0 <= res.filled_notional_ratio <= 1.0

    cost_zero = costs.calc_trade_cost(
        size=0,
        price_y=10.0,
        price_x=10.0,
        per_trade_fixed=0.1,
        slippage_pct=0.01,
        min_fee=1.0,
        charge_fixed_when_zero_fill=False,
    )
    assert cost_zero == 1.0

    cost_min = costs.calc_trade_cost(
        size=1,
        price_y=10.0,
        price_x=10.0,
        per_trade_fixed=0.0,
        slippage_pct=0.0,
        min_fee=1.0,
    )
    assert cost_min == 1.0


def test_fee_schedule_compute_costs_and_post_lob() -> None:
    cfg = {
        "venues": {
            "XNAS": {
                "fees": {
                    "maker_bps": -1.0,
                    "taker_bps": 2.0,
                    "maker_per_share": 0.001,
                    "taker_per_share": 0.002,
                    "min_fee": 0.1,
                    "max_fee": 10.0,
                    "commission_per_share": 0.0005,
                    "clearing_per_trade": 0.01,
                    "clearing_min_per_trade": 0.0,
                },
                "maker_tiers": [
                    {"thresh": 0.0, "bps": -0.5},
                    {"thresh": 1_000.0, "bps": -1.0},
                ],
                "taker_tiers": [
                    {"metric": "shares", "thresh": 0.0, "bps": 2.0, "per_share": 0.002}
                ],
            }
        },
        "backtest": {
            "auctions": {
                "fees": {
                    "venue_profiles": {
                        "XNAS": {"open": {"bps": 1.0}, "close": {"bps": 2.5}}
                    }
                }
            }
        },
        "execution": {
            "mode": "light",
            "light": {"enabled": True, "fees": {"bps": 0.1, "per_share": 0.0002}},
            "fees": {
                "default": {"maker_bps": -0.5, "taker_bps": 1.5, "min_fee": 0.05},
                "venue_overrides": {"XNAS": {"taker_bps": 2.5}},
                "global_commission_per_share": 0.0001,
                "global_clearing_per_trade": 0.02,
                "global_clearing_min_per_trade": 0.01,
            },
        },
    }
    schedule = costs.normalize_fee_schedule_from_cfg(cfg)

    fills = pd.DataFrame(
        {
            "venue": ["XNAS", "XNAS"],
            "qty": [100, 200],
            "vwap": [10.0, 20.0],
            "maker_flag": [True, False],
            "is_open_auction": [True, False],
            "order_kind": ["MOO", "MOC"],
            "slippage_cost": [-1.0, -2.0],
            "impact_cost": [-0.5, -0.2],
        }
    )
    out = costs.compute_costs(fills, schedule)
    assert {"fee", "fees", "auctions_cost", "total_costs", "fee_bps_equiv"}.issubset(
        out.columns
    )
    assert np.isfinite(out["total_costs"]).all()

    trades = pd.DataFrame(
        {
            "units_y": [10],
            "units_x": [10],
            "exec_entry_vwap_y": [10.0],
            "exec_entry_vwap_x": [20.0],
            "exec_exit_vwap_y": [11.0],
            "exec_exit_vwap_x": [21.0],
            "liquidity_entry_y": ["maker"],
            "liquidity_entry_x": ["taker"],
            "liquidity_exit_y": ["taker"],
            "liquidity_exit_x": ["maker"],
        }
    )
    post = costs.compute_post_lob_costs(
        trades,
        {
            "post_costs": {
                "per_trade": 0.1,
                "maker_bps": -1.0,
                "taker_bps": 2.0,
                "min_fee": 0.01,
            }
        },
    )
    assert {"fees", "total_costs", "borrow_cost"}.issubset(post.columns)


def test_compute_costs_with_maker_share_and_flags() -> None:
    cfg = {
        "venues": {
            "XNYS": {
                "fees": {"maker_bps": -0.5, "taker_bps": 1.0, "maker_rebate_bps": -0.2}
            }
        },
        "execution": {
            "mode": "light",
            "light": {"enabled": True, "fees": {"bps": 0.05}},
        },
    }
    schedule = costs.normalize_fee_schedule_from_cfg(cfg)

    fills = pd.DataFrame(
        {
            "venue": ["XNYS", "XNYS"],
            "qty": [50, 75],
            "vwap": [20.0, 21.0],
            "maker_share_eff": [0.8, 0.2],
            "order_kind": ["MOO", "MOC"],
            "slippage_cost": [0.0, 0.0],
            "impact_cost": [0.0, 0.0],
        }
    )
    out = costs.compute_costs(
        fills,
        schedule,
        auction_flags={
            "is_open_auction": [True, False],
            "is_close_auction": [False, True],
        },
    )
    assert out["fee"].shape[0] == 2
    assert out["auction_applied"].isin(["open", "close", ""]).any()


def test_exec_params_sanitized_and_misc_helpers() -> None:
    bad = costs.ExecParams(
        impact_model="invalid", min_fill_prob=-1.0, power_coefficient=-2.0
    )
    cleaned = bad.sanitized()
    assert cleaned.impact_model == "sqrt"
    assert 0.0 <= cleaned.min_fill_prob <= 1.0
    assert cleaned.power_coefficient == 0.0

    cfg = {"execution": {"heuristic": {"power_coefficient": "bad"}}}
    params = costs.exec_params_from_cfg(cfg)
    assert params.power_coefficient is None

    res = costs.SimulationResult(
        requested_size=0,
        filled_size=0,
        fill_ratio=0.0,
        fill_probability=1.0,
        notional_requested=0.0,
        notional_filled=0.0,
        slippage_pct_total=0.0,
        slippage_pct_base=0.0,
        slippage_pct_impact=0.0,
        slippage_cost_base=0.0,
        slippage_cost_impact=0.0,
        fixed_fee_cost=0.0,
        min_fee_topup=0.0,
        total_fee_cost=0.0,
        total_cost=0.0,
    )
    assert res.filled_notional_ratio == 0.0

    assert costs.square_root_impact(-1.0, coeff=0.2) == 0.0
    assert costs.calc_adv_slippage(-1.0, adv=100.0, vol=0.1, params=cleaned) == 0.0

    assert costs._combine_adv(100.0, None) == 100.0
    assert costs._combine_vol(None, 0.2) == 0.2


def test_exec_params_from_cfg_error_paths() -> None:
    class BadMapping:
        def get(self, *_args, **_kwargs):
            raise RuntimeError("bad get")

    cfg = {"execution": {"heuristic": BadMapping()}}
    params = costs.exec_params_from_cfg(cfg)
    assert params.impact_model == costs.DEFAULT_EXECUTION_PARAMS.impact_model


def test_simulate_execution_and_price_helpers_edge_cases() -> None:
    class BadFloat:
        def __float__(self):
            raise TypeError("bad float")

    res = costs.simulate_execution(
        size=BadFloat(),
        price_y=10.0,
        price_x=10.0,
        per_trade_fixed=0.1,
        adv=1_000.0,
        vol=0.2,
        params={"min_fee": 1.0},
        charge_fixed_when_zero_fill=False,
    )
    assert res.total_fee_cost >= 0.0

    bad_row = pd.Series({"entry_price_y": -1.0, "entry_price_x": "bad"})
    assert costs._price_for_leg(bad_row, "y") is None
    assert costs._price_for_leg(bad_row, "x") is None

    liq_row = pd.Series({"liquidity_y": "maker", "x_maker": True})
    assert costs._liquidity_flag(liq_row, "y") == "maker"
    assert costs._liquidity_flag(liq_row, "x") == "maker"


def test_compute_costs_empty_and_alt_columns() -> None:
    empty = costs.compute_costs(
        pd.DataFrame(), costs.normalize_fee_schedule_from_cfg({})
    )
    assert empty.empty

    fills = pd.DataFrame(
        {
            "venue": ["XNAS"],
            "qty_y": [10],
            "qty_x": [5],
            "px_y": [10.0],
            "px_x": [20.0],
            "role": ["maker"],
        }
    )
    schedule = costs.normalize_fee_schedule_from_cfg(
        {"venues": {"XNAS": {"fees": {"maker_bps": -1.0}}}}
    )
    out = costs.compute_costs(fills, schedule)
    assert np.isfinite(out["fee"]).all()


def test_compute_post_lob_costs_empty_and_caps() -> None:
    empty = costs.compute_post_lob_costs(
        pd.DataFrame(), {"post_costs": {"per_trade": 0.1}}
    )
    assert empty.empty

    trades = pd.DataFrame(
        {
            "units_y": [10],
            "units_x": [10],
            "exec_entry_vwap_y": [10.0],
            "exec_entry_vwap_x": [10.0],
            "exec_exit_vwap_y": [10.0],
            "exec_exit_vwap_x": [10.0],
            "liquidity_entry_y": ["taker"],
            "liquidity_entry_x": ["taker"],
            "liquidity_exit_y": ["taker"],
            "liquidity_exit_x": ["taker"],
        }
    )
    post = costs.compute_post_lob_costs(
        trades,
        {"post_costs": {"per_trade": 1.0, "taker_bps": 50.0, "max_fee": 0.5}},
    )
    assert post["fees"].iloc[0] >= -0.5
