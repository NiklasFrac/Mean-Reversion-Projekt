from execution import calc_adv_slippage, simulate_execution


def test_calc_adv_slippage_basic():
    params = {
        "base_slippage": 0.0002,
        "sqrt_coefficient": 0.08,
        "adv_impact_k": 0.5,
        "impact_model": "sqrt",
        "max_slippage": 0.02,
    }
    # small notional relative to ADV -> slip approx base
    slip_small = calc_adv_slippage(notional=1e4, adv=1e8, vol=0.02, params=params)
    assert slip_small >= params["base_slippage"]
    assert slip_small < 0.001  # should be small

    # large notional -> larger slippage but capped
    slip_big = calc_adv_slippage(notional=1e7, adv=1e8, vol=0.02, params=params)
    assert slip_big >= slip_small
    assert slip_big <= params["max_slippage"]


def test_simulate_execution_returns_consistent_values():
    params = {
        "base_slippage": 0.0002,
        "sqrt_coefficient": 0.08,
        "impact_model": "sqrt",
        "max_slippage": 0.02,
        "min_fee": 0.5,
        "fill_prob_k": 3.0,
        "min_fill_prob": 0.05,
    }
    filled, cost, slip = simulate_execution(
        size=100,
        price_y=100.0,
        price_x=100.0,
        per_trade_fixed=1.0,
        adv=1e7,
        vol=0.02,
        params=params,
    )
    # deterministic expected properties
    assert isinstance(filled, int)
    assert 0 <= filled <= 100
    assert isinstance(cost, float)
    assert cost >= 0.0
    assert 0.0 <= slip <= params["max_slippage"]
