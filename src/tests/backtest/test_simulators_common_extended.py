import pandas as pd

from backtest.config.types import PricingCfg
from backtest.simulators import common


def test_common_helpers() -> None:
    cfg_vwap = PricingCfg(reference="vwap_target")
    assert (
        common.pick_reference_price(
            pricing_cfg=cfg_vwap, side="buy", submit_mid=100.0, target_vwap=101.0
        )
        == 101.0
    )

    cfg_mid = PricingCfg(reference="mid_on_submit")
    assert (
        common.pick_reference_price(
            pricing_cfg=cfg_mid, side="buy", submit_mid=100.0, target_vwap=101.0
        )
        == 100.0
    )

    assert (
        common.cap_cross_price(side="buy", submit_mid=100.0, cap_bps=10.0)
        == 100.0 * 1.001
    )
    assert (
        common.cap_cross_price(side="sell", submit_mid=100.0, cap_bps=10.0)
        == 100.0 * 0.999
    )

    row_units = pd.Series({"units_y": -5, "price_y": 10.0})
    assert common.infer_units(row_units, "y") == 5

    row_notional = pd.Series({"notional_y": 1000.0, "price_y": 20.0})
    assert common.infer_units(row_notional, "y") == 50

    row_side = pd.Series({"side_y": "sell"})
    assert common.infer_side(row_side, "y") == "sell"

    row_qty = pd.Series({"qty_y": -3})
    assert common.infer_side(row_qty, "y") == "sell"

    assert common.opposite_side("buy") == "sell"

    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3))
    assert common.price_at_or_prior(s, pd.Timestamp("2024-01-02")) == 2.0

    assert common.limit_from_bps("buy", 100.0, 10.0) == 100.0 * 1.001
    assert common.limit_from_bps("sell", 100.0, 10.0) == 100.0 * 0.999


def test_common_helpers_edge_cases() -> None:
    cfg_vwap = PricingCfg(reference="vwap_target")
    assert (
        common.pick_reference_price(
            pricing_cfg=cfg_vwap, side="buy", submit_mid=None, target_vwap="nan"
        )
        is None
    )
    assert (
        common.pick_reference_price(
            pricing_cfg=cfg_vwap, side="buy", submit_mid=float("nan"), target_vwap=None
        )
        is None
    )

    assert common.cap_cross_price(side="buy", submit_mid=None, cap_bps=10.0) is None
    assert common.cap_cross_price(side="buy", submit_mid=100.0, cap_bps=None) is None

    row_generic = pd.Series({"quantity": -7})
    assert common.infer_units(row_generic, "y") == 7
    assert common.infer_units(pd.Series({}), "y") == 1
    assert common.infer_side(pd.Series({}), "y", default="sell") == "sell"

    s_bad_idx = pd.Series([1.0, 2.0], index=[0, 1])
    assert common.price_at_or_prior(s_bad_idx, pd.Timestamp("2024-01-02")) is None
    assert (
        common.price_at_or_prior(pd.Series(dtype=float), pd.Timestamp("2024-01-02"))
        is None
    )

    assert common.limit_from_bps("buy", None, 10.0) is None
    assert common.limit_from_bps("buy", 100.0, 0.0) is None
