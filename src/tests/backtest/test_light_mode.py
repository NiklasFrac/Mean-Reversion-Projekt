from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backtest.config.cfg import make_config_from_yaml
from backtest.simulators.light import annotate_with_light
from backtest.utils.run.strategy import _validate_cfg_strict


def test_make_config_from_yaml_accepts_light_mode() -> None:
    cfg = make_config_from_yaml(
        {
            "backtest": {
                "splits": {
                    "train": {"start": "2024-01-02", "end": "2024-01-03"},
                    "test": {"start": "2024-01-04", "end": "2024-01-05"},
                }
            },
            "execution": {
                "mode": "light",
                "light": {
                    "reject_on_missing_price": True,
                    "fees": {
                        "per_trade": 0.5,
                        "bps": 1.25,
                        "per_share": 0.01,
                        "min_fee": 0.1,
                        "max_fee": 5.0,
                    },
                },
            },
        }
    )

    assert cfg.exec_mode == "light"
    assert cfg.exec_light is not None
    assert bool(cfg.exec_light["enabled"]) is True
    assert bool(cfg.exec_light["reject_on_missing_price"]) is True
    assert float(cfg.exec_light["fees"]["bps"]) == pytest.approx(1.25)


def test_validate_cfg_strict_accepts_light_mode() -> None:
    _validate_cfg_strict(
        {
            "data": {"prices_path": __file__, "pairs_path": __file__},
            "backtest": {},
            "execution": {"mode": "light"},
        }
    )


def test_annotate_with_light_copies_prices_and_computes_fees() -> None:
    trades = pd.DataFrame(
        {
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [110.0],
            "exit_price_x": [55.0],
            "notional_y": [1000.0],
            "notional_x": [-500.0],
        }
    )
    cfg = SimpleNamespace(
        exec_light={
            "enabled": True,
            "reject_on_missing_price": True,
            "fees": {
                "per_trade": 1.0,
                "bps": 10.0,
                "per_share": 0.0,
                "min_fee": 0.0,
                "max_fee": 0.0,
            },
        },
        raw_yaml={},
    )

    out = annotate_with_light(trades, None, cfg)

    assert float(out.loc[0, "exec_entry_vwap_y"]) == pytest.approx(100.0)
    assert float(out.loc[0, "exec_exit_vwap_x"]) == pytest.approx(55.0)
    assert float(out.loc[0, "fees"]) == pytest.approx(-7.15)
    assert float(out.loc[0, "fees_entry"]) == pytest.approx(-3.5)
    assert float(out.loc[0, "fees_exit"]) == pytest.approx(-3.65)
    assert float(out.loc[0, "slippage_cost"]) == 0.0
    assert float(out.loc[0, "impact_cost"]) == 0.0
    assert bool(out.loc[0, "exec_rejected"]) is False
    assert str(out.loc[0, "exec_mode_used"]) == "light"


def test_annotate_with_light_rejects_missing_prices() -> None:
    trades = pd.DataFrame(
        {
            "y_symbol": ["AAA"],
            "x_symbol": ["BBB"],
            "entry_price_y": [100.0],
            "entry_price_x": [50.0],
            "exit_price_y": [None],
            "exit_price_x": [55.0],
        }
    )
    cfg = SimpleNamespace(
        exec_light={"reject_on_missing_price": True, "fees": {}}, raw_yaml={}
    )

    out = annotate_with_light(trades, None, cfg)

    assert bool(out.loc[0, "exec_rejected"]) is True
    assert str(out.loc[0, "exec_reject_reason"]) == "missing_exit_price"
    assert float(out.loc[0, "fees"]) == 0.0
