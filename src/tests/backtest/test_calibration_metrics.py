from __future__ import annotations

import pytest
import pandas as pd

from backtest.calibration.metrics import derive_exec_metrics


def test_derive_exec_metrics_separates_realized_and_diagnostic_costs() -> None:
    trades = pd.DataFrame(
        {
            "gross_notional": [1_000.0],
            "fees": [-1.0],
            "slippage_cost": [-2.0],
            "impact_cost": [-3.0],
            "buyin_penalty_cost": [-4.0],
            "exec_emergency_penalty_cost": [-5.0],
            "exec_diag_costs_only": [True],
            "lob_adv_usd_y": [1_000_000.0],
            "lob_adv_usd_x": [1_000_000.0],
            "lob_spread_ticks_y": [1.0],
            "lob_spread_ticks_x": [2.0],
        }
    )

    out = derive_exec_metrics(trades)

    assert float(out.loc[0, "exec_realized_costs"]) == pytest.approx(-10.0)
    assert float(out.loc[0, "exec_diagnostic_costs"]) == pytest.approx(-5.0)
    assert float(out.loc[0, "exec_total_costs"]) == pytest.approx(-15.0)
    assert float(out.loc[0, "exec_realized_bps"]) == pytest.approx(100.0)
    assert float(out.loc[0, "exec_diagnostic_bps"]) == pytest.approx(50.0)
    assert float(out.loc[0, "exec_total_bps"]) == pytest.approx(150.0)
