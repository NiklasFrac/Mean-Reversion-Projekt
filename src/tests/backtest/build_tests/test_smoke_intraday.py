# tests/smoke_intraday.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Projektpfad für Module
sys.path.insert(0, str(Path.cwd() / "src" / "backtest" / "src"))

from intraday_exec import annotate_trades_with_intraday_bar  # noqa


def main():
    # Mini-Trade (beide Legs, Stück & Preis gegeben → kein price_data nötig)
    trades = pd.DataFrame(
        [
            {
                "pair": "AAPL-MSFT",
                "entry_date": pd.Timestamp("2024-09-03 09:30"),
                "exit_date": pd.Timestamp("2024-09-10 16:00"),
                "gross_pnl": 1234.56,
                "symbol_y": "AAPL",
                "symbol_x": "MSFT",
                "units_y": 150,
                "units_x": 100,
                "entry_price_y": 185.0,
                "entry_price_x": 330.0,
            }
        ]
    )

    cfg = {
        "execution": {
            "mode": "intraday_bar",  # <- Test unseres Hooks
            "n_buckets": 7,
            "ucurve": "U_DEFAULT",
            "fallback": {"adv_currency": 5_000_000.0, "vol_daily": 0.02},
        },
        "costs": {
            "fee_bps": 0.00005,  # 5 bps auf Notional
            "per_share_fee": 0.0,  # optional
            "sqrt_impact_coeff": 0.1,  # Impact-Modell an
        },
        "output": {"stats_dir": None},  # kein Log schreiben
    }

    out, log_path = annotate_trades_with_intraday_bar(trades, price_data=None, cfg=cfg)

    r = out.iloc[0]
    # Invarianten prüfen
    sums_ok = np.isclose(
        float(r["total_costs"]),
        float(r["fees"]) + float(r["slippage_cost"]) + float(r["impact_cost"]),
        atol=1e-6,
    )
    buckets_ok = int(r.get("n_child", 0)) == int(cfg["execution"]["n_buckets"])
    nonpos_ok = all(
        float(r[c]) <= 1e-9
        for c in ["fees", "slippage_cost", "impact_cost", "total_costs"]
    )
    mode_ok = (str(r.get("exec_mode")) == "INTRADAY_BAR") and (
        str(r.get("scheduler")) == "VWAP"
    )

    print("=== INTRADAY_BAR Smoke Test ===")
    print(
        f"fees={r['fees']:.2f}  slippage={r['slippage_cost']:.2f}  impact={r['impact_cost']:.2f}  total={r['total_costs']:.2f}"
    )
    print(
        f"n_child={int(r.get('n_child', -1))}  exec_mode={r.get('exec_mode')}  scheduler={r.get('scheduler')}"
    )
    print(
        f"checks: sums_ok={sums_ok}  buckets_ok={buckets_ok}  nonpos_ok={nonpos_ok}  mode_ok={mode_ok}"
    )

    if all([sums_ok, buckets_ok, nonpos_ok, mode_ok]):
        print("✅ ALL CHECKS PASSED")
        raise SystemExit(0)
    else:
        print("❌ CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
