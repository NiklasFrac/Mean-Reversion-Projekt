import numpy as np
import pandas as pd
from src.calibrate_slippage import fit_models, prepare_data


def test_calibrate_on_synthetic():
    # synthetic trades
    n = 200
    ts = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "trade_id": range(n),
            "timestamp": ts,
            "symbol": ["X"] * n,
            "size": [100] * n,
            "price": (100 + 0.01 * pd.Series(range(n))).values,
        }
    )
    # create fake executed price with sqrt-impact noise
    adv = 1e6
    notional = df["size"] * df["price"]
    vol = 0.02
    frac = notional / adv
    realized = 0.0002 + 0.05 * vol * (frac**0.5) + 0.0001 * np.random.randn(n)
    df["executed_price"] = df["price"] * (1 + realized)
    df["notional"] = notional
    df_prep = prepare_data(df, {"X": adv})
    params = fit_models(df_prep)
    assert "intercept" in params and "coef_sqrt" in params
