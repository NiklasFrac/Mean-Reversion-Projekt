import numpy as np
import pandas as pd
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from universe.filters import apply_filters


def _mk_df(n=120):
    rng = np.random.default_rng(42)
    idx = [f"T{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "price": rng.uniform(0.5, 200.0, size=n),
            "market_cap": rng.uniform(1e6, 1e11, size=n),
            "volume": rng.uniform(1e3, 1e7, size=n),
            "float_pct": rng.uniform(0.0, 1.0, size=n),
            "dividend": rng.integers(0, 2, size=n).astype(bool),
            "is_etf": rng.integers(0, 2, size=n).astype(bool),
            "shares_out": rng.uniform(1e3, 1e12, size=n),
        },
        index=idx,
    )


@settings(
    max_examples=60,
    phases=[Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    derandomize=True,
    database=None,
)
@given(
    a=st.floats(min_value=0.0, max_value=100.0),
    b=st.floats(min_value=0.0, max_value=100.0),
)
def test_min_price_monotonicity(a, b):
    df = _mk_df(160)
    f1 = {"min_price": min(a, b)}
    f2 = {"min_price": max(a, b)}
    out1 = apply_filters(df, f1)
    out2 = apply_filters(df, f2)
    assert set(out2.index).issubset(set(out1.index))


@settings(
    max_examples=60,
    phases=[Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    derandomize=True,
    database=None,
)
@given(
    a=st.floats(min_value=1e6, max_value=1e11),
    b=st.floats(min_value=1e6, max_value=1e11),
)
def test_min_mcap_monotonicity(a, b):
    df = _mk_df(160)
    f1 = {"min_market_cap": min(a, b)}
    f2 = {"min_market_cap": max(a, b)}
    out1 = apply_filters(df, f1)
    out2 = apply_filters(df, f2)
    assert set(out2.index).issubset(set(out1.index))
