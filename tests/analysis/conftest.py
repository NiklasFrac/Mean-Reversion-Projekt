from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(12345)


def _utc_dates(n):
    start = datetime(2010, 1, 1, tzinfo=timezone.utc)
    return pd.DatetimeIndex([start + timedelta(days=i) for i in range(n)], tz="UTC")


@pytest.fixture
def synthetic_prices_wide() -> pd.DataFrame:
    """
    Erzeugt synthetische Preise (wide, N=60 Ticker, T=1200), mit einigen hoch korrelierten Paaren.
    """
    T, N = 1200, 60
    dates = _utc_dates(T)
    # Basis-GBM
    mu = 0.0002
    sigma = 0.01
    rets = RNG.normal(mu, sigma, size=(T, N))
    # Injektion korrelierter Faktoren auf Paaren
    for j in range(0, 10, 2):  # 5 Paare
        factor = RNG.normal(mu, sigma, size=T)
        rets[:, j] = 0.7 * factor + 0.3 * rets[:, j]
        rets[:, j + 1] = 0.7 * factor + 0.3 * rets[:, j + 1]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    df = pd.DataFrame(prices, index=dates, columns=[f"T{c:02d}" for c in range(N)])
    return df


@pytest.fixture
def tmp_prices_file(tmp_path, synthetic_prices_wide) -> Path:
    p = tmp_path / "filled_data.pkl"
    synthetic_prices_wide.to_pickle(p)
    return p


@pytest.fixture
def make_prices_file(tmp_path: Path):
    """
    Speichert ein DataFrame als .pkl und gibt den Pfad zurück.
    """

    def _writer(df: pd.DataFrame, name: str = "prices.pkl") -> Path:
        p = tmp_path / name
        df.to_pickle(p)
        return p

    return _writer
