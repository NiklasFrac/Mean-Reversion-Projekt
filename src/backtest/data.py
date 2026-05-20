from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = pd.Index(str(c) for c in df.columns)
    return df.apply(pd.to_numeric, errors="coerce")
