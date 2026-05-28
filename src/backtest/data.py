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


def load_sectors(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, usecols=["Symbol", "Sector"], dtype=str)
    symbols = df["Symbol"].map(_symbol)
    sectors = df["Sector"].fillna("").str.strip()
    out = pd.Series(sectors.to_numpy(), index=symbols, name="sector")
    out = out[out.ne("")]
    return out[~out.index.duplicated(keep="last")]


def _symbol(value) -> str:
    return str(value).strip().upper().replace(".", "-").replace("/", "-")
