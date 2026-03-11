from __future__ import annotations

import pandas as pd


def price_tickers(
    prices_df: pd.DataFrame,
    *,
    require_data: bool,
    include_bare_columns: bool = False,
) -> set[str]:
    out: set[str] = set()
    if prices_df is None or prices_df.empty:
        return out

    cols = [str(c) for c in prices_df.columns]
    close_cols = [c for c in cols if c.endswith("_close")]
    for col in close_cols:
        sym = col[: -len("_close")]
        if require_data:
            series = pd.to_numeric(prices_df[col], errors="coerce")
            if series.notna().any():
                out.add(sym)
        else:
            out.add(sym)

    if include_bare_columns:
        for col in cols:
            if col.endswith(("_open", "_high", "_low", "_close")):
                continue
            if require_data:
                series = pd.to_numeric(prices_df[col], errors="coerce")
                if series.notna().any():
                    out.add(col)
            else:
                out.add(col)
    return out


def volume_tickers(volumes_df: pd.DataFrame, *, require_data: bool) -> set[str]:
    out: set[str] = set()
    if volumes_df is None or volumes_df.empty:
        return out
    for col in volumes_df.columns:
        name = str(col)
        if require_data:
            series = pd.to_numeric(volumes_df[col], errors="coerce")
            if series.notna().any():
                out.add(name)
        else:
            out.add(name)
    return out
