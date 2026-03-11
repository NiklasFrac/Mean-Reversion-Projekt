from __future__ import annotations

import pandas as pd


def equity_from_stats(stats: pd.DataFrame | None) -> pd.Series:
    eq = pd.Series(dtype=float, name="equity")
    if stats is None or stats.empty:
        return eq
    if "equity" in stats.columns:
        if "date" in stats.columns:
            idx = pd.to_datetime(stats["date"], errors="coerce")
            eq = pd.Series(
                pd.to_numeric(stats["equity"], errors="coerce").to_numpy(dtype=float),
                index=idx,
                name="equity",
            )
            eq = eq.loc[~eq.index.isna()]
            if isinstance(eq.index, pd.DatetimeIndex):
                eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        else:
            eq = pd.to_numeric(stats["equity"], errors="coerce")
            eq.name = "equity"
    return eq
