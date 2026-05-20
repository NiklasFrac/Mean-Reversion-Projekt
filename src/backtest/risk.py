from __future__ import annotations

import pandas as pd

from backtest.config import RiskConfig


def cap_positions(
    positions: pd.DataFrame, zscores: pd.DataFrame, risk: RiskConfig
) -> pd.DataFrame:
    out = positions.copy()
    max_open = int(risk.max_open_pairs)
    for ts, row in out.iterrows():
        active = row[row != 0].index
        if len(active) > max_open:
            keep = (
                zscores.loc[ts, active]
                .abs()
                .sort_values(ascending=False)
                .head(max_open)
                .index
            )
            out.loc[ts, active.difference(keep)] = 0
    return out.astype("int8")
