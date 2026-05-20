from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.config import MarkovConfig


def markov_gate(
    z: pd.Series,
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    cfg: MarkovConfig,
    *,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    out = pd.Series(True, index=test)
    if not cfg.enabled:
        return out
    train_z = z.reindex(train).dropna()
    if len(train_z) < cfg.min_train_observations:
        return out
    neutral = abs(cfg.neutral_z if cfg.neutral_z is not None else exit_z)
    entry = abs(cfg.entry_z if cfg.entry_z is not None else entry_z)
    probs = {side: _revert_prob(train_z, side, entry, neutral, cfg) for side in (-1, 1)}
    for ts, value in z.reindex(test).items():
        if pd.notna(value) and abs(value) >= entry:
            out.at[ts] = probs[int(np.sign(value))] >= cfg.min_revert_prob
    return out


def _revert_prob(
    z: pd.Series, side: int, entry: float, neutral: float, cfg: MarkovConfig
) -> float:
    hits = np.where(
        (np.sign(z.to_numpy(float)) == side) & (z.abs().to_numpy() >= entry)
    )[0]
    if len(hits) < cfg.min_state_observations:
        return 1.0
    ok = 0
    values = z.to_numpy(float)
    for i in hits:
        future = values[i + 1 : i + 1 + int(cfg.horizon_days)]
        ok += bool(len(future) and np.nanmin(np.abs(future)) <= neutral)
    smooth = float(cfg.transition_smoothing)
    return float((ok + smooth) / (len(hits) + 2 * smooth))
