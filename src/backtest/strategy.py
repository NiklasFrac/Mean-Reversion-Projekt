from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.config import StrategyConfig
from backtest.pair_selection import PairMeta
from backtest.walkforward import Window


@dataclass(frozen=True)
class StrategyOutput:
    positions: pd.DataFrame
    zscores: pd.DataFrame
    betas: dict[str, float] | pd.DataFrame
    exit_reasons: pd.DataFrame | None = None


@dataclass(frozen=True)
class WindowPlan:
    window: Window
    pairs: dict[str, PairMeta]
    strategy: StrategyConfig
    max_hold_days_by_pair: dict[str, int] = field(default_factory=dict)


def run_baseline(
    prices: pd.DataFrame,
    pairs: dict[str, PairMeta],
    window: Window,
    strategy: StrategyConfig,
    max_hold_days_by_pair: dict[str, int] | None = None,
) -> StrategyOutput:
    idx = prices.loc[window.train_start : window.test_end].index
    eval_idx = prices.loc[window.test_start : window.test_end].index
    pos, zcols, betas = {}, {}, {}
    max_hold_days_by_pair = max_hold_days_by_pair or {}

    for pair, meta in pairs.items():
        if meta.y not in prices or meta.x not in prices:
            continue
        spread = log_residual_spread(prices, idx, meta)
        z = rolling_zscore(spread, _z_window(meta, strategy))
        z_eval = z.reindex(eval_idx)
        pos[pair] = positions_from_z(
            z_eval,
            strategy,
            max_hold_days=max_hold_days_by_pair.get(pair),
        )
        zcols[pair] = z_eval
        betas[pair] = float(meta.beta)

    return StrategyOutput(
        positions=pd.DataFrame(pos, index=eval_idx).fillna(0).astype("int8"),
        zscores=pd.DataFrame(zcols, index=eval_idx),
        betas=betas,
    )


def build_continuous_signals(
    prices: pd.DataFrame, plans: list[WindowPlan]
) -> StrategyOutput:
    idx = prices.loc[plans[0].window.test_start : plans[-1].window.test_end].index
    pairs = {pair: cols for plan in plans for pair, cols in plan.pairs.items()}
    positions = pd.DataFrame(0, index=idx, columns=list(pairs), dtype="int8")
    zscores = pd.DataFrame(np.nan, index=idx, columns=list(pairs))
    betas = pd.DataFrame(np.nan, index=idx, columns=list(pairs))
    exit_reasons = pd.DataFrame(pd.NA, index=idx, columns=list(pairs), dtype="object")
    zcache = _zcache(prices, plans, idx[-1])
    states, cooldowns = {}, {pair: 0 for pair in pairs}

    for plan_i, plan in enumerate(plans):
        test_idx = prices.loc[plan.window.test_start : plan.window.test_end].index
        for ts in test_idx:
            final_day = ts == idx[-1]
            exited = _update_open(
                ts,
                final_day,
                states,
                cooldowns,
                zcache,
                positions,
                zscores,
                exit_reasons,
            )
            for pair, days in list(cooldowns.items()):
                if pair not in states and pair not in exited and days > 0:
                    cooldowns[pair] = days - 1
            if not final_day:
                _open_new(ts, plan_i, plan, states, cooldowns, zcache)
            for pair, state in states.items():
                if pd.isna(zscores.at[ts, pair]):
                    zscores.at[ts, pair] = zcache[(state["plan_i"], pair)].get(
                        ts, np.nan
                    )
                positions.at[ts, pair] = state["pos"]
                betas.at[ts, pair] = state["beta"]
    return StrategyOutput(positions, zscores, betas, exit_reasons)


def log_residual_spread(
    prices: pd.DataFrame, idx: pd.DatetimeIndex, meta: PairMeta
) -> pd.Series:
    y = prices[meta.y].reindex(idx).ffill()
    x = prices[meta.x].reindex(idx).ffill()
    spread = np.log(y) - (float(meta.alpha) + float(meta.beta) * np.log(x))
    return spread.rename("spread")


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    base = pd.to_numeric(spread, errors="coerce").shift(1)
    window = int(window)
    mean = base.rolling(window, min_periods=window).mean()
    std = base.rolling(window, min_periods=window).std(ddof=0)
    return ((spread - mean) / std.replace(0.0, np.nan)).rename("z")


def _zcache(
    prices: pd.DataFrame, plans: list[WindowPlan], final_day: pd.Timestamp
) -> dict[tuple[int, str], pd.Series]:
    out = {}
    for i, plan in enumerate(plans):
        idx = prices.loc[plan.window.train_start : final_day].index
        for pair, meta in plan.pairs.items():
            spread = log_residual_spread(prices, idx, meta)
            out[(i, pair)] = rolling_zscore(spread, _z_window(meta, plan.strategy))
    return out


def _z_window(meta: PairMeta, strategy: StrategyConfig) -> int:
    multiplier = float(strategy.z_window_multiplier)
    half_life = float(meta.half_life)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("strategy.z_window_multiplier must be positive")
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("pair half_life must be positive")
    window = multiplier * half_life
    if not math.isfinite(window):
        raise ValueError("strategy.z_window_multiplier * half_life must be finite")
    return max(1, int(math.ceil(window)))


def _update_open(
    ts, final_day, states, cooldowns, zcache, positions, zscores, exit_reasons
) -> set[str]:
    exited = set()
    for pair, state in list(states.items()):
        z = zcache[(state["plan_i"], pair)].get(ts, np.nan)
        zscores.at[ts, pair] = z
        cfg = state["strategy"]
        should_exit = False
        reason = None
        if final_day:
            should_exit = True
            reason = "forced_window_end"
        else:
            state["held"] += 1
            z_value = float(z)
            max_hold_days = state.get("max_hold_days")
            time_exit = (
                max_hold_days is not None and state["held"] >= int(max_hold_days)
            )
            stop_exit = np.isfinite(z_value) and abs(z_value) >= abs(cfg.stop_z)
            normal_exit = np.isfinite(z_value) and abs(z_value) <= abs(cfg.exit_z)
            if stop_exit:
                should_exit = True
                reason = "stop"
            elif normal_exit:
                should_exit = True
                reason = "normal"
            elif time_exit:
                should_exit = True
                reason = "timeout"
        if should_exit:
            cooldowns[pair] = int(state["strategy"].cooldown_days)
            exited.add(pair)
            exit_reasons.at[ts, pair] = reason
            del states[pair]
        else:
            positions.at[ts, pair] = state["pos"]
    return exited


def _open_new(ts, plan_i, plan, states, cooldowns, zcache) -> None:
    candidates = []
    entry, stop = abs(plan.strategy.entry_z), abs(plan.strategy.stop_z)
    for pair in plan.pairs:
        if pair in states or cooldowns.get(pair, 0) > 0:
            continue
        z = zcache[(plan_i, pair)].get(ts, np.nan)
        prev = zcache[(plan_i, pair)].shift(1).get(ts, np.nan)
        z_value = float(z)
        if not np.isfinite(z_value):
            continue
        prev_value = float(prev)
        pos = 0
        if (
            z_value <= -entry
            and z_value > -stop
            and (not np.isfinite(prev_value) or prev_value > -entry)
        ):
            pos = 1
        elif (
            z_value >= entry
            and z_value < stop
            and (not np.isfinite(prev_value) or prev_value < entry)
        ):
            pos = -1
        if pos:
            candidates.append((abs(z_value), pair, pos))
    for _, pair, pos in sorted(candidates, reverse=True):
        states[pair] = {
            "pos": pos,
            "held": 0,
            "plan_i": plan_i,
            "beta": plan.pairs[pair].beta,
            "strategy": plan.strategy,
            "max_hold_days": plan.max_hold_days_by_pair.get(pair),
        }


def positions_from_z(
    z: pd.Series,
    cfg: StrategyConfig,
    max_hold_days: int | None = None,
) -> pd.Series:
    out = pd.Series(0, index=z.index, dtype="int8")
    pos, held, cool_left, prev = 0, 0, 0, np.nan
    entry, exit_z, stop = abs(cfg.entry_z), abs(cfg.exit_z), abs(cfg.stop_z)
    max_hold_days = None if max_hold_days is None else int(max_hold_days)

    for ts, raw in z.items():
        zt = float(raw) if pd.notna(raw) else np.nan
        if cool_left > 0:
            cool_left -= 1
        elif pos == 0:
            if (
                np.isfinite(zt)
                and zt <= -entry
                and zt > -stop
                and (not np.isfinite(prev) or prev > -entry)
            ):
                pos, held = 1, 0
            elif (
                np.isfinite(zt)
                and zt >= entry
                and zt < stop
                and (not np.isfinite(prev) or prev < entry)
            ):
                pos, held = -1, 0
        else:
            held += 1
            leave = np.isfinite(zt) and (abs(zt) <= exit_z or abs(zt) >= stop)
            flip = (pos == 1 and np.isfinite(zt) and zt >= entry) or (
                pos == -1 and np.isfinite(zt) and zt <= -entry
            )
            time_exit = max_hold_days is not None and held >= max_hold_days
            if leave or flip or time_exit:
                pos, held, cool_left = 0, 0, max(0, cfg.cooldown_days)
        out.at[ts] = pos
        prev = zt
    return out
