from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.config import MarkovConfig, RiskConfig, StrategyConfig
from backtest.markov import markov_gate
from backtest.walkforward import Window


@dataclass(frozen=True)
class StrategyOutput:
    positions: pd.DataFrame
    zscores: pd.DataFrame
    betas: dict[str, float] | pd.DataFrame


@dataclass(frozen=True)
class WindowPlan:
    window: Window
    pairs: dict[str, tuple[str, str]]
    strategy: StrategyConfig
    markov: MarkovConfig
    betas: dict[str, float]


def run_baseline(
    prices: pd.DataFrame,
    pairs: dict[str, tuple[str, str]],
    window: Window,
    strategy: StrategyConfig,
    markov: MarkovConfig,
) -> StrategyOutput:
    idx = prices.loc[window.train_start : window.test_end].index
    eval_idx = prices.loc[window.test_start : window.test_end].index
    pos, zcols, betas = {}, {}, {}

    for pair, (y_name, x_name) in pairs.items():
        if y_name not in prices or x_name not in prices:
            continue
        y = prices[y_name].reindex(idx).ffill()
        x = prices[x_name].reindex(idx).ffill()
        train = (
            pd.DataFrame({"y": y, "x": x})
            .loc[window.train_start : window.train_end]
            .dropna()
        )
        beta = estimate_beta(train["y"], train["x"]) if len(train) >= 2 else None
        if beta is None:
            continue
        spread = (y - beta * x).rename("spread")
        z = rolling_zscore(spread, strategy.z_window, strategy.z_min_periods)
        gate = markov_gate(
            z,
            pd.DatetimeIndex(train.index),
            pd.DatetimeIndex(eval_idx),
            markov,
            entry_z=strategy.entry_z,
            exit_z=strategy.exit_z,
        )
        z_eval = z.reindex(eval_idx)
        pos[pair] = positions_from_z(z_eval, strategy, gate)
        zcols[pair] = z_eval
        betas[pair] = float(beta)

    return StrategyOutput(
        positions=pd.DataFrame(pos, index=eval_idx).fillna(0).astype("int8"),
        zscores=pd.DataFrame(zcols, index=eval_idx),
        betas=betas,
    )


def build_continuous_signals(
    prices: pd.DataFrame, plans: list[WindowPlan], risk: RiskConfig
) -> StrategyOutput:
    idx = prices.loc[plans[0].window.test_start : plans[-1].window.test_end].index
    pairs = {pair: cols for plan in plans for pair, cols in plan.pairs.items()}
    positions = pd.DataFrame(0, index=idx, columns=list(pairs), dtype="int8")
    zscores = pd.DataFrame(np.nan, index=idx, columns=list(pairs))
    betas = pd.DataFrame(np.nan, index=idx, columns=list(pairs))
    zcache = _zcache(prices, plans, idx[-1])
    states, cooldowns = {}, {pair: 0 for pair in pairs}

    for plan_i, plan in enumerate(plans):
        test_idx = prices.loc[plan.window.test_start : plan.window.test_end].index
        train_idx = prices.loc[plan.window.train_start : plan.window.train_end].index
        gates = {
            pair: markov_gate(
                zcache[(plan_i, pair)],
                pd.DatetimeIndex(train_idx),
                pd.DatetimeIndex(test_idx),
                plan.markov,
                entry_z=plan.strategy.entry_z,
                exit_z=plan.strategy.exit_z,
            )
            for pair in plan.pairs
        }
        for ts in test_idx:
            final_day = ts == idx[-1]
            exited = _update_open(
                ts, final_day, states, cooldowns, zcache, positions, zscores
            )
            for pair, days in list(cooldowns.items()):
                if pair not in states and pair not in exited and days > 0:
                    cooldowns[pair] = days - 1
            if not final_day:
                _open_new(ts, plan_i, plan, gates, states, cooldowns, zcache, risk)
            for pair, state in states.items():
                if pd.isna(zscores.at[ts, pair]):
                    zscores.at[ts, pair] = zcache[(state["plan_i"], pair)].get(
                        ts, np.nan
                    )
                positions.at[ts, pair] = state["pos"]
                betas.at[ts, pair] = state["beta"]
    return StrategyOutput(positions, zscores, betas)


def estimate_betas(
    prices: pd.DataFrame, pairs: dict[str, tuple[str, str]], window: Window
) -> dict[str, float]:
    out = {}
    for pair, (y_name, x_name) in pairs.items():
        train = prices[[y_name, x_name]].loc[window.train_start : window.train_end]
        beta = estimate_beta(train[y_name], train[x_name])
        if beta is not None:
            out[pair] = beta
    return out


def estimate_beta(y: pd.Series, x: pd.Series) -> float | None:
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < 2 or df["x"].std(ddof=0) <= 0 or df["y"].std(ddof=0) <= 0:
        return None
    mat = np.column_stack([np.ones(len(df)), df["x"].to_numpy(float)])
    beta = float(np.linalg.lstsq(mat, df["y"].to_numpy(float), rcond=None)[0][1])
    return (
        beta
        if np.isfinite(beta)
        and beta > 0
        and not np.isclose(beta, 0.0, rtol=0.0, atol=np.finfo(float).eps)
        else None
    )


def rolling_zscore(spread: pd.Series, window: int, min_periods: int) -> pd.Series:
    base = pd.to_numeric(spread, errors="coerce").shift(1)
    mean = base.rolling(int(window), min_periods=int(min_periods)).mean()
    std = base.rolling(int(window), min_periods=int(min_periods)).std(ddof=0)
    return ((spread - mean) / std.replace(0.0, np.nan)).rename("z")


def _zcache(
    prices: pd.DataFrame, plans: list[WindowPlan], final_day: pd.Timestamp
) -> dict[tuple[int, str], pd.Series]:
    out = {}
    for i, plan in enumerate(plans):
        idx = prices.loc[plan.window.train_start : final_day].index
        for pair, (y_name, x_name) in plan.pairs.items():
            y = prices[y_name].reindex(idx).ffill()
            x = prices[x_name].reindex(idx).ffill()
            spread = y - plan.betas[pair] * x
            out[(i, pair)] = rolling_zscore(
                spread, plan.strategy.z_window, plan.strategy.z_min_periods
            )
    return out


def _update_open(
    ts, final_day, states, cooldowns, zcache, positions, zscores
) -> set[str]:
    exited = set()
    for pair, state in list(states.items()):
        z = zcache[(state["plan_i"], pair)].get(ts, np.nan)
        zscores.at[ts, pair] = z
        cfg = state["strategy"]
        should_exit = final_day
        if not should_exit:
            state["held"] += 1
            z_value = float(z)
            should_exit = state["held"] >= int(cfg.max_hold_days) or (
                np.isfinite(z_value)
                and (abs(z_value) <= abs(cfg.exit_z) or abs(z_value) >= abs(cfg.stop_z))
            )
        if should_exit:
            cooldowns[pair] = int(state["strategy"].cooldown_days)
            exited.add(pair)
            del states[pair]
        else:
            positions.at[ts, pair] = state["pos"]
    return exited


def _open_new(ts, plan_i, plan, gates, states, cooldowns, zcache, risk) -> None:
    free = int(risk.max_open_pairs) - len(states)
    if free <= 0:
        return
    candidates = []
    entry, stop = abs(plan.strategy.entry_z), abs(plan.strategy.stop_z)
    for pair in plan.pairs:
        if pair in states or cooldowns.get(pair, 0) > 0:
            continue
        z = zcache[(plan_i, pair)].get(ts, np.nan)
        prev = zcache[(plan_i, pair)].shift(1).get(ts, np.nan)
        if not bool(gates[pair].get(ts, True)):
            continue
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
    for _, pair, pos in sorted(candidates, reverse=True)[:free]:
        states[pair] = {
            "pos": pos,
            "held": 0,
            "plan_i": plan_i,
            "beta": plan.betas[pair],
            "strategy": plan.strategy,
        }


def positions_from_z(
    z: pd.Series, cfg: StrategyConfig, gate: pd.Series | None = None
) -> pd.Series:
    out = pd.Series(0, index=z.index, dtype="int8")
    pos, held, cool_left, prev = 0, 0, 0, np.nan
    entry, exit_z, stop = abs(cfg.entry_z), abs(cfg.exit_z), abs(cfg.stop_z)
    gate = gate.reindex(z.index) if gate is not None else None

    for ts, raw in z.items():
        zt = float(raw) if pd.notna(raw) else np.nan
        if cool_left > 0:
            cool_left -= 1
        elif pos == 0:
            ok = True if gate is None or pd.isna(gate.at[ts]) else bool(gate.at[ts])
            if (
                ok
                and np.isfinite(zt)
                and zt <= -entry
                and zt > -stop
                and (not np.isfinite(prev) or prev > -entry)
            ):
                pos, held = 1, 0
            elif (
                ok
                and np.isfinite(zt)
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
            if leave or flip or held >= cfg.max_hold_days:
                pos, held, cool_left = 0, 0, max(0, cfg.cooldown_days)
        out.at[ts] = pos
        prev = zt
    return out
