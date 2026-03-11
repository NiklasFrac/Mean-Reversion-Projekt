from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from backtest.config.execution_costs import resolve_execution_cost_spec
from backtest.simulators.performance import apply_costs, calculate_pair_daily_pnl
from backtest.utils import strategy as _strat_helpers


def _estimate_beta_ols(y: pd.Series, x: pd.Series) -> float | None:
    yy = pd.to_numeric(y, errors="coerce")
    xx = pd.to_numeric(x, errors="coerce")
    mask = yy.notna() & xx.notna()
    if int(mask.sum()) < 2:
        return None
    yv = yy.loc[mask].to_numpy(dtype=float, copy=False)
    xv = xx.loc[mask].to_numpy(dtype=float, copy=False)
    xm = float(np.mean(xv))
    ym = float(np.mean(yv))
    denom = float(np.sum((xv - xm) ** 2))
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    beta = float(np.sum((xv - xm) * (yv - ym)) / denom)
    return beta if np.isfinite(beta) else None


def _extract_cost_cfg(
    cfg: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    spec = resolve_execution_cost_spec(cfg)
    return spec.per_trade, spec.fee_bps, spec.per_share_fee, spec.min_fee, spec.max_fee


def _resolve_z_min_periods(cfg: Mapping[str, Any], *, z_window: int) -> int:
    zcfg = (
        cfg.get("spread_zscore", {})
        if isinstance(cfg.get("spread_zscore"), Mapping)
        else {}
    )
    z_window_eff = max(2, int(z_window))
    z_minp_raw = zcfg.get("z_min_periods", None)
    if z_minp_raw is None:
        return max(1, int(math.ceil(0.5 * float(z_window_eff))))
    try:
        return max(1, min(z_window_eff, int(z_minp_raw)))
    except Exception:
        return max(1, int(math.ceil(0.5 * float(z_window_eff))))


def _resolve_pair_runtime(
    yz: Mapping[str, Any],
    *,
    z_window_default: int,
    max_hold_default: int,
    cfg: Mapping[str, Any],
) -> tuple[int, int, int]:
    try:
        pair_z_window = int(yz.get("z_window", z_window_default))
    except Exception:
        pair_z_window = int(z_window_default)
    try:
        pair_max_hold = int(yz.get("max_hold_days", max_hold_default))
    except Exception:
        pair_max_hold = int(max_hold_default)
    pair_z_window = max(2, int(pair_z_window))
    pair_max_hold = int(pair_max_hold)
    pair_z_minp = _resolve_z_min_periods(cfg, z_window=pair_z_window)
    return pair_z_window, pair_max_hold, pair_z_minp


def _rolling_zscore(
    spread: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    w = int(max(2, window))
    s = pd.to_numeric(spread, errors="coerce").astype(float)
    base = s.shift(1)
    minp = max(
        1,
        int(min_periods) if min_periods is not None else int(math.ceil(0.5 * float(w))),
    )
    m = base.rolling(w, min_periods=minp).mean()
    sd = base.rolling(w, min_periods=minp).std(ddof=0).replace(0.0, np.nan)
    z = (s - m) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def _positions_from_z(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_days: int,
    cooldown_days: int,
) -> pd.Series:
    """
    Stateful position process s_t in {-1,0,+1}:
      - entry on *crossing* +/- entry_z
      - exit if |z| <= exit_z, stop if |z| >= stop_z, or max holding reached
      - cooldown enforces flat for cooldown_days after exit
    """
    z = pd.to_numeric(z, errors="coerce")
    idx = z.index
    out = pd.Series(0, index=idx, dtype="int8", name="pos")
    if z.empty:
        return out

    e = abs(float(entry_z))
    x = abs(float(exit_z))
    s = abs(float(stop_z))
    hmax = int(max_hold_days)
    unbounded_hold = hmax <= 0
    cool = int(max(0, cooldown_days))

    pos = 0
    held = 0
    cool_left = 0
    prev = float(z.iloc[0]) if math.isfinite(float(z.iloc[0])) else float("nan")

    for i in range(len(idx)):
        zt = float(z.iloc[i]) if math.isfinite(float(z.iloc[i])) else float("nan")

        if cool_left > 0:
            out.iat[i] = 0
            cool_left -= 1
            prev = zt
            continue

        if pos == 0:
            long_entry = (math.isfinite(zt) and zt <= -e and zt > -s) and (
                not math.isfinite(prev) or prev > -e
            )
            short_entry = (math.isfinite(zt) and zt >= e and zt < s) and (
                not math.isfinite(prev) or prev < e
            )
            if long_entry:
                pos = 1
                held = 0
            elif short_entry:
                pos = -1
                held = 0
            out.iat[i] = pos
            prev = zt
            continue

        # pos != 0
        held += 1
        exit_flag = False
        if math.isfinite(zt):
            if abs(zt) <= x:
                exit_flag = True
            if abs(zt) >= s:
                exit_flag = True
        if (not unbounded_hold) and held >= hmax:
            exit_flag = True

        # reversal entry crossing triggers exit (no same-day flip)
        if (
            pos == 1
            and (math.isfinite(zt) and zt >= e)
            and (not math.isfinite(prev) or prev < e)
        ):
            exit_flag = True
        if (
            pos == -1
            and (math.isfinite(zt) and zt <= -e)
            and (not math.isfinite(prev) or prev > -e)
        ):
            exit_flag = True

        if exit_flag:
            pos = 0
            held = 0
            cool_left = cool

        out.iat[i] = pos
        prev = zt

    return out


def _portfolio_pnl_equal_weight(
    pnl_by_pair: Mapping[str, pd.Series], calendar: pd.DatetimeIndex
) -> pd.Series:
    ser_list = []
    for pair in sorted((pnl_by_pair or {}).keys(), key=str):
        s = pnl_by_pair.get(pair)
        if isinstance(s, pd.Series) and not s.empty:
            ser_list.append(
                pd.to_numeric(s, errors="coerce")
                .reindex(calendar)
                .fillna(0.0)
                .astype(float)
            )
    if not ser_list:
        return pd.Series(dtype=float, name="pnl")
    df = pd.concat(ser_list, axis=1).fillna(0.0)
    pnl = df.mean(axis=1)
    pnl.name = "pnl"
    return pnl


def _precompute_spreads(
    per_pair: Mapping[str, Mapping[str, pd.Series]],
    *,
    cfg: Mapping[str, Any],
    z_window_for_beta: int,
) -> dict[str, pd.Series]:
    from backtest.utils.alpha import compute_spread_zscore

    out: dict[str, pd.Series] = {}
    for pair in sorted((per_pair or {}).keys(), key=str):
        yz = per_pair.get(pair)
        if not isinstance(yz, Mapping):
            continue
        y = yz["y"]
        x = yz["x"]
        try:
            spread, _, _ = compute_spread_zscore(
                y,
                x,
                cfg={
                    "z_window": int(z_window_for_beta),
                },
            )
            s = (
                pd.to_numeric(spread, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not s.empty:
                out[str(pair)] = s
        except Exception:
            continue
    return out


def _simulate_stage_pnl(
    *,
    spreads: Mapping[str, pd.Series],
    per_pair_prices: Mapping[str, Mapping[str, pd.Series]],
    z_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_days: int,
    cooldown_days: int,
    cfg: Mapping[str, Any],
    calendar: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    per_trade, fee_bps, per_share_fee, min_fee, max_fee = _extract_cost_cfg(cfg)

    out: dict[str, pd.Series] = {}
    for pair in sorted((spreads or {}).keys(), key=str):
        spread = spreads.get(pair)
        if not isinstance(spread, pd.Series):
            continue
        yz = per_pair_prices.get(pair)
        if not yz:
            continue
        y = yz["y"]
        x = yz["x"]
        pair_z_window, pair_max_hold, pair_z_minp = _resolve_pair_runtime(
            yz,
            z_window_default=z_window,
            max_hold_default=max_hold_days,
            cfg=cfg,
        )
        idx = spread.index.union(y.index).union(x.index).sort_values()
        if idx.empty:
            continue
        spread_al = pd.to_numeric(spread, errors="coerce").reindex(idx).ffill()
        y_al = pd.to_numeric(y, errors="coerce").reindex(idx).ffill()
        x_al = pd.to_numeric(x, errors="coerce").reindex(idx).ffill()
        valid = spread_al.notna() & y_al.notna() & x_al.notna()
        if not valid.any():
            continue
        spread_al = spread_al.loc[valid]
        y_al = y_al.loc[valid]
        x_al = x_al.loc[valid]

        z = _rolling_zscore(spread_al, int(pair_z_window), min_periods=pair_z_minp)
        pos = _positions_from_z(
            z,
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            max_hold_days=int(pair_max_hold),
            cooldown_days=int(cooldown_days),
        )

        pnl = calculate_pair_daily_pnl(pos, y_al, x_al).reindex(calendar).fillna(0.0)
        costs = (
            apply_costs(
                pos,
                y_al,
                x_al,
                per_trade_cost=0.0,
                slippage_pct=0.0,
            )
            .reindex(calendar)
            .fillna(0.0)
        )
        if per_trade or fee_bps or per_share_fee:
            from backtest.simulators.performance import apply_execution_costs

            fees = (
                apply_execution_costs(
                    pos,
                    y_al,
                    x_al,
                    per_trade=float(per_trade),
                    fee_bps=float(fee_bps),
                    per_share_fee=float(per_share_fee),
                    min_fee=float(min_fee),
                    max_fee=float(max_fee),
                )
                .reindex(calendar)
                .fillna(0.0)
            )
            costs = costs.add(fees, fill_value=0.0)
        out[str(pair)] = (pnl - costs).astype(float)
    return out


def _simulate_stage_pnl_refit(
    *,
    per_pair_prices: Mapping[str, Mapping[str, pd.Series]],
    train_dates: pd.DatetimeIndex,
    z_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_days: int,
    cooldown_days: int,
    cfg: Mapping[str, Any],
    calendar: pd.DatetimeIndex,
    eval_dates: pd.DatetimeIndex | None = None,
) -> dict[str, pd.Series]:
    per_trade, fee_bps, per_share_fee, min_fee, max_fee = _extract_cost_cfg(cfg)

    out: dict[str, pd.Series] = {}
    for pair in sorted((per_pair_prices or {}).keys(), key=str):
        yz = per_pair_prices.get(pair)
        if not yz:
            continue
        y = yz.get("y")
        x = yz.get("x")
        if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
            continue
        pair_z_window, pair_max_hold, pair_z_minp = _resolve_pair_runtime(
            yz,
            z_window_default=z_window,
            max_hold_default=max_hold_days,
            cfg=cfg,
        )
        idx = y.index.union(x.index).sort_values()
        if idx.empty:
            continue
        y_al = pd.to_numeric(y, errors="coerce").reindex(idx).ffill()
        x_al = pd.to_numeric(x, errors="coerce").reindex(idx).ffill()
        valid = y_al.notna() & x_al.notna()
        if not valid.any():
            continue
        y_al = y_al.loc[valid]
        x_al = x_al.loc[valid]

        train_idx = y_al.index.intersection(train_dates)
        if train_idx.empty:
            continue
        beta_hat = _estimate_beta_ols(y_al.loc[train_idx], x_al.loc[train_idx])
        if beta_hat is None:
            beta_hat = 1.0

        spread = (y_al - beta_hat * x_al).rename("spread")
        if eval_dates is not None:
            eval_idx = pd.DatetimeIndex(eval_dates)
            history_idx = _strat_helpers.prior_train_history(
                train_idx, eval_index=eval_idx
            )
            allowed_idx = history_idx.union(eval_idx).sort_values()
            z_eval = _strat_helpers.rolling_zscore_on_allowed_dates(
                spread,
                allowed_index=allowed_idx,
                window=int(pair_z_window),
                min_periods=pair_z_minp,
            ).reindex(eval_idx)
            z_eval = z_eval.dropna()
            if z_eval.empty:
                continue
            y_eval = y_al.reindex(z_eval.index).ffill()
            x_eval = x_al.reindex(z_eval.index).ffill()
            pos = _positions_from_z(
                z_eval,
                entry_z=float(entry_z),
                exit_z=float(exit_z),
                stop_z=float(stop_z),
                max_hold_days=int(pair_max_hold),
                cooldown_days=int(cooldown_days),
            )
            pnl = (
                calculate_pair_daily_pnl(pos, y_eval, x_eval)
                .reindex(calendar)
                .fillna(0.0)
            )
            costs = (
                apply_costs(
                    pos,
                    y_eval,
                    x_eval,
                    per_trade_cost=0.0,
                    slippage_pct=0.0,
                )
                .reindex(calendar)
                .fillna(0.0)
            )
            if per_trade or fee_bps or per_share_fee:
                from backtest.simulators.performance import apply_execution_costs

                fees = (
                    apply_execution_costs(
                        pos,
                        y_eval,
                        x_eval,
                        per_trade=float(per_trade),
                        fee_bps=float(fee_bps),
                        per_share_fee=float(per_share_fee),
                        min_fee=float(min_fee),
                        max_fee=float(max_fee),
                    )
                    .reindex(calendar)
                    .fillna(0.0)
                )
                costs = costs.add(fees, fill_value=0.0)
            out[str(pair)] = (pnl - costs).astype(float)
            continue

        z = _rolling_zscore(spread, int(pair_z_window), min_periods=pair_z_minp)
        pos = _positions_from_z(
            z,
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            max_hold_days=int(pair_max_hold),
            cooldown_days=int(cooldown_days),
        )

        pnl = calculate_pair_daily_pnl(pos, y_al, x_al).reindex(calendar).fillna(0.0)
        costs = (
            apply_costs(
                pos,
                y_al,
                x_al,
                per_trade_cost=0.0,
                slippage_pct=0.0,
            )
            .reindex(calendar)
            .fillna(0.0)
        )
        if per_trade or fee_bps or per_share_fee:
            from backtest.simulators.performance import apply_execution_costs

            fees = (
                apply_execution_costs(
                    pos,
                    y_al,
                    x_al,
                    per_trade=float(per_trade),
                    fee_bps=float(fee_bps),
                    per_share_fee=float(per_share_fee),
                    min_fee=float(min_fee),
                    max_fee=float(max_fee),
                )
                .reindex(calendar)
                .fillna(0.0)
            )
            costs = costs.add(fees, fill_value=0.0)
        out[str(pair)] = (pnl - costs).astype(float)
    return out
