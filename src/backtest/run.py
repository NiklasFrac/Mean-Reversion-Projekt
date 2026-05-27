from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import pandas as pd

from backtest.config import Config, StrategyConfig, load_config
from backtest.data import load_prices
from backtest.engine import BacktestResult, run_engine
from backtest.optimize import optimize_params
from backtest.pair_selection import select_pairs
from backtest.report import write_report
from backtest.strategy import WindowPlan, build_continuous_signals, estimate_betas
from backtest.walkforward import make_windows

logger = logging.getLogger("backtest")


def run_config(cfg: Config) -> BacktestResult:
    _setup_logging(cfg.output.dir, cfg.logging.level)
    prices = load_prices(cfg.data.prices_path)
    wf = cfg.backtest.walkforward
    if wf.enabled and wf.step_months != wf.test_months:
        raise ValueError("continuous walkforward requires step_months == test_months")
    windows = make_windows(pd.DatetimeIndex(prices.index), cfg.backtest)
    if not windows:
        raise ValueError("No walkforward windows produced")

    plans, win_rows, pair_rows, bo_rows, bo_best = [], [], [], [], []
    for window in windows:
        logger.info(
            "window %s train=%s..%s test=%s..%s",
            window.i,
            window.train_start.date(),
            window.train_end.date(),
            window.test_start.date(),
            window.test_end.date(),
        )
        pairs, selected = select_pairs(prices, window, cfg.pair_selection)
        if not pairs:
            raise ValueError(f"No pairs selected for window {window.i}")
        logger.info("window %s selected %d pair(s)", window.i, len(pairs))
        strategy = cfg.strategy
        max_hold_days_by_pair = _max_hold_days_by_pair(selected, strategy)
        strategy, markov, trials, best = optimize_params(
            prices,
            pairs,
            window,
            strategy,
            cfg.markov,
            cfg.risk,
            cfg.costs,
            cfg.bo,
            initial_capital=cfg.backtest.initial_capital,
            seed=cfg.seed + window.i,
            max_hold_days_by_pair=max_hold_days_by_pair,
        )
        if not trials.empty:
            trials.insert(0, "window", window.i)
            bo_rows.append(trials)
        if best:
            bo_best.append({"window": window.i, **best})
        betas = estimate_betas(prices, pairs, window)
        pairs = {pair: cols for pair, cols in pairs.items() if pair in betas}
        max_hold_days_by_pair = {
            pair: days for pair, days in max_hold_days_by_pair.items() if pair in pairs
        }
        selected = (
            selected[selected["pair"].isin(pairs)] if not selected.empty else selected
        )
        if not pairs:
            raise ValueError(f"No pairs with valid beta for window {window.i}")
        window_dates = {
            "train_start": str(window.train_start.date()),
            "train_end": str(window.train_end.date()),
            "test_start": str(window.test_start.date()),
            "test_end": str(window.test_end.date()),
        }
        if not selected.empty:
            if max_hold_days_by_pair:
                selected = selected.assign(
                    max_hold_days=selected["pair"].map(max_hold_days_by_pair)
                )
            pair_rows.append(selected.assign(window=window.i, **window_dates))
        plans.append(
            WindowPlan(window, pairs, strategy, markov, betas, max_hold_days_by_pair)
        )
        win_rows.append(
            {
                "window": window.i,
                **window_dates,
                "n_pairs": len(pairs),
                "pairs": ";".join(pairs),
                "entry_z": strategy.entry_z,
                "exit_z": strategy.exit_z,
                "stop_z": strategy.stop_z,
                "min_revert_prob": markov.min_revert_prob,
                "horizon_days": markov.horizon_days,
            }
        )

    all_pairs = {pair: cols for plan in plans for pair, cols in plan.pairs.items()}
    sig = build_continuous_signals(prices, plans)
    result = run_engine(
        prices,
        all_pairs,
        sig.betas,
        sig.positions,
        sig.zscores,
        initial_capital=cfg.backtest.initial_capital,
        costs=cfg.costs,
        risk=cfg.risk,
    )
    write_report(
        result,
        cfg.output.dir,
        cfg=cfg,
        windows=pd.DataFrame(win_rows),
        selected_pairs=pd.concat(pair_rows, ignore_index=True)
        if pair_rows
        else pd.DataFrame(),
        bo_trials=pd.concat(bo_rows, ignore_index=True) if bo_rows else pd.DataFrame(),
        bo_best=bo_best,
    )
    return result


def _max_hold_days_by_pair(
    selected: pd.DataFrame, strategy: StrategyConfig
) -> dict[str, int]:
    multiplier = strategy.max_hold_half_life_multiplier
    if selected.empty or "half_life" not in selected:
        return {}
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("strategy.max_hold_half_life_multiplier must be positive")
    out = {}
    for row in selected[["pair", "half_life"]].itertuples(index=False):
        half_life = float(row.half_life)
        if math.isfinite(half_life):
            out[str(row.pair)] = max(1, int(math.ceil(half_life * multiplier)))
    return out


def _setup_logging(out_dir: str | Path, level: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(Path(out_dir) / "backtest.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="runs/configs/config_backtest.yaml")
    args = parser.parse_args()
    run_config(load_config(args.config))


if __name__ == "__main__":
    main()
