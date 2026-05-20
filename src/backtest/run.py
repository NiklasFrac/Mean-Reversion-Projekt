from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.config import Config, load_config
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
        strategy, markov, trials, best = optimize_params(
            prices,
            pairs,
            window,
            cfg.strategy,
            cfg.markov,
            cfg.risk,
            cfg.costs,
            cfg.bo,
            initial_capital=cfg.backtest.initial_capital,
            seed=cfg.seed + window.i,
        )
        if not trials.empty:
            trials.insert(0, "window", window.i)
            bo_rows.append(trials)
        if best:
            bo_best.append({"window": window.i, **best})
        betas = estimate_betas(prices, pairs, window)
        pairs = {pair: cols for pair, cols in pairs.items() if pair in betas}
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
            pair_rows.append(selected.assign(window=window.i, **window_dates))
        plans.append(WindowPlan(window, pairs, strategy, markov, betas))
        win_rows.append(
            {
                "window": window.i,
                **window_dates,
                "n_pairs": len(pairs),
                "pairs": ";".join(pairs),
            }
        )

    all_pairs = {pair: cols for plan in plans for pair, cols in plan.pairs.items()}
    sig = build_continuous_signals(prices, plans, cfg.risk)
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
