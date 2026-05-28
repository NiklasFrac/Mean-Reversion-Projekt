# Mean-Reversion Pair Backtest

This repository contains a compact research pipeline for a pairs-trading backtest. It downloads price data, filters pair candidates, runs walk-forward tests, applies mean-reversion rules, and writes reproducible result files. The main intent is to keep the workflow readable and configuration-driven.

## Start

The primary workflow uses `uv`. If `uv` is not installed, use the existing
virtual environment or run the modules with any Python environment that has the
project dependencies installed.

```powershell
uv sync --extra backtest
uv run python -m download.runner_download --cfg runs/configs/config_download.yaml
uv run python -m backtest.run --config runs/configs/config_backtest.yaml

# fallback when using the local virtual environment
.\.venv\Scripts\python.exe -m download.runner_download --cfg runs/configs/config_download.yaml
.\.venv\Scripts\python.exe -m backtest.run --config runs/configs/config_backtest.yaml
```

## Download

`download.runner_download` reads the screener defined in `runs/configs/config_download.yaml`, normalizes tickers, filters unwanted securities, and downloads adjusted close data with `yfinance`.

The date range, batching, retry behavior, quality checks, and output paths are set in `runs/configs/config_download.yaml`.

## Pair Prefilter

`backtest.pair_selection` selects candidate pairs inside each training window. The filter uses correlation, the Engle-Granger test, a positive hedge ratio, half-life, and the Hurst value.

If a screener file is configured, pair selection loads sector data and only tests pairs inside the same sector group.

All thresholds and the pair limit are defined in `runs/configs/config_backtest.yaml`.

## Walkforward

`backtest.walkforward` builds separate train and test windows from the backtest configuration. Pairs are selected again for each window, and strategy parameters can be updated per window.

The window mode, lengths, step size, start, and end are set in `runs/configs/config_backtest.yaml`.

## Mean Reversion

The strategy trades the spread of a selected pair using a rolling z-score. Entry, exit, stop, cooldown, and maximum holding logic are read from the config.

Costs and position sizing are also configuration-driven.

## Optimization

If `bo.enabled` or `gridsearch.enabled` is active, `backtest.optimize` tunes selected strategy parameters on the training part of each walk-forward window. The search space and trial budget are defined in `runs/configs/config_backtest.yaml`.

If Bayesian Optimization is not available, the code falls back to random search using the same config.

Gridsearch uses the configured candidate lists directly.

## Output

Download outputs:

- `raw_close.csv`
- `filled_close.csv`
- `dropped_tickers.csv`

The output paths are defined in `runs/configs/config_download.yaml`.

Backtest outputs in the configured `output.dir`:

- `summary.json`
- `config_used.yaml`
- `daily.csv`
- `positions.csv`, `weights.csv`, `trades.csv`
- `windows.csv`, `selected_pairs.csv`
- `bo_trials.csv`, `bo_best.json`
- `wf_debug.json`
- `backtest.log`
- Plots: `equity.png`, `drawdown.png`
