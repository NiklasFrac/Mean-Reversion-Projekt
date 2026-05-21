# Mean-Reversion Pair Trading Backtest

A modular Python system for downloading equity close prices, selecting
statistically related pairs, generating mean-reversion signals, and running a
walk-forward backtest with optional Bayesian parameter optimization.

The pipeline is designed as a research backtest: prices are downloaded and
cleaned first, pairs are selected only on training windows, strategy parameters
can be optimized inside each window, and the final simulation runs with
continuous signals, transaction costs, risk caps, and reproducible report files.

**Stack:** pandas, NumPy, yfinance, statsmodels, Bayesian Optimization,
matplotlib, pytest, Ruff

---

## 1. Data Download

The downloader reads `runs/configs/config_download.yaml`, loads adjusted daily
close prices through `yfinance`, applies basic quality checks, and writes the
price matrix used by the backtest.

Run:

```powershell
uv run python -m download.runner_download --cfg runs/configs/config_download.yaml
```

### Configuration

| Section | Purpose |
| --- | --- |
| `input` | Screener path, symbol column, optional ETF/fund filter. |
| `download` | Date range, batch size, retry count, and request pause. |
| `quality` | Minimum data coverage, maximum gap length, minimum final ticker count. |
| `output` | Paths for raw prices, cleaned prices, and dropped ticker report. |

### Quality Rules

Each ticker is validated before it enters the backtest input:

1. Fully missing price series are dropped.
2. Non-positive prices are dropped.
3. Series below `quality.min_coverage` are dropped.
4. Leading or trailing gaps are dropped.
5. Internal gaps larger than `quality.max_gap` are dropped.
6. Remaining valid gaps are forward-filled.

### Outputs

| File | Description |
| --- | --- |
| `raw_close.csv` | Raw adjusted close-price matrix from Yahoo Finance. |
| `filled_close.csv` | Cleaned and forward-filled price matrix for the backtest. |
| `dropped_tickers.csv` | Tickers removed during filtering, download, or quality checks. |

Example `dropped_tickers.csv` structure:

```csv
ticker,reason
ABC,coverage_below_threshold
XYZ,gap_too_large
ETF1,etf_filtered
```

Example `filled_close.csv` structure:

```csv
date,AAPL,MSFT,NVDA,...
2020-03-02,298.81,160.62,59.98,...
2020-03-03,289.32,155.89,57.74,...
```

---

## 2. Pair Selection

Pairs are selected independently inside each training window. The selector uses
only positive, complete price series from the current training period.

Candidate pairs first need sufficient return correlation:

```text
corr(return_y, return_x) >= pair_selection.min_corr
```

For correlated candidates, the system tests both directions and keeps the
orientation with the lower Engle-Granger p-value. The spread diagnostics are
then filtered by the configured thresholds:

| Config field | Meaning |
| --- | --- |
| `pair_selection.max_eg_pvalue` | Maximum accepted cointegration p-value. |
| `pair_selection.min_half_life` / `max_half_life` | Accepted mean-reversion speed range. |
| `pair_selection.max_hurst` | Maximum accepted Hurst exponent. |
| `pair_selection.max_pairs` | Maximum selected pairs per window. |

Selected pairs and diagnostics are written to `selected_pairs.csv`.

---

## 3. Mean-Reversion Signal

For each selected pair, one asset is treated as `y` and the other as `x`.
The hedge ratio is estimated on the training window by ordinary least squares:

```text
y_t = alpha + beta * x_t + epsilon_t
```

The trading spread is:

```text
spread_t = y_t - beta * x_t
```

The spread is normalized into a rolling z-score. The rolling mean and standard
deviation are shifted by one step, so the current value is never used to
normalize itself:

```text
z_t = (spread_t - mean(spread_{t-window:t-1}))
      / std(spread_{t-window:t-1})
```

The window length and minimum observation count are configured as
`strategy.z_window` and `strategy.z_min_periods`.

### Entry and Exit Rules

Long spread position:

```text
z_t <= -strategy.entry_z
position = +1
```

Short spread position:

```text
z_t >= strategy.entry_z
position = -1
```

Positions are closed when one of the configured exit conditions is met:

```text
abs(z_t) <= strategy.exit_z
abs(z_t) >= strategy.stop_z
holding_days >= strategy.max_hold_days
```

After an exit, `strategy.cooldown_days` blocks immediate re-entry in the same
pair.

---

## 4. Markov Gate

The Markov gate is an optional entry filter. It does not create trades by
itself; it only decides whether an existing z-score entry signal is allowed.

For each side of the z-score distribution, the gate checks how often historical
entry states reverted back to a neutral zone within the configured horizon:

```text
hit = abs(z_t) >= entry_z and sign(z_t) == side

reverted = min(abs(z_{t+1:t+horizon})) <= neutral_z
```

The side-specific reversion probability is:

```text
revert_prob_side =
    (reverted_hits + transition_smoothing)
    / (total_hits + 2 * transition_smoothing)
```

A new entry is allowed only if:

```text
revert_prob_side >= markov.min_revert_prob
```

Main configuration fields:

| Config field | Meaning |
| --- | --- |
| `markov.enabled` | Enables or disables the gate. |
| `markov.horizon_days` | Historical look-ahead horizon for reversion checks. |
| `markov.min_revert_prob` | Minimum required reversion probability. |
| `markov.min_train_observations` | Minimum observations before the gate becomes active. |
| `markov.min_state_observations` | Minimum historical entry hits per side. |
| `markov.neutral_z` | Neutral zone used for reversion detection. |
| `markov.entry_z` | Optional Markov-specific entry threshold. |

If there is not enough training data, the gate stays open instead of blocking
all trades.

---

## 5. Risk Manager

The risk layer limits the number of simultaneously active pair positions.
The main cap is configured as:

```text
risk.max_open_pairs
```

If more pairs are active than allowed, the system keeps the strongest signals by
absolute z-score and sets the others to zero:

```text
keep = top_n(abs(z_t), n=risk.max_open_pairs)
```

Portfolio weights are applied later in the engine:

```text
weight_pair_t = position_pair_t * risk.max_pair_weight
```

`risk.max_drawdown` is available as a configuration-level risk assumption.

---

## 6. Walk-Forward Backtest

The backtest separates training and evaluation through walk-forward windows.
The setup is configured under `backtest.walkforward`.

For each window:

1. Build train and test date ranges.
2. Select pairs on the training window.
3. Optionally optimize strategy and Markov parameters.
4. Estimate hedge ratios on the training window.
5. Generate signals on the test window.
6. Carry open trades across window boundaries when needed.

Important fields:

| Config field | Meaning |
| --- | --- |
| `backtest.walkforward.enabled` | Enables walk-forward mode. |
| `backtest.walkforward.train_mode` | `rolling` or `expanding`. |
| `backtest.walkforward.train_months` | Training window length. |
| `backtest.walkforward.test_months` | Test window length. |
| `backtest.walkforward.step_months` | Step size between windows. |

Continuous walk-forward execution requires:

```text
step_months == test_months
```

This keeps the test windows contiguous.

---

## 7. Bayesian Optimization

Bayesian Optimization can tune selected strategy and Markov parameters before a
walk-forward test window. The search space is configured under `bo.ranges`.

For each candidate parameter set:

1. Split the training window into blocked cross-validation folds.
2. Run baseline signals on each fold.
3. Apply risk capping.
4. Run the engine.
5. Score the parameter set by median fold Sharpe.

```text
score(params) = median(sharpe_fold_1, ..., sharpe_fold_n)
```

Blocked CV is configured under `bo.cv`:

| Config field | Meaning |
| --- | --- |
| `bo.cv.n_blocks` | Number of chronological blocks. |
| `bo.cv.k_test_blocks` | Adjacent blocks used as the test fold. |
| `bo.cv.purge` | Removes observations around fold boundaries. |
| `bo.cv.embargo` | Removes observations after the test block from training. |

If Bayesian Optimization is unavailable, the system falls back to seeded random
search over the same parameter ranges.

Optimization outputs:

| File | Description |
| --- | --- |
| `bo_trials.csv` | Evaluated parameter sets and scores. |
| `bo_best.json` | Best parameter set per walk-forward window. |

Example `bo_trials.csv` structure:

```csv
window,score,entry_z,exit_z,stop_z,min_revert_prob,horizon_days
0,...,...,...,...,...,...
1,...,...,...,...,...,...
```

---

## 8. Backtest Engine

The engine converts pair positions into portfolio returns, applies transaction
costs, builds the equity curve, and records trades.

Pair returns are hedge-adjusted:

```text
hedge_return_t =
    (return_y_t - beta_t * return_x_t)
    / (1 + abs(beta_t))
```

The pair contribution uses the previous position weight:

```text
pair_pnl_t = weight_{t-1} * hedge_return_t
```

Turnover and costs are applied on weight changes:

```text
turnover_t = sum(abs(weight_t - weight_{t-1}))

return_t = sum(pair_pnl_t)
           - turnover_t * (costs.fee_bps + costs.slippage_bps) / 10000
```

Equity and drawdown are computed from the return series:

```text
equity_t = initial_capital * cumprod(1 + return_t)

drawdown_t = equity_t / rolling_max(equity_t) - 1
```

The engine writes all reports to `output.dir` from
`runs/configs/config_backtest.yaml`.

| File | Description |
| --- | --- |
| `summary.json` | Aggregate performance statistics. |
| `daily.csv` | Return, turnover, equity, and drawdown by date. |
| `equity.csv` | Equity curve. |
| `positions.csv` | Pair position matrix. |
| `trades.csv` | Position changes with date, pair, position, and z-score. |
| `windows.csv` | Walk-forward windows and selected pair counts. |
| `selected_pairs.csv` | Pair-selection diagnostics per window. |
| `config_used.yaml` | Full config snapshot used for the run. |
| `equity.png` | Equity curve plot. |
| `drawdown.png` | Drawdown plot. |

Example `summary.json` structure:

```json
{
  "total_return": "...",
  "sharpe": "...",
  "max_drawdown": "...",
  "trades": "..."
}
```

Example `trades.csv` structure:

```csv
date,pair,position,z
2021-01-04,AAA-BBB,1,-1.42
2021-01-11,AAA-BBB,0,-0.08
```
