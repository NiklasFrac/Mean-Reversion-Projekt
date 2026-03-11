# Backtest

## Overview

The backtest runner evaluates pair strategies with a **LOB** or **light** execution model and
conservative windowing. It consumes artifacts from Universe, Processing, and Analysis
and produces audit-ready outputs and reports.

## Installation

- Base tooling: `uv sync`
- Backtest runtime dependencies: `uv sync --extra backtest`
- Full four-stage local stack: `uv sync --extra analysis --extra backtest --extra processing`

## Inputs

Required artifacts:
- Price panel with MultiIndex columns `(symbol, field)` (OHLCV).
- Filtered pairs file from Analysis.

Optional artifacts:
- ADV map for liquidity/participation sizing.

Key config keys:
- `data.prices_path`: panel path (required for LOB liquidity).
- `data.pairs_path`: filtered pairs path.
- `data.adv_map_path`: optional ADV map.
- `data.input_mode`: `explicit` or `analysis_meta` (pins to Analysis run + freezes pairs).

## Run

```bash
uv run --extra backtest python -m backtest.runner_backtest --cfg runs/configs/config_backtest.yaml
```

Flags:
- `--out <dir>`: custom output directory.
- `--quick`: smaller windows and fewer pairs for dev speed.
- Default output directory without `--out`: `runs/results/performance/BT-<timestamp>/`.

## Strategy Selection

```yaml
strategy:
  name: "baseline"
  pair_z_window_as_volatility_window: false  # optional: reuse resolved per-pair z_window for signal.volatility_window
  params: {}              # optional kwargs if supported by the strategy
```

Optional entry overlay:

```yaml
markov_filter:
  enabled: false
  horizon_days: 10
  min_revert_prob: 0.55
  min_train_observations: 30
  min_state_observations: 5
  transition_smoothing: 0.0
  neutral_z: null   # default: signal.exit_z
  entry_z: null     # default: signal.entry_z
```

Behavior:
- The filter is fit only on the local train window of each run or walk-forward window.
- It uses the same spread and rolling Z-score as the baseline strategy.
- It can only block new entries.
- Exits, stops, cooldown, sizing, execution lag, and trade replay stay unchanged.
- If the train sample is too small or a state is unsupported, the strategy falls back to pass-through for that state.

## Windowing

Single conservative run:
- Provide `backtest.splits.analysis/train/test` with ordered, non-overlapping ranges.
- `backtest.execution_lag_bars` controls the central signal-to-execution lag for all strategies.

Walk-forward:
- Configure `backtest.walkforward.*` and `backtest.range.*`.
- `train_mode`: `expanding` or `rolling` (rolling keeps a fixed-length train window).
- `step_months` must be greater than or equal to `test_months`; overlapping test windows are rejected.
- Each window produces a diagnostic `WF-###` sub-run; the recombined global walk-forward output
  is the authoritative OOS result.
- New entries are limited to each local test window.
- Open trades carry automatically across intermediate windows.
- After local runs, raw trades are ledgered globally, resized statefully with MTM equity, and
  replayed over the full global OOS horizon.
- Only the final global OOS day is a hard cutoff; clipped trades are marked with
  `hard_exit` / `hard_exit_reason`.

## Execution Model

`execution.mode` can be `lob` or `light`.

Key components:
- `lob`: execution-authoritative LOB overlay with stress regimes, delayed/blocked entries, delayed/forced exits, and diagnostic slippage/impact.
- `light`: deterministic execution on trade prices with explicit fees and no LOB heuristics.
- Participation: `execution.max_participation` caps order size vs ADV before either execution overlay.
- `execution.override_pnl` is removed. In `lob` mode, PnL always follows executed prices and executed dates.
- `execution.lob.stress_model` is the first-class realism control for turbulent sessions.

Calibration:
- See `src/backtest/calibration/README.md` for method and artifacts.

## Borrow

Borrow costs are computed per trade:
- `borrow.accrual_mode` (entry notional vs mtm daily).
- `borrow.rates`, `borrow.rate_series_by_symbol`, `borrow.availability` (optional).

## Bayesian Optimization

BO is two-stage when the Markov overlay is enabled:
- per-pair `z_window` and `signal.max_hold_days` are derived from the
  cointegration residual half-life when `pair_prefilter.prefilter_active=true`.
- optional: `strategy.pair_z_window_as_volatility_window=true` also reuses the
  resolved per-pair `z_window` for `signal.volatility_window`.
- `spread_zscore.z_window` and `signal.max_hold_days` are fallback-only globals
  when pair half-life metadata is unavailable.
- Stage 1 always optimizes the signal parameters via root-level keys under
  `bo.*` such as `entry_z_range`, `exit_z_range`, `stop_z_range`,
  `init_points`, `n_iter`, `patience`.
- If `markov_filter.enabled=true`, Stage 2 optimizes
  `markov_filter.min_revert_prob` and `markov_filter.horizon_days` via
  `bo.min_revert_prob_range`, `bo.horizon_days_range`, plus optional
  Stage-2 budgets `bo.markov_init_points`, `bo.markov_n_iter`,
  `bo.markov_patience`.
- If the Stage-2 search ranges are omitted, BO falls back to the point values
  already configured under `markov_filter.*`.
- Markov BO is supported only with `bo.mode=realistic`. Using
  `markov_filter.enabled=true` together with `bo.mode=fast` raises an error.
- Legacy `bo.stage1` / `bo.stage2` configs are no longer supported.

## Risk Manager

Enable portfolio-level risk caps:
- `risk.max_open_positions`
- `risk.max_trade_pct`, `risk.max_gross_exposure`, `risk.max_net_exposure`
- optional concentration caps: `risk.max_per_name_pct`, `risk.max_positions_per_symbol`

## Reporting & Artifacts

Per run:
- `config_effective.json` (resolved config after BO overrides).
- `report/test_summary.json`, `report/test_equity.csv`, `report/test_trades.csv`.
- `report/train_selection_summary.json`, `report/train_cv_scores.*`, `report/train_refit_equity.*`.
- `report/test_tearsheet/` with the slim OOS tear sheet.

Walk-forward:
- `report/test_window_summary.csv` for per-window OOS diagnostics.
- In `reporting.mode=core`, `WF-###` sub-runs are not persisted.
- In `reporting.mode=debug`, per-window diagnostics are written under `debug/WF-###/`.

Reproducibility:
- `data.input_mode=analysis_meta` resolves the upstream Analysis metadata and freezes the exact pairs input into `<out_dir>/inputs/`.
- `inputs_provenance.json` records the resolved upstream files and hashes.
- `config_effective.json` captures the fully resolved config after BO overrides.
- For paper/audit work, prefer pinned upstream inputs from `runs/data/by_run/...` over mutable `runs/data/*`.

## Validation / Robustness

Optional checks:
- `pit_guard` (leakage checks)
- `overfit` (BO diagnostics)

## Quality Gates

```bash
uv run ruff check src/backtest src/tests/backtest
uv run ruff format --check src/backtest src/tests/backtest
uv run --extra backtest mypy --config-file mypy_backtest.ini
uv run --extra backtest pytest -q src/tests/backtest
```

Git hooks:
- `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`
- `uv run pre-commit run --all-files --hook-stage pre-push`
