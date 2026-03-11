# Calibration (LOB Liquidity Model) - Free Data Only

This repository implements a **LOB-only** execution model for US equities using **free data**.
Because free data does not provide a trade/quote tape (NBBO, prints, queue position, etc.),
the LOB is necessarily a **heuristic simulator** calibrated to **stylized facts** and
**plausibility constraints** rather than "ground-truth microstructure".

This document describes:
1) What the liquidity model does
2) What inputs it uses
3) How calibration avoids circularity / leakage
4) The calibration workflow and produced artifacts

---

## 1) What is calibrated?

The calibrated component is the **symbol-/date-aware liquidity heuristic** used by the LOB
execution annotator:

- `src/backtest/simulators/liquidity_model.py`
- wired in `src/backtest/simulators/lob.py`

For each `(symbol, date)` it derives:
- `min_spread_ticks` (effective spread proxy)
- per-level book depth (shares) via `level_sizes`
- book dynamics (`lam`, `cancel_prob`, `max_add`, `max_cancel`)

Additionally, an optional **fill uncertainty layer** can be enabled:

- `src/backtest/simulators/fill_model.py`

It models partial (or rejected) fills as a **heuristic** function of:
order size vs synthetic depth, participation vs ADV, and volatility. It is stochastic but
seeded/deterministic for reproducibility.

Important design choice: the model is intentionally *inactive* for "small" orders
(below `safe_depth_share` and `safe_participation`) to avoid artificially reducing
fills where a real market order would almost surely fill.

All of these are derived **only from free-data features** (OHLCV + rolling ADV + rolling vol).

---

## 2) Available inputs (free data)

The calibration uses the existing universe/processing outputs (no changes required):

- `runs/data/processed/filled_prices_panel_exec.parquet`
  - Daily panel with MultiIndex columns `(symbol, field)` and fields:
    `open`, `high`, `low`, `close`, `volume`
- `runs/data/adv_map_usd_filtered.csv`
  - Symbol -> $ADV proxy (and metadata)
- `runs/data/filtered_pairs.pkl`
  - Preselected pair candidates (analysis output)

Important: this is **daily** data. Therefore the LOB is a *synthetic intraday model* used
to produce realistic execution costs conditional on liquidity/volatility; it cannot
reconstruct true microstructure.

---

## 3) Avoiding circularity and leakage

### Circularity (what not to do)
Circular calibration would mean:
> "Use the simulator's own `trades.csv` as the ground-truth target to fit parameters."

That would just fit the model to itself.

### What we do instead
The calibration uses simulator outputs only as **observables** (measurements), while
the **targets** are external / indirect:

- broad stylized-fact ranges for costs vs. liquidity (ADV deciles)
- monotonicity constraints (impact increases with participation; costs decrease with ADV)
- plausibility bounds relative to daily range/volatility

No parameter is fit to maximize trading performance (Sharpe/PNL). This keeps alpha and
execution realism separate.

### Windowing / conservative mode
The backtest engine evaluates only `backtest.splits.test`. For calibration we run the
engine multiple times with mapped splits so that **train/val/test evaluations stay
disjoint and ordered**.

---

## 4) Calibration workflow

The calibration runner:

1) Loads data (`prices_panel`, `pairs`, `adv_map`)
2) Builds `pairs_data` and the strategy `portfolio` **once** (cached)
3) Runs a parameter search over `execution.lob.liq_model.*`
4) Scores each trial using plausibility targets (not PnL)
5) Selects the best parameters and validates on holdout

Runner entrypoint:

```bash
python -m backtest.runner_calibration --cfg runs/configs/config_backtest.yaml
```

### Capital / participation note
Execution realism depends on **participation** (order size relative to liquidity/ADV).
If your configured `backtest.initial_capital` is small (e.g. 100k) and you cap sizing via
`execution.max_participation`, most orders may fall into very low participation buckets,
making impact almost negligible (which can be realistic for small AUM).

For calibration/validation it can be helpful to *also* run at a larger capital level to
exercise a wider participation range:

```bash
python -m backtest.runner_calibration --cfg runs/configs/config_backtest.yaml --capital 1000000
```

### Outputs
The runner writes a calibration directory (by default under `runs/results/calibration/`):

- `trials.csv` - all tried parameters + scores
- `best_patch.yaml` - YAML patch for `execution.lob.liq_model`
- `report_train_*.csv`, `report_val_*.csv`, `report_test_*.csv`
  - decile tables / participation buckets / sanity summaries
- `score_breakdown.json` - why the best trial scored best

### How to interpret validation
With free data, validation is "realistic-looking" if:
- costs are within plausible bps ranges for liquid vs illiquid names
- impact increases with participation and volatility
- extreme costs are rare and concentrated in illiquid/volatile segments

If these properties hold *out-of-sample* (test split), the model is considered calibrated
for the purposes of the research.
