# Processing Pipeline README

## Purpose
- Offline pipeline to load raw OHLCV/universe artefacts, clean/fill prices, and emit execution outputs plus diagnostics. Corporate actions are **not** applied here (assumes vendor-adjusted inputs).
- Runtime path is intentionally direct: `runner_processing -> pipeline.main` (no runtime monkeypatch bridge).

## Data Expectations
- Indices: timestamps convertible to `America/New_York`; naive indices are treated as `vendor_tz` (default `UTC`) then converted.
- Raw inputs: `raw_prices.*` (wide, columns symbols), optional `raw_volume.*`, optional panel (`raw_prices_panel.*`).
- Loader safety: if `tickers_universe.csv` exists under `data.dir`, the loader derives an expected symbol count and can ignore suspiciously tiny placeholder files for large universes (helps avoid accidentally using corrupted `runs/data/*` outputs).
- `vendor_tz`: override the assumed timezone for naive raw indices (default: `UTC`).
- `vendor_tz_policy`: resolve config vs. upstream universe timezone metadata (`extra.data_policy.raw_index_naive_tz`):
  - `config_wins`: keep config value and ignore upstream mismatch.
  - `upstream_override`: use upstream value when present (legacy behavior).

## Config Schema (key fields)
- `data.dir`: base input directory; `out_dir`: output directory (defaults to `runs/data`, unless legacy `filled_prices_path` is used). Relative processing paths are resolved from the project root, not the current shell working directory.
- `data.input_mode`: where to read raw inputs from (`run_latest`, `run_pinned`, `legacy_latest`; default is `legacy_latest` if omitted). Prefer `run_pinned` for thesis/paper so inputs are immutable and deterministic.
- `data.pinned_universe_outputs_dir`: required for `input_mode=run_pinned` (points to `runs/data/by_run/RUN-.../outputs`).
- `data.strict_inputs`: enforces hard checks for run-scoped modes. For `run_latest`/`run_pinned`, `tickers_universe.csv` under the resolved input dir must exist and be readable; prices/volume columns must match that symbol contract. If `data.allow_fallback_to_legacy=true`, fallback to `legacy_latest` is still allowed when run-scoped inputs are missing.
- `data.raw_prices_globs` / `raw_volume_globs`: glob patterns for inputs. Prefer unadjusted volume (e.g., `raw_volume_unadj*.pkl`) when using adjusted prices so ADV is split-neutral.
- `data_processing`: `max_gap_bars` (required; bars-only), `keep_pct_threshold`, `n_jobs`, `grid_mode`, `max_start_na`, `max_end_na`, `rth_only`, `parquet_compression`.
- `data_processing.outliers`: `enabled`, `zscore`, `window`, `use_log_returns`.
- `data_processing.return_caps`, `staleness` (default disabled for EOD to avoid over-dropping flat lines), `filling` (`causal_only`, `hard_drop`).
- `data_processing.vendor_guards.bad_rows.enabled`: set to `false` to fully disable manual correction file loading (no `vendor_bad_rows` file needed).
- `data_processing.calendars.symbol_calendar_csv_enabled`: set to `false` to disable loading `symbol_calendar_csv` (default `true`).
- `data_processing.vendor_tz`: timezone string applied to naive indices before converting to America/New_York (default `UTC`).
- `data_processing.vendor_tz_policy`: timezone conflict policy between `data_processing.vendor_tz` and upstream universe metadata (`config_wins`, `upstream_override`; default `upstream_override`).
- `data_processing.adv_min_volume_coverage`: minimum fraction of non-NA volume rows per symbol to keep it in ADV (default `0.2`).
- `data_processing.adv_mode`: ADV units for the exported `adv_map.pkl` (`shares` or `dollar`). `dollar` uses `price * volume` (USD/day) while `shares` uses volume only (shares/day).
- `data_processing.adv_stat`: rolling statistic (`mean` or `median`) used to compute ADV over `data_processing.adv_window`.
- `data_processing.adv_price_source`: price series used for `adv_mode=dollar`. Prefer `raw_unadjusted` (`raw_prices_unadj.pkl`) so `Close * Volume` is split-neutral; `exec` uses the processed execution close (can be split-distorted when multiplied by raw volume).
- `data_processing.quality_flags`: `enabled`, `path`, `format`, `invert_meaning` (optional, default `false`; failure to read/parse still aborts the run when enabled).
- `data_processing.panel_field_policies`: optional per-field overrides for caps/outliers/staleness and keep/max_gap/max_start/max_end; by default only `close`/`adj_close` apply caps/outliers/staleness and other fields pass through unless you opt in explicitly.
- `data_processing.pip_freeze`: set false to skip writing `runs/metadata/requirements.lock` during runs.
- `mlflow`: `enabled`, optional `tracking_uri`, `experiment_name`, `run_name`. If enabled and no `tracking_uri` is configured, processing defaults to `sqlite:///runs/metadata/mlflow.db`. Relative SQLite/file tracking URIs are resolved from the project root. When no experiment name is configured, processing uses `processing` and creates artifacts under `runs/mlruns`. On Windows, legacy `file://C:\\...` URIs are normalized before calling MLflow. Prefer SQLite/server tracking URIs over filesystem backends.

## Run Instructions
- CLI: `python -m processing.runner_processing --cfg runs/configs/config_processing.yaml`
- Library: `from processing.pipeline import main as processing_main; processing_main(Path("cfg.yaml"))`

## Outputs/Artefacts
- Filled execution prices (`filled_prices_exec.parquet`).
- Panel execution parquet, removed symbols pickle, and ADV map. ADV is skipped when no symbols pass volume-coverage filters; hard failures occur for true ADV build errors.
- Diagnostics JSON (`*.diag.json`) uses `schema_version=3` with centralized `quality` and `processing` blocks.
- Manifest JSON (`*_manifest.json`) mirrors the same processing summary in `extra.processing`.
- Immutable per-run snapshot: every run additionally writes a copy under `runs/data/by_run/PRC-<timestamp>-<cfg_hash>/outputs/processed/*` and snapshots the resolved config under `runs/data/by_run/.../inputs/`.
- `runs/data/processed/*` is **latest mutable** (overwritten on reruns); `runs/data/by_run/PRC-*/...` is the audit/repro source of truth.
- Latest outputs are synced in a batch after all run-scoped writes succeed; sync uses staged temp files with rollback to reduce partial-latest states on failures.
- Panel input: MultiIndex columns are expected as `(symbol, field)`; `(field, symbol)` layouts are auto-detected and normalized. Non-close fields are processed when present, with default caps/outliers/staleness disabled unless explicitly overridden via `panel_field_policies`.
- Post-processing reconciliation clamps OHLC so `close` lies inside `[low, high]`, `open` is clipped to the band, and `high>=low`; bounds are only widened, never shrunk toward close.

## What gets dropped (and why)
- Coverage gates: symbols with `non_na_pct < keep_pct_threshold`, excessive leading/trailing NAs (`max_start_na`, `max_end_na`), or longest gap > `max_gap_bars` (unless `hard_drop=false`), all computed after caps/outlier/staleness have been applied so cleaned data must satisfy thresholds.
- Non-positive prices: any symbol with `<=0` in the tradable window is dropped before filling.
- OHLC coherence: after field-level processing, symbols missing in any of `open/high/low/close` are removed to enforce consistent panels.
- Filling defaults: `causal_only=true` avoids using future bars; unfilled edge gaps are tolerated, interior gaps beyond `max_gap_bars` cause drops when `hard_drop=true`. Gaps that overlap non-tradable rows are still enforced on tradable sub-segments so max-gap rules cannot be bypassed.
- ADV sanity: symbols with insufficient volume coverage (below `adv_min_volume_coverage`) are skipped for ADV outputs.

## Tests
- `python -m pytest src/tests/processing`
- Key coverage: filling/outliers, loader fallbacks, runner CLI, manifest/atomic writes.
