# Universe Runner Overview

This document outlines the purpose, flow, and data guarantees of the **Universe Runner** (`src/universe/runner_universe.py`) and its supporting modules. 

## Quickstart
Prereqs:
- Install `uv`: `python -m pip install --upgrade uv`
- Sync project environment (uses `pyproject.toml` + `uv.lock`): `uv sync`
- Provide a screener snapshot at `runs/data/nasdaq_screener_*.csv` (required for normal seed loading; see fallback note below for checkpoint-seed reuse).

Run:
- `uv run python -m universe.runner_universe --cfg runs/configs/config_universe.yaml`
- Alternative: `uv run python src/universe/runner_universe.py --cfg runs/configs/config_universe.yaml`
- Optional: add `--force` to rebuild (ignores caches/checkpoints for this run).

Outputs:
- Latest outputs (mutable working directory): `runs/data/*`
- Immutable per-run copy (recommended for papers/audits, when `runtime.persist_run_scoped_outputs=true`): `runs/data/by_run/<run_id>_<cfg_hash8>/outputs/*`
- Run manifest (provenance + config hash + git commit): `runs/data/universe_manifest.json`

Reset (fresh start):
- Execute (default): `uv run python -m universe.reset_universe --cfg runs/configs/config_universe.yaml`
- Dry-run: `uv run python -m universe.reset_universe --cfg runs/configs/config_universe.yaml --dry-run`

## Purpose
Build a vetted equity universe suitable for downstream research and backtests by:
- Loading a ticker seed from a screener snapshot (default: `runs/data/nasdaq_screener_*.csv`) with basic junk filters.
- Fetching core fundamentals (price, market cap, volume, float).
- Applying liquidity/quality filters to produce a final ticker list.
- Downloading historical prices/volumes.
- Emitting artifacts (CSVs, pickles, manifest, report) with reproducibility metadata.

## Key Modules and Responsibilities
- `runner_universe.py`: Orchestrates the end-to-end run (config load, build, downloads, artifacts, manifest, report).
- `builder.py`: Loads exchange tickers, manages checkpoint/cache reuse, fetches fundamentals in parallel, applies filters, enforces canary guards.
- `fundamentals.py`: Fetches fundamentals from yfinance with rate limiting, junk guards, and postfill for missing price/volume.
- `filters.py`: Applies numeric and symbol-level filters, tracking reason codes.
- `vendor.py`: Downloads price/volume panels from yfinance with chunking and retries.
- `outputs.py`: Writes CSVs and manifest; writes markdown report/run-scoped copies when `runtime.persist_run_scoped_outputs=true`.
- `storage.py`: Handles artifact paths and hashing.

## Execution Flow (runner)
1. **Config & Logging**: Load YAML (`config_universe.yaml` by default), initialize logging, and start Prometheus as configured.
2. **Run Metadata**: Compute config hash and generate run_id (or override via `runtime.run_id_override` / `UNIVERSE_RUN_ID`).
3. **Build Universe** (`build_universe`):
   - Load tickers from `universe.screener_glob` with drop_prefix/suffix/regex filters.
   - If a valid checkpoint seed exists and `runtime.reuse_exchange_seed=true`, the build can reuse that seed and still tries to reload the screener for provenance consistency.
   - Screener selection is controlled by `universe.screener_selection_mode`:
      - `error_if_ambiguous` (recommended): fail if the glob matches more than one file.
      - `latest_mtime`: pick newest by mtime.
   - Reuse fundamentals cache/checkpoint when valid.
   - Fetch fundamentals in parallel (yfinance), rate-limited, with checkpointing and fail-fast breaker.
   - Apply filters (price/mcap/volume/dollar ADV/float/dividend whitelist/blacklist), record reason codes.
   - Enforce canary (min_valid_tickers, max_nan_pct) and return filtered universe + monitoring stats.
4. **Prices/Volumes**:
   - Download price/volume history (batching, retries, optional group download).
   - Download knobs are parsed centrally via `universe.downloads.build_download_plan(...)` (used by runner + ADV path) to keep behavior consistent.
   - Runtime request timeout (`runtime.request_timeout`) is propagated to vendor history calls.
   - Default posture for yfinance: `auto_adjust=True` (vendor-adjusted closes only; manual corporate-action knobs removed).
   - Vendor-adjusted closes include splits/dividends for returns; ADV calculations use a separate unadjusted `Close * Volume` panel to stay split-neutral.
   - Historical ADV and price warmup are computed on the **first** `adv_window` trading days of available history (front-loaded; no trailing window/look-ahead).
   - An additional unadjusted close panel is downloaded and cached; ADV uses unadjusted `Close * Volume` to avoid split-driven distortions (adjusted panel remains available for returns).
   - `data.download_pause` now throttles normal download flow as well (between sequential chunks and between threaded chunk submissions), not only retry backoff.
   - Retry missing tickers sequentially and run an additional low-coverage rescue pass when needed.
   - If `allow_incomplete_history=false` abort on unrecoverable gaps, else drop unrecoverable symbols before final outputs are written.
   - Persist price/volume panels (hashed or canonical) and ADV cache/CSV (dollar ADV preferred when prices available).
5. **Outputs (tickers + manifest)**:
   - Write `tickers_universe.csv` (bare tickers) and `tickers_universe_ext.csv` (enriched) with ticker column guaranteed.
   - In `tickers_universe_ext.csv`, `price` is the effective screened price (`price_eff` when present), while `price_snapshot` preserves run-time snapshot quotes.
   - Write manifest with hashes, config path, git commit, run_id, counts, monitoring/extra payload.
6. **Report**:
   - When `runtime.persist_run_scoped_outputs=true`, write a detailed `report.md` into `runs/data/by_run/<run_id>_<cfg_hash8>/`.
7. **ADV provenance**: Warmup window start/end and the ADV fingerprint (window/period/interval/ticker hash) are recorded in the manifest for reproducibility.

## Data Quality Controls
- **Junk guards**: Prefix/suffix/contains filters at exchange load and fundamentals fetch.
- **Canary**: Enforces minimum valid tickers and maximum NaN share; core fields (`price`, `market_cap`, `volume`) are always checked when present.
- **Rate limiting & breaker**: Token buckets and circuit breaker protect vendor calls.
- **History completeness**: Configurable `allow_incomplete_history`; missing tickers either abort the run or are dropped before final outputs are written.
- **Artifacts hashing**: Optional hashed panel filenames to avoid collisions and ensure immutability.

## What Gets Written (defaults)
- `runs/data/tickers_universe.csv`, `runs/data/tickers_universe_ext.csv`
- `runs/data/raw_prices*.pkl`, `runs/data/raw_volume*.pkl`
- `runs/data/raw_prices_unadj*.pkl`, `runs/data/raw_volume_unadj*.pkl` (unadjusted panels for ADV/auditability)
- `runs/data/ticker_adv.pkl`, `runs/data/adv_map_usd.csv`
- `runs/data/adv_map_usd_filtered.csv` (ADV map aligned to the final universe, when configured)
- `runs/data/fundamentals_universe.parquet` (cache/store)
- `runs/data/universe_manifest.json`
- Run report under `runs/data/by_run/<run_id>_<cfg_hash8>/report.md` (when `runtime.persist_run_scoped_outputs=true`).

## Artifact Policy (latest vs. immutable)
- `runs/data/*` is **latest mutable**: reruns overwrite these files; treat them as a working directory, not a permanent record.
- When `runtime.persist_run_scoped_outputs=true`, each Universe run writes **immutable** copies under `runs/data/by_run/<run_id>_<cfg_hash8>/outputs/*` (and inputs under `.../inputs/*`), plus `.../report.md`.
- For paper/reviewer-grade auditability, cite `run_id` and use `runs/data/by_run/...` as the source of truth, not `runs/data/*`.
- Downstream (Processing/Analysis) is configured to prefer `runs/data/by_run/RUN-*/outputs/*` inputs when present, to avoid accidental distortions from overwritten/corrupted "latest" files.

## Reproducibility Contract
- **Standard mode (default)**: with live Yahoo/yfinance data (e.g., `download_end_date: "today"`), small numeric drift between reruns is expected; bit-identical outputs are not guaranteed.
- **Frozen mode (recommended for CI/papers)**: use a fixed screener snapshot and fixed `download_end_date` to reduce drift and improve rerun stability.
- **Acceptance criterion**: compare with numeric tolerances and downstream decision metrics, not raw file-hash equality.

## Config Highlights (config_universe.yaml)
- `runtime`: workers, progress_bar, request timeout/retries/backoff, checkpoint path, force_rebuild, canary thresholds, hashed-artifact toggle, `persist_run_scoped_outputs`.
- `universe`: output paths, ADV cache path, `screener_glob`, `screener_selection_mode`.
- `filters`: min price/mcap/volume, dollar ADV, float thresholds, dividend flag, whitelist/blacklist.
- `data`: price/volume download knobs (period/interval/batch/pause/retries/backoff, threads/group mode), `raw_prices_cache`/`volume_path`, `adv_window`, per-ticker warmup availability ratio `adv_min_valid_ratio` (default `0.7`, required valid warmup days = `ceil(adv_window * ratio)`), `allow_incomplete_history`, fundamentals cache toggles/TTL, split caches for warmup unadjusted prices (`raw_prices_unadj_warmup_cache`) vs. filtered unadjusted prices (`raw_prices_unadj_cache`), optional filtered ADV CSV (`adv_filtered_path`), ADV cache quality gate `adv_cache_min_coverage_ratio` (cache-level coverage only), and optional strict guard `strict_unadjusted_validation` (default `false`).
- `vendor`: rate limits and backoff for yfinance.

## Data Lineage (inputs -> artifacts -> consumers)
- Screener CSV (`runs/data/nasdaq_screener_*.csv`) or a valid checkpoint seed (`runtime.reuse_exchange_seed=true`) -> ticker seed (pre-filter) -> filters -> `runs/data/tickers_universe.csv` / `_ext.csv`.
- Fundamentals fetch (yfinance) -> cached store (`runs/data/fundamentals_universe.parquet`) -> filters -> `df_universe`.
- Price/volume (auto_adjust=True) -> normalized panels (`runs/data/raw_prices*.pkl`, `runs/data/raw_volume*.pkl`) -> manifests/run-scoped outputs -> downstream returns/backtests.
- Price/volume (auto_adjust=False) warmup -> cached warmup panel (`raw_prices_unadj_warmup_cache`) -> ADV map (`runs/data/adv_map_usd.csv`) -> filtered ADV map (`adv_filtered_path`) used by filters and recorded in manifest.
- Final unadjusted panel for filtered tickers (`raw_prices_unadj_cache`, auto-adjust False) is kept separate from the warmup cache to preserve auditability for ADV.

Notes:
- Time handling: prices/volumes are normalized to UTC and saved tz-naive by design. If downstream needs NY-localized indices, consumers must localize explicitly.
- Unused keys (e.g., pairs_path, adv_build/universe_meta/delist toggles) were removed from universe configs to avoid drift; only documented keys have effect.

## Known Trade-offs
- Fundamentals cache reuse is TTL-based; set `runtime.force_rebuild=true` or reduce `fundamentals_cache_ttl_days` if you need fresher coverage or lower NaN risk.
- `allow_incomplete_history=true` permits dropping symbols with unrecoverable history instead of aborting; set to false for stricter runs.
- External dependency on yfinance means vendor outages/data anomalies propagate; retries and breakers mitigate but cannot eliminate this risk.
- Logging is intentionally informative; for demos keep level at INFO to avoid noise (DEBUG emits chunk-level download details).
- Resilience under vendor failure is best-effort (retries/backoff/circuit-breakers); no chaos testing is in place.
- Saved volumes are raw/unadjusted, while adjusted closes are used for returns; ADV uses a separate unadjusted close panel to avoid mixing adjusted prices with raw volumes.
- Tests exist in `src/tests/universe/`; run `uv run pytest src/tests/universe` locally.
- Optional coverage: `uv run pytest src/tests/universe --cov=src/universe --cov-report=term-missing`.
- Ticker seed normally requires a local `runs/data/nasdaq_screener_*.csv`; if checkpoint seed reuse is enabled, runs may continue from cached seed depending on `runtime.allow_cached_seed_without_screener`.
- ADV warmup is anchored at the earliest available history across the panel (global start), not per-symbol. IPOs/late starters can end up with `NaN` warmup ADV and be filtered when `min_dollar_adv` is set; retaining very recent IPOs is out of scope for this universe.
- Optional diagnostics: keep both warmup (pre-filter) unadjusted prices and filtered unadjusted prices; preserve both unfiltered and filtered ADV CSVs for exploratory checks.
- With `data.allow_incomplete_history=true`, any tickers that still lack recoverable history after retries are dropped post-filter during price download. This makes the final universe sensitive to transient vendor gaps; for recruiter-facing runs prefer setting this to false so the run fails loudly instead of silently shrinking the universe.
- Exact bit-for-bit reruns are not guaranteed with yfinance/Yahoo data (vendor data can change). The per-run artifacts under `runs/data/by_run/...` (and the manifest) are the audit/reproducibility record.

## Limitations / Bias (explicit)
- **Survivorship bias**: The universe is seeded from a *current* Nasdaq screener snapshot (today's listed symbols). Delisted symbols during the sample are typically absent, so historical results can be biased upward (direction plausible; magnitude unknown).
- **Data-availability bias**: Free EOD data (Yahoo Finance via `yfinance`) can be missing, stale, or temporarily unavailable for some tickers. Symbols with insufficient history/coverage (e.g., warmup/liquidity metrics not computable) are excluded or can cause the run to fail, which couples the universe to vendor coverage.
- **Mixed "as-of" fields**: Liquidity/price filters use *front-loaded warmup medians* (e.g., `adv_window` first trading days), while fundamentals (e.g., market cap) are fetched from live Yahoo quote/fundamentals at run time. This is an intentional implementation trade-off and should be treated as a limitation in any research claims; if strict point-in-time fundamentals are required, a dated fundamentals snapshot must be used and recorded as an artifact.
- **Instrument-type filtering**: The Nasdaq screener CSV export used here does not include an authoritative "common equity" flag. Instrument-type exclusion is therefore rule-based (text/field heuristics), which can lead to small false inclusions/exclusions.

## Tests
Universe tests live in `src/tests/universe/` and cover: builder flow, filters, checkpoint/cache behavior, fundamentals outputs, runner edge cases, ADV dollar calculations, vendor downloads, and storage artifacts. Validate from the current branch/commit via `uv run pytest -q src/tests/universe --cov=src/universe --cov-report=term-missing`; resilience under vendor failure is not exhaustively tested.

## Developer Quality Checks
- Lint: `uv run ruff check src/universe src/tests/universe`
- Format (check): `uv run ruff format --check src/universe src/tests/universe`
- Format (write): `uv run ruff format src/universe src/tests/universe`
- Type-check profile used in CI for Universe: `uv run mypy --config-file mypy_universe.ini`
- Tests: `uv run pytest -q src/tests/universe`
- Install local git hooks (once): `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`
- Run all hooks manually (inkl. `pre-push`-Hooks): `uv run pre-commit run --all-files --hook-stage pre-push`
