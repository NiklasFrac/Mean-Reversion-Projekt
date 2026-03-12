from __future__ import annotations

import datetime as dt
import os
import threading
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import pandas as pd

from universe.adv import (
    adv_fingerprint,
    load_or_compute_adv_map,
    load_price_volume_panels,
)
from universe.checkpoint import Checkpointer, norm_symbol
from universe.coercion import cfg_bool, cfg_float, cfg_int, is_truthy
from universe.config import validate_cfg as _validate_cfg
from universe.exchange_source import get_last_screener_meta, load_exchange_tickers
from universe.filters import _apply_filters_with_reasons
from universe.fundamentals import (
    INCOMPLETE_SAMPLE_LIMIT,
    CircuitBreaker,
    fetch_fundamentals_parallel,
    is_junk,
    set_junk_overrides,
)
from universe.monitoring import (
    logger,
    prom_set_total,
    stage_timer,
)
from universe.storage import (
    load_fundamentals_store,
    resolve_artifact_paths,
    save_fundamentals_store,
)
from universe.symbol_utils import normalize_symbols
from universe.utils import (
    _enforce_canary,
    _ensure_not_cancelled,
    _ensure_updated_at_column,
    _sha1,
)
from universe.vendor import VendorConfig, YFinanceVendor


# -------- TypedDicts for mypy --------
class Monitoring(TypedDict, total=False):
    failed: list[str]
    timeouts: int
    error: str
    cancelled: int
    history_missing: list[str]
    incomplete_core: list[dict[str, Any]]
    incomplete_core_total: int
    missing_field_counts: dict[str, int]
    postfill_total: int
    postfill_chunks: int
    postfill_completed_chunks: int
    postfill_duration_sec: float
    postfill_unresolved: list[str]
    postfill_unresolved_all: list[str]
    postfill_unresolved_total: int


class Extra(TypedDict, total=False):
    n_tickers_total: int
    n_fundamentals_ok: int
    n_failed: int
    n_failed_hard: int
    n_incomplete_core: int
    n_filtered: int
    universe_flow: list[dict[str, Any]]
    filter_flow: list[dict[str, Any]]
    cfg_path: str
    reason_codes: dict[str, int]
    artifacts: dict[str, str]
    tickers_all: list[str]
    canary: dict[str, float | int | str | None]
    run_id: str
    data_policy: dict[str, Any]
    screener_provenance: dict[str, Any]
    fundamentals_provenance: dict[str, Any]
    adv_provenance: dict[str, Any]


def _merge_fundamentals_frames(
    existing: pd.DataFrame, fresh: pd.DataFrame
) -> pd.DataFrame:
    if existing.empty and fresh.empty:
        merged = pd.DataFrame()
    elif existing.empty:
        merged = fresh.copy()
    elif fresh.empty:
        merged = existing.copy()
    else:
        existing_norm = existing.copy()
        fresh_norm = fresh.copy()
        existing_norm.index = existing_norm.index.map(lambda x: str(x).strip())
        fresh_norm.index = fresh_norm.index.map(lambda x: str(x).strip())
        existing_norm = existing_norm[~existing_norm.index.duplicated(keep="last")]
        fresh_norm = fresh_norm[~fresh_norm.index.duplicated(keep="last")]
        cols = existing_norm.columns.union(fresh_norm.columns)
        existing_norm = existing_norm.reindex(columns=cols)
        fresh_norm = fresh_norm.reindex(columns=cols)
        # Fresh non-null values should win, but fresh NaNs must not erase valid cache values.
        merged = fresh_norm.combine_first(existing_norm)
    if merged.index.name is None:
        merged.index.name = "ticker"
    merged.index = merged.index.map(lambda x: str(x).strip())
    if merged.index.duplicated().any():
        merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return _ensure_updated_at_column(merged)


def _normalize_seed_symbols(
    symbols: list[str],
    *,
    junk_filter: Any = is_junk,
) -> list[str]:
    keep = [s for s in symbols if s and not junk_filter(s)]
    return normalize_symbols(keep, unique=True)


def build_universe(
    config: dict[str, Any],
    cfg_path: Path,
    run_id: str,
    stop_event: threading.Event | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Monitoring, Extra]:
    cfg = _validate_cfg(config)
    _ensure_not_cancelled(stop_event)

    universe_cfg: dict[str, Any] = cfg.get("universe", {}) or {}
    filters_cfg: dict[str, Any] = cfg.get("filters", {}) or {}
    runtime_cfg: dict[str, Any] = cfg.get("runtime", {}) or {}
    data_cfg: dict[str, Any] = cfg.get("data", {}) or {}
    vendor_cfg: dict[str, Any] = cfg.get("vendor", {}) or {}
    artifact_paths = resolve_artifact_paths(
        universe_cfg=universe_cfg,
        data_cfg=data_cfg,
        runtime_cfg=runtime_cfg,
    )
    cfg_hash = _sha1(cfg_path)

    fundamentals_store_path = artifact_paths.fundamentals_out
    fundamentals_cache_enabled = bool(
        data_cfg.get("fundamentals_cache_enabled", bool(fundamentals_store_path))
    )
    fundamentals_cache_ttl_days = cfg_float(
        data_cfg,
        "fundamentals_cache_ttl_days",
        30.0,
        min_value=0.0,
        logger=logger,
        section_name="data",
    )
    fundamentals_store_mtime_utc: str | None = None
    checkpoint_ttl_seconds = (
        fundamentals_cache_ttl_days * 86400.0
        if fundamentals_cache_enabled and fundamentals_cache_ttl_days > 0
        else None
    )
    cached_fundamentals = pd.DataFrame()
    cached_funda_rows = 0
    cache_stale = False

    try:
        set_junk_overrides(
            suffixes=filters_cfg.get("drop_suffixes"),
            contains=filters_cfg.get("drop_contains"),
        )
    except Exception:
        logger.warning(
            "Could not apply junk overrides; using default values.",
            exc_info=False,
        )

    workers = cfg_int(
        runtime_cfg, "workers", 16, min_value=1, logger=logger, section_name="runtime"
    )
    progress_bar = cfg_bool(runtime_cfg, "progress_bar", True)
    timeout = cfg_int(
        runtime_cfg,
        "request_timeout",
        20,
        min_value=1,
        logger=logger,
        section_name="runtime",
    )
    req_retries = cfg_int(
        runtime_cfg,
        "request_retries",
        3,
        min_value=0,
        logger=logger,
        section_name="runtime",
    )
    backoff = cfg_float(
        runtime_cfg,
        "request_backoff",
        1.7,
        strictly_positive=True,
        logger=logger,
        section_name="runtime",
    )
    fundamentals_heartbeat_logging = cfg_bool(
        runtime_cfg,
        "fundamentals_heartbeat_logging",
        True,
    )
    postfill_mode = str(runtime_cfg.get("fundamentals_postfill_mode", "drop"))
    vendor_client: Any | None = None
    use_token_bucket = True
    fundamentals_rate_limit_per_sec = cfg_float(
        vendor_cfg,
        "rate_limit_per_sec",
        0.5,
        strictly_positive=True,
        logger=logger,
        section_name="vendor",
    )
    try:
        vendor_runtime_cfg = VendorConfig.from_mapping(vendor_cfg)
        fundamentals_rate_limit_per_sec = float(vendor_runtime_cfg.rate_limit_per_sec)
        vendor_client = YFinanceVendor(vendor_runtime_cfg)
        use_token_bucket = not bool(vendor_runtime_cfg.use_internal_rate_limiter)
    except Exception as e:
        logger.warning(
            "Invalid vendor config; falling back to direct yfinance calls: %s", e
        )

    # Canary configuration (fail fast)
    canary_cfg: dict[str, Any] = runtime_cfg.get("canary", {}) or {}
    min_valid_tickers = cfg_int(
        canary_cfg,
        "min_valid_tickers",
        1,
        min_value=1,
        logger=logger,
        section_name="runtime.canary",
    )
    max_nan_pct = cfg_float(
        canary_cfg,
        "max_nan_pct",
        1.01,
        min_value=0.0,
        logger=logger,
        section_name="runtime.canary",
    )  # >1.0 = disabled

    fail_fast_cfg: dict[str, Any] = runtime_cfg.get("fail_fast", {}) or {}
    fail_fast_enabled = cfg_bool(fail_fast_cfg, "enabled", False)
    breaker = (
        CircuitBreaker(
            max_consec_fail=cfg_int(
                fail_fast_cfg,
                "max_consecutive_failures",
                50,
                min_value=1,
                logger=logger,
                section_name="runtime.fail_fast",
            )
        )
        if fail_fast_enabled
        else None
    )
    if breaker is not None:
        breaker.reset()
    max_inflight_raw = runtime_cfg.get("max_inflight_requests")
    if max_inflight_raw in (None, ""):
        max_inflight_requests = int(workers)
    else:
        max_inflight_requests = cfg_int(
            runtime_cfg,
            "max_inflight_requests",
            workers,
            min_value=1,
            logger=logger,
            section_name="runtime",
        )
    reuse_exchange_seed = cfg_bool(runtime_cfg, "reuse_exchange_seed", True)
    allow_cached_seed_without_screener = cfg_bool(
        runtime_cfg, "allow_cached_seed_without_screener", False
    )

    env_force_raw = os.environ.get("UNIVERSE_FORCE", "")
    force_rebuild = is_truthy(runtime_cfg.get("force_rebuild")) or is_truthy(
        env_force_raw
    )
    if force_rebuild:
        logger.info("Force rebuild enabled: caches and checkpoints will be ignored.")

    if fundamentals_cache_enabled and fundamentals_store_path:
        if fundamentals_cache_ttl_days > 0 and fundamentals_store_path.exists():
            try:
                age_days = (
                    time.time() - fundamentals_store_path.stat().st_mtime
                ) / 86400.0
                if age_days > fundamentals_cache_ttl_days:
                    cache_stale = True
                    logger.info(
                        "Fundamentals cache stale (age=%.1fd > ttl=%.1fd); ignoring cache at %s",
                        age_days,
                        fundamentals_cache_ttl_days,
                        fundamentals_store_path,
                    )
            except Exception:
                cache_stale = True
        if force_rebuild:
            logger.info(
                "Force rebuild active; ignoring fundamentals cache at %s",
                fundamentals_store_path,
            )
        elif cache_stale:
            cached_fundamentals = pd.DataFrame()
        else:
            try:
                fundamentals_store_mtime_utc = dt.datetime.fromtimestamp(
                    fundamentals_store_path.stat().st_mtime, dt.UTC
                ).isoformat()
            except Exception:
                fundamentals_store_mtime_utc = None
            cached_fundamentals = load_fundamentals_store(fundamentals_store_path)
            cached_funda_rows = int(cached_fundamentals.shape[0])
    elif not fundamentals_cache_enabled and fundamentals_store_path:
        logger.info(
            "Fundamentals cache disabled via config; ignoring existing store at %s",
            fundamentals_store_path,
        )

    cp_val = artifact_paths.checkpoint_path
    checkpoint: Checkpointer | None = None
    if cp_val is not None:
        checkpoint = Checkpointer(Path(cp_val))
        checkpoint.load()
        if force_rebuild:
            checkpoint.reset()
    else:
        logger.info("Checkpoint disabled.")
    checkpoint_valid: set[str] = set()
    checkpoint_failed_valid: set[str] = set()
    removed_stale_entries = 0
    if checkpoint is not None and not force_rebuild:
        removed_stale_entries = checkpoint.purge_invalid(
            cfg_hash=cfg_hash,
            max_age=checkpoint_ttl_seconds,
        )
        if removed_stale_entries:
            logger.info(
                "Checkpoint cleaned: removed %d stale entries.", removed_stale_entries
            )
        checkpoint_valid = checkpoint.valid_symbols(
            cfg_hash=cfg_hash, max_age=checkpoint_ttl_seconds
        )
        checkpoint_failed_valid = checkpoint.failed_symbols(
            cfg_hash=cfg_hash,
            max_age=checkpoint_ttl_seconds,
        )

    cached_exchange_seed: list[str] | None = None
    cached_seed_provenance: dict[str, Any] | None = None
    if checkpoint is not None and reuse_exchange_seed and not force_rebuild:
        cached_seed_entry = checkpoint.symbol_seed_entry(
            cfg_hash=cfg_hash, max_age=checkpoint_ttl_seconds
        )
        if cached_seed_entry and cached_seed_entry.get("symbols"):
            cached_exchange_seed = list(cached_seed_entry.get("symbols") or [])
            prov = cached_seed_entry.get("provenance")
            if isinstance(prov, dict) and prov:
                cached_seed_provenance = dict(prov)
            logger.info(
                "Reusing cached exchange ticker seed from checkpoint (n=%d).",
                len(cached_exchange_seed),
            )

    tickers_all: list[str] = []
    if cached_exchange_seed:
        tickers_all = list(cached_exchange_seed)

        # Even when reusing the seed, always reload the screener CSV (fast) to record
        # provenance in the manifest/run-scoped artifacts, and to detect silent drift if the glob
        # now points to a different file.
        try:
            screener_tickers = load_exchange_tickers(
                filters_cfg=filters_cfg,
                universe_cfg=universe_cfg,
            )
            cached_norm = _normalize_seed_symbols(tickers_all)
            screener_norm = _normalize_seed_symbols(list(screener_tickers))
            if screener_norm and cached_norm and screener_norm != cached_norm:
                logger.warning(
                    "Checkpoint seed differs from current screener CSV; rebuilding seed from CSV "
                    "for provenance consistency (checkpoint=%d, csv=%d).",
                    len(cached_norm),
                    len(screener_norm),
                )
                tickers_all = list(screener_norm)
        except Exception as e:
            msg = (
                "Screener CSV reload for provenance failed while reusing checkpoint seed. "
                f"error={e}"
            )
            if allow_cached_seed_without_screener:
                logger.warning(
                    "%s; continuing with checkpoint seed only because "
                    "runtime.allow_cached_seed_without_screener=true.",
                    msg,
                )
            else:
                raise RuntimeError(
                    msg
                    + ". Set runtime.allow_cached_seed_without_screener=true to opt in to this "
                    "fallback."
                ) from e
    else:
        with stage_timer("exchange_load"):
            tickers_all = load_exchange_tickers(
                filters_cfg=filters_cfg,
                universe_cfg=universe_cfg,
            )

    if not tickers_all:
        raise RuntimeError(
            "No seed tickers loaded from screener; aborting universe build."
        )

    seed_meta: dict[str, Any] = {}
    try:
        seed_meta = dict(get_last_screener_meta())
    except Exception:
        seed_meta = {}
    if seed_meta:
        try:
            logger.info(
                "Seed flow (screener->symbols): rows_total=%d -> kept_instrument=%d (dropped=%d) "
                "-> after_symbol_prefilter=%d (dropped=%d) -> after_symbol_dedup=%d (dropped=%d).",
                int(seed_meta.get("rows_total", 0) or 0),
                int(seed_meta.get("rows_kept_instrument_type", 0) or 0),
                int(seed_meta.get("rows_dropped_instrument_type", 0) or 0),
                int(seed_meta.get("symbols_after_symbol_prefilter", 0) or 0),
                int(seed_meta.get("symbols_dropped_symbol_prefilter", 0) or 0),
                int(seed_meta.get("symbols_after_symbol_dedup", 0) or 0),
                int(seed_meta.get("symbols_dropped_symbol_dedup", 0) or 0),
            )
        except Exception:
            pass

    seed_before_junk = len(tickers_all)
    tickers_all = [t for t in tickers_all if not is_junk(t)]
    seed_dropped_junk = seed_before_junk - len(tickers_all)
    if seed_dropped_junk > 0:
        logger.info(
            "Seed junk-filter (prefix/suffix/contains): %d -> %d (dropped=%d).",
            seed_before_junk,
            len(tickers_all),
            seed_dropped_junk,
        )

    prom_set_total(len(tickers_all))
    # Keep completion log concise (per-run only); detailed screener logs already emitted.
    logger.debug(
        "Screener seed load complete: %d tickers.",
        len(tickers_all),
    )
    _ensure_not_cancelled(stop_event)
    seed_before_norm = len(tickers_all)
    tickers_all = normalize_symbols(tickers_all, unique=True)
    seed_after_norm = len(tickers_all)
    if seed_after_norm != seed_before_norm:
        logger.info(
            "Seed normalize+dedup: %d -> %d (dropped=%d).",
            seed_before_norm,
            seed_after_norm,
            seed_before_norm - seed_after_norm,
        )
    if checkpoint is not None:
        seed_prov: dict[str, Any] = {}
        try:
            seed_prov = dict(get_last_screener_meta())
        except Exception:
            seed_prov = {}
        if not seed_prov and cached_seed_provenance:
            seed_prov = dict(cached_seed_provenance)
        checkpoint.store_symbol_seed(
            tickers_all,
            cfg_hash=cfg_hash,
            provenance=seed_prov or None,
        )
    ticker_norms = set(tickers_all)
    if not cached_fundamentals.empty:
        cached_fundamentals.index = cached_fundamentals.index.map(norm_symbol)
        cached_fundamentals = cached_fundamentals[
            cached_fundamentals.index.isin(ticker_norms)
        ]
        logger.info(
            "Loaded fundamentals cache: %d rows (relevant=%d).",
            cached_funda_rows,
            int(cached_fundamentals.shape[0]),
        )
    if checkpoint is not None and not force_rebuild and not cached_fundamentals.empty:
        meta = checkpoint.entries()
        ts_map: dict[str, float] = {}
        for sym in cached_fundamentals.index.astype(str):
            ts_raw = meta.get(sym, {}).get("ts")
            if ts_raw is None:
                continue
            try:
                ts_map[sym] = float(ts_raw)
            except Exception:
                continue
        if ts_map:
            ts_cached = pd.Series(ts_map)
            cached_fundamentals.loc[ts_cached.index, "updated_at"] = ts_cached
    cached_symbols = set(cached_fundamentals.index)
    checkpoint_skip: set[str] | None = None
    if checkpoint is not None and not force_rebuild:
        checkpoint_entries = checkpoint.entries()
        has_checkpoint_entries = (
            bool(checkpoint_entries) or int(removed_stale_entries) > 0
        )
        if not cached_fundamentals.empty and has_checkpoint_entries:
            cached_before = int(cached_fundamentals.shape[0])
            cached_fundamentals = cached_fundamentals[
                cached_fundamentals.index.isin(checkpoint_valid)
            ]
            cached_symbols = set(cached_fundamentals.index)
            dropped_cached = cached_before - int(cached_fundamentals.shape[0])
            if dropped_cached:
                logger.info(
                    "Checkpoint TTL invalidated %d cached fundamentals rows; forcing re-fetch.",
                    dropped_cached,
                )
        elif not cached_fundamentals.empty and not has_checkpoint_entries:
            logger.info(
                "Checkpoint has no fundamentals entries; reusing %d cached fundamentals rows.",
                int(cached_fundamentals.shape[0]),
            )
        orphaned = (checkpoint_valid - cached_symbols) - checkpoint_failed_valid
        if orphaned:
            checkpoint.drop_many(orphaned)
            checkpoint_valid = checkpoint_valid - orphaned
            logger.info(
                "Checkpoint cleaned: removed %d entries without stored fundamentals.",
                len(orphaned),
            )
        if cached_symbols or checkpoint_failed_valid:
            checkpoint_skip = set(cached_symbols) | set(checkpoint_failed_valid)
            logger.info(
                "Checkpoint resume: skip fundamentals fetch for %d symbols (cached=%d, known_failed=%d).",
                len(checkpoint_skip),
                len(cached_symbols),
                len(checkpoint_failed_valid),
            )
        elif checkpoint_valid and not checkpoint_failed_valid:
            logger.info(
                "Checkpoint exists, but no fundamentals cache was found - performing a full refetch."
            )

    checkpoint_for_fetch = checkpoint
    if (not fundamentals_cache_enabled) or cached_fundamentals.empty:
        checkpoint_for_fetch = None
        checkpoint_skip = None
        if checkpoint is not None and not force_rebuild:
            logger.info(
                "Fundamentals cache disabled or empty - checkpoint will not be used for fundamentals fetching."
            )

    with stage_timer("fundamentals_fetch"):
        fundamentals_started_at_utc = dt.datetime.now(dt.UTC).isoformat()
        df_funda_new, monitoring_raw = fetch_fundamentals_parallel(
            tickers=tickers_all,
            workers=workers,
            show_progress=progress_bar,
            rate_limit_per_sec=fundamentals_rate_limit_per_sec,
            request_timeout=float(timeout),
            request_retries=req_retries,
            request_backoff=backoff,
            breaker=breaker,
            checkpoint=checkpoint_for_fetch,
            checkpoint_filter=checkpoint_skip,
            checkpoint_cfg_hash=cfg_hash,
            checkpoint_ttl=checkpoint_ttl_seconds,
            max_inflight=max_inflight_requests,
            stop_event=stop_event,
            ensure_not_cancelled=_ensure_not_cancelled,
            incomplete_sample_limit=INCOMPLETE_SAMPLE_LIMIT,
            junk_filter=is_junk,
            postfill_mode=postfill_mode,
            vendor=vendor_client,
            use_token_bucket=use_token_bucket,
            heartbeat_logging=fundamentals_heartbeat_logging,
        )
    monitoring: Monitoring = cast(Monitoring, monitoring_raw)
    fundamentals_finished_at_utc = dt.datetime.now(dt.UTC).isoformat()
    df_funda = _merge_fundamentals_frames(cached_fundamentals, df_funda_new)
    logger.info(
        "Fundamentals merged: cached=%d | fetched_now=%d | merged=%d",
        int(cached_fundamentals.shape[0]),
        int(df_funda_new.shape[0]),
        int(df_funda.shape[0]),
    )
    try:
        seed_final = int(len(tickers_all))
        funda_ok = int(df_funda.shape[0])
        missing = seed_final - funda_ok
        if missing:
            logger.info(
                "Fundamentals coverage: seed=%d | fundamentals_rows=%d | missing=%d (failed/filtered during fetch).",
                seed_final,
                funda_ok,
                int(missing),
            )
    except Exception:
        pass
    _ensure_not_cancelled(stop_event)

    checkpoint_retain_symbols: set[str] | None = None
    if checkpoint is not None:
        ts_series: pd.Series[Any] | None = None
        if "updated_at" in df_funda.columns:
            try:
                ts_series = pd.to_numeric(df_funda["updated_at"], errors="coerce")
            except Exception:
                ts_series = None
        now_ts = float(time.time())
        failed_for_checkpoint: set[str] = set()
        for sym in monitoring.get("failed", []) or []:
            ns = norm_symbol(str(sym))
            if ns:
                failed_for_checkpoint.add(ns)
        unresolved_all = monitoring.get("postfill_unresolved_all")
        unresolved_symbols = (
            unresolved_all if isinstance(unresolved_all, list) else None
        )
        if unresolved_symbols is None:
            unresolved_sample = monitoring.get("postfill_unresolved")
            unresolved_symbols = (
                unresolved_sample if isinstance(unresolved_sample, list) else []
            )
        for sym in unresolved_symbols:
            ns = norm_symbol(str(sym))
            if ns:
                failed_for_checkpoint.add(ns)
        done_symbols_norm = {
            norm_symbol(str(sym)) for sym in df_funda.index.astype(str)
        }
        failed_for_checkpoint = {
            s for s in failed_for_checkpoint if s and s not in done_symbols_norm
        }

        ts_map_done: dict[str, float] = {}
        if ts_series is not None:
            for sym in df_funda.index:
                try:
                    ts_raw = ts_series.get(sym)
                    if pd.notna(ts_raw):
                        ts_map_done[str(sym)] = float(ts_raw)
                except Exception:
                    continue
        checkpoint.mark_done_many(
            df_funda.index,
            cfg_hash=cfg_hash,
            timestamps=ts_map_done,
            default_timestamp=now_ts,
        )
        if failed_for_checkpoint:
            checkpoint.mark_failed_many(
                failed_for_checkpoint,
                cfg_hash=cfg_hash,
                default_timestamp=now_ts,
            )
            logger.info(
                "Checkpoint marked %d unresolved fundamentals symbols as failed for resume skip.",
                len(failed_for_checkpoint),
            )
        checkpoint_retain_symbols = (
            set(df_funda.index.astype(str))
            | set(failed_for_checkpoint)
            | set(checkpoint_failed_valid)
        )
        removed = checkpoint.retain_only(checkpoint_retain_symbols)
        if removed:
            logger.info(
                "Checkpoint cleaned after fetch: removed %d stale entries.", removed
            )

    def _log_df_stats(df: pd.DataFrame, name: str) -> None:
        n = int(df.shape[0])
        if n == 0:
            logger.info("%s: empty", name)
            return
        cols = [
            "price",
            "market_cap",
            "volume",
            "float_pct",
            "dollar_adv",
            "dividend",
            "is_etf",
        ]
        have = [c for c in cols if c in df.columns]
        nan_info = {c: float(df[c].isna().mean()) if df[c].size else 1.0 for c in have}
        logger.info(
            "%s: rows=%d | NaN%%: %s",
            name,
            n,
            ", ".join(f"{k}:{v:.2%}" for k, v in nan_info.items()),
        )

    _log_df_stats(df_funda, "Fundamentals/raw")

    if fundamentals_cache_enabled and fundamentals_store_path:
        try:
            fundamentals_store_path = save_fundamentals_store(
                df_funda, fundamentals_store_path
            )
            if checkpoint is not None and checkpoint_retain_symbols is not None:
                checkpoint.retain_only(checkpoint_retain_symbols)
        except Exception as e:
            logger.warning(
                "Failed to write fundamentals output (%s): %s", fundamentals_store_path, e
            )

    # --- Historical ADV computation ---
    warmup_window = cfg_int(
        data_cfg,
        "adv_window",
        30,
        min_value=1,
        logger=logger,
        section_name="data",
    )
    adv_seed_tickers = [norm_symbol(str(t)) for t in df_funda.index.tolist()]
    if not adv_seed_tickers:
        adv_seed_tickers = list(tickers_all)
    # Load price/volume panels once for ADV and price warmup metrics (front-loaded window, unadjusted closes for split-neutral ADV).
    with stage_timer("price_volume_load_for_adv"):
        prices_panel, volumes_panel = load_price_volume_panels(
            adv_seed_tickers,
            data_cfg,
            stop_event=stop_event,
            auto_adjust=False,
            request_timeout=float(timeout),
            prices_cache_key="raw_prices_unadj_warmup_cache",
        )
    # The warmup window is anchored at the start of available history (first adv_window trading days).
    first_window_end_ts = None
    first_window_start_ts = None
    if not prices_panel.empty:
        idx = pd.to_datetime(prices_panel.index, errors="coerce").normalize()
        idx = idx[idx.notna()]
        trading_days = idx.drop_duplicates().sort_values()
        if trading_days.size:
            first_window_start_ts = trading_days[0]
            idx_end = min(warmup_window - 1, len(trading_days) - 1)
            first_window_end_ts = trading_days[idx_end]
    if first_window_end_ts is None:
        logger.info("Warmup window fallback: using earliest available prices.")

    adv_meta: dict[str, Any] = {}
    with stage_timer("adv_compute"):
        adv_df, adv_meta = load_or_compute_adv_map(
            adv_seed_tickers,
            cfg.raw,
            prices_panel,
            volumes_panel,
            warmup_end=first_window_end_ts,
            return_meta=True,
        )

    adv_cols = [
        "dollar_adv_hist",
        "price_warmup_med",
        "volume_warmup_avg",
        "adv_window",
        "adv_asof",
    ]
    if not adv_df.empty:
        adv_join = adv_df.copy()
        for col in adv_cols:
            if col not in adv_join.columns:
                adv_join[col] = float("nan")
        df_funda = df_funda.join(adv_join[adv_cols], how="left")
    else:
        logger.warning(
            "No ADV warmup values available; falling back to snapshot price/volume where possible."
        )
        if "dollar_adv_hist" not in df_funda.columns:
            df_funda["dollar_adv_hist"] = float("nan")
        if "price_warmup_med" not in df_funda.columns:
            df_funda["price_warmup_med"] = float("nan")
        if "adv_window" not in df_funda.columns:
            df_funda["adv_window"] = float(warmup_window)
        if "adv_asof" not in df_funda.columns:
            df_funda["adv_asof"] = pd.NaT
        if "volume_warmup_avg" not in df_funda.columns:
            df_funda["volume_warmup_avg"] = float("nan")

    logger.info(
        "ADV warmup: %d of %d tickers with historical dollar_adv_hist and price_warmup_med "
        "(window=%d, start=%s, end=%s).",
        int(df_funda["dollar_adv_hist"].notna().sum()),
        int(df_funda.shape[0]),
        warmup_window,
        first_window_start_ts,
        first_window_end_ts,
    )

    filter_flow: list[dict[str, Any]] = []
    with stage_timer("filter_apply"):
        try:
            before = int(df_funda.shape[0])
            df_filtered, reasons = _apply_filters_with_reasons(
                df_funda, filters_cfg, audit=filter_flow
            )
            after = int(df_filtered.shape[0])
            logger.info("Filter angewendet: %d -> %d", before, after)
            if filter_flow:
                logger.info(
                    "Filter flow (exclusive, first-fail order; whitelist may re-add tickers as +added):"
                )
                for row in filter_flow:
                    removed = int(row.get("removed", 0) or 0)
                    added = int(row.get("added", 0) or 0)
                    if removed == 0 and added == 0:
                        continue
                    logger.info(
                        "  - %s: %d -> %d (removed=%d, added=%d)%s",
                        str(row.get("step") or row.get("code") or "?"),
                        int(row.get("n_before", 0) or 0),
                        int(row.get("n_after", 0) or 0),
                        removed,
                        added,
                        f" [{row.get('code')}]" if row.get("code") else "",
                    )
            # Canary is enforced after this try block (do not swallow exceptions).
            if reasons:
                top = ", ".join(
                    f"{k}:{v}"
                    for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:8]
                )
                logger.info("Reason-Codes (Top): %s", top)
            if before > 0 and after == 0:
                logger.warning(
                    "0 tickers remain after filtering. Filters may be too strict or the data may be incomplete."
                )
            elif before >= 50 and after <= 5:
                logger.warning(
                    "Only %d of %d tickers remain after filtering. Filters may be restrictive or many NaNs may be present.",
                    after,
                    before,
                )
        except Exception as e:
            logger.error("Filter application failed: %s", e, exc_info=True)
            # Fail fast to avoid emitting an unfiltered universe when filters misfire.
            raise RuntimeError("Filter application failed") from e

    _log_df_stats(df_filtered, "Fundamentals/filtered")

    # Check final consistency (including the float_pct heuristic) outside the try block:
    can_stats = _enforce_canary(
        df_filtered,
        min_valid_tickers=min_valid_tickers,
        max_nan_pct=max_nan_pct,
    )

    hard_failures = len(monitoring.get("failed", []) or [])
    incomplete_core_total = int(monitoring.get("incomplete_core_total", 0))

    # Paper-friendly, internally consistent flow summary (additive metadata only).
    universe_flow: list[dict[str, Any]] = []
    try:
        if seed_meta:
            rows_total = int(seed_meta.get("rows_total", 0) or 0)
            kept_instr = int(seed_meta.get("rows_kept_instrument_type", 0) or 0)
            drop_instr = int(seed_meta.get("rows_dropped_instrument_type", 0) or 0)
            if rows_total:
                universe_flow.append(
                    {
                        "step": "Screener CSV (rows_total)",
                        "n_before": rows_total,
                        "n_after": rows_total,
                        "removed": 0,
                        "added": 0,
                    }
                )
            universe_flow.append(
                {
                    "step": "Instrument-type filter (screener)",
                    "n_before": rows_total,
                    "n_after": kept_instr,
                    "removed": drop_instr,
                    "added": 0,
                }
            )
            universe_flow.append(
                {
                    "step": "Symbol prefilter (screener)",
                    "n_before": int(
                        seed_meta.get("symbols_before_symbol_prefilter", kept_instr)
                        or 0
                    ),
                    "n_after": int(
                        seed_meta.get("symbols_after_symbol_prefilter", 0) or 0
                    ),
                    "removed": int(
                        seed_meta.get("symbols_dropped_symbol_prefilter", 0) or 0
                    ),
                    "added": 0,
                }
            )
            universe_flow.append(
                {
                    "step": "Symbol dedup (screener)",
                    "n_before": int(
                        seed_meta.get("symbols_before_symbol_dedup", 0) or 0
                    ),
                    "n_after": int(seed_meta.get("symbols_after_symbol_dedup", 0) or 0),
                    "removed": int(
                        seed_meta.get("symbols_dropped_symbol_dedup", 0) or 0
                    ),
                    "added": 0,
                }
            )
        universe_flow.append(
            {
                "step": "Seed junk-filter",
                "n_before": int(seed_before_junk),
                "n_after": int(seed_before_junk - seed_dropped_junk),
                "removed": int(seed_dropped_junk),
                "added": 0,
            }
        )
        universe_flow.append(
            {
                "step": "Seed normalize+dedup",
                "n_before": int(seed_before_norm),
                "n_after": int(seed_after_norm),
                "removed": int(seed_before_norm - seed_after_norm),
                "added": 0,
            }
        )

        n_seed_final = int(seed_after_norm)
        n_funda = int(df_funda.shape[0])
        universe_flow.append(
            {
                "step": "Fundamentals available",
                "n_before": n_seed_final,
                "n_after": n_funda,
                "removed": int(max(0, n_seed_final - n_funda)),
                "added": 0,
            }
        )
        n_adv_hist = (
            int(df_funda["dollar_adv_hist"].notna().sum())
            if "dollar_adv_hist" in df_funda.columns
            else 0
        )
        universe_flow.append(
            {
                "step": "ADV warmup available (diagnostic)",
                "n_before": n_funda,
                "n_after": n_adv_hist,
                "removed": int(max(0, n_funda - n_adv_hist)),
                "added": 0,
            }
        )

        filt_before = int(df_funda.shape[0])
        filt_after = int(df_filtered.shape[0])
        filt_removed = int(
            sum(int(r.get("removed", 0) or 0) for r in (filter_flow or []))
        )
        filt_added = int(sum(int(r.get("added", 0) or 0) for r in (filter_flow or [])))
        if filter_flow and (filt_before - filt_removed + filt_added) != filt_after:
            logger.warning(
                "Filter flow reconciliation mismatch: before=%d, -removed=%d, +added=%d, after=%d.",
                filt_before,
                filt_removed,
                filt_added,
                filt_after,
            )
        universe_flow.append(
            {
                "step": "Filters applied",
                "n_before": filt_before,
                "n_after": filt_after,
                "removed": filt_removed,
                "added": filt_added,
            }
        )
        if universe_flow:
            logger.info(
                "Universe flow summary (paper-ready; reconciles by construction):"
            )
            for row in universe_flow:
                logger.info(
                    "  - %s: %d -> %d (removed=%d, added=%d)",
                    str(row.get("step") or "?"),
                    int(row.get("n_before", 0) or 0),
                    int(row.get("n_after", 0) or 0),
                    int(row.get("removed", 0) or 0),
                    int(row.get("added", 0) or 0),
                )
    except Exception:
        universe_flow = []

    extra: Extra = {
        "n_tickers_total": len(tickers_all),
        "n_fundamentals_ok": int(df_funda.shape[0]),
        "n_failed": int(hard_failures + incomplete_core_total),
        "n_failed_hard": int(hard_failures),
        "n_incomplete_core": incomplete_core_total,
        "n_filtered": int(df_filtered.shape[0]),
        "universe_flow": list(universe_flow),
        "filter_flow": list(filter_flow),
        "cfg_path": str(cfg_path),
        "reason_codes": reasons,
        "artifacts": {},
        "tickers_all": tickers_all,
        "run_id": run_id,
    }
    try:
        screener_meta = dict(get_last_screener_meta())
    except Exception:
        screener_meta = {}
    if not screener_meta and cached_seed_provenance:
        screener_meta = dict(cached_seed_provenance)
    extra["screener_provenance"] = screener_meta
    extra["fundamentals_provenance"] = {
        "source": "yfinance",
        "started_at_utc": fundamentals_started_at_utc,
        "fetched_at_utc": fundamentals_finished_at_utc,
        "cache_enabled": bool(fundamentals_cache_enabled),
        "cache_used": bool(
            not cached_fundamentals.empty and not cache_stale and not force_rebuild
        ),
        "cache_ttl_days": float(fundamentals_cache_ttl_days),
        "cache_store_path": str(fundamentals_store_path)
        if fundamentals_store_path
        else None,
        "cache_store_mtime_utc": fundamentals_store_mtime_utc,
    }
    adv_fp: str | None = None
    if isinstance(adv_meta, dict):
        fp_meta_raw = adv_meta.get("fingerprint")
        if isinstance(fp_meta_raw, str) and fp_meta_raw.strip():
            adv_fp = fp_meta_raw.strip()
    if not adv_fp:
        try:
            adv_fp = adv_fingerprint(
                adv_seed_tickers,
                cfg.raw.get("data", {}) or {},
                warmup_window,
                warmup_end=first_window_end_ts,
            )
        except Exception:
            adv_fp = None
    extra["adv_provenance"] = {
        "window": warmup_window,
        "warmup_start": (
            first_window_start_ts.isoformat()
            if first_window_start_ts is not None
            else None
        ),
        "warmup_end": first_window_end_ts.isoformat()
        if first_window_end_ts is not None
        else None,
        "fingerprint": adv_fp,
        "download_start_date": data_cfg.get("download_start_date"),
        "download_end_date": data_cfg.get("download_end_date"),
        "download_period": data_cfg.get("download_period"),
        "download_interval": data_cfg.get("download_interval", "1d"),
        "adv_cache": str(artifact_paths.adv_csv),
    }
    extra["canary"] = can_stats

    return df_filtered, df_funda, monitoring, extra
