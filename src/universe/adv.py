from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd

from universe.checkpoint import norm_symbol
from universe.coercion import cfg_float, cfg_int, clamp01, coerce_int
from universe.datetime_utils import coerce_utc_naive_timestamp
from universe.downloads import build_download_plan
from universe.monitoring import logger
from universe.panel_utils import coerce_utc_naive_index, collapse_duplicate_index_rows
from universe.storage import resolve_artifact_paths
from universe.symbol_utils import normalize_symbols
from universe.utils import _atomic_write_pickle
from universe.vendor import _retry_missing_history, fetch_price_volume_data

__all__ = [
    "compute_adv_map",
    "load_or_compute_adv_map",
    "load_price_volume_panels",
    "adv_fingerprint",
]


def _adv_fingerprint(
    tickers: list[str],
    data_cfg: dict[str, Any],
    window: int,
    *,
    warmup_end: Any | None = None,
) -> str:
    """
    Build a deterministic fingerprint for the ADV cache based on config and ticker set.
    This guards against reusing a cache built with different parameters.
    """
    warmup_end_norm: str | None = None
    try:
        ts_any = (
            pd.to_datetime(warmup_end, errors="coerce")
            if warmup_end is not None
            else pd.NaT
        )
        ts_raw: Any
        if isinstance(ts_any, pd.DatetimeIndex):
            ts_raw = ts_any[0] if len(ts_any) else pd.NaT
        else:
            ts_raw = ts_any
        ts0 = _coerce_ts(ts_raw)
        if ts0 is not None and not pd.isna(ts0):
            if getattr(ts0, "tzinfo", None) is not None:
                ts0 = ts0.tz_convert("UTC").tz_localize(None)
            warmup_end_norm = ts0.normalize().isoformat()
    except Exception:
        warmup_end_norm = str(warmup_end) if warmup_end is not None else None

    min_valid_ratio_raw = data_cfg.get("adv_min_valid_ratio", 0.7)
    try:
        min_valid_ratio = float(min_valid_ratio_raw)
    except Exception:
        min_valid_ratio = 0.7
    min_valid_ratio = clamp01(min_valid_ratio, default=0.7)

    payload = {
        "window": int(window),
        "min_valid_ratio": float(min_valid_ratio),
        "start_date": str(data_cfg.get("download_start_date") or ""),
        "end_date": str(data_cfg.get("download_end_date") or ""),
        "period": str(data_cfg.get("download_period") or ""),
        "interval": str(data_cfg.get("download_interval", "1d")),
        "warmup_end": warmup_end_norm,
        # Keep the ticker influence bounded via a stable hash of the sorted list.
        "tickers_sha1": hashlib.sha1(
            "|".join(sorted(normalize_symbols(tickers))).encode("utf-8")
        ).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def adv_fingerprint(
    tickers: list[str],
    data_cfg: dict[str, Any],
    window: int,
    *,
    warmup_end: Any | None = None,
) -> str:
    """Public helper to compute the ADV cache fingerprint (used in manifests/tests)."""
    return _adv_fingerprint(tickers, data_cfg, window, warmup_end=warmup_end)


def _coerce_ts(value: Any) -> pd.Timestamp | None:
    return coerce_utc_naive_timestamp(value, normalize=False, errors="coerce")


def _clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    idx = coerce_utc_naive_index(out.index, normalize=True)
    out.index = idx
    out = out.sort_index()
    out = collapse_duplicate_index_rows(out)
    return out


def _adv_and_price_for_symbol(
    sym: str,
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    window: int,
    warmup_end: pd.Timestamp | None,
    min_valid_ratio: float,
) -> tuple[float | None, float | None, float | None, pd.Timestamp | None, int]:
    price_col = f"{sym}_close" if f"{sym}_close" in prices_df.columns else sym
    if price_col not in prices_df.columns or sym not in volumes_df.columns:
        return None, None, None, None, 0
    px = pd.to_numeric(prices_df[price_col], errors="coerce")
    vol = pd.to_numeric(volumes_df[sym], errors="coerce")
    df = pd.concat([px, vol], axis=1, keys=["price", "volume"]).dropna()
    if df.empty:
        return None, None, None, None, 0
    df = df[df["price"].apply(math.isfinite) & df["volume"].apply(math.isfinite)]
    df = df.sort_index()

    idx_norm = coerce_utc_naive_index(df.index, normalize=True)
    if warmup_end is not None:
        cutoff = _coerce_ts(warmup_end)
        if cutoff is not None and not pd.isna(cutoff):
            if getattr(cutoff, "tzinfo", None) is not None:
                cutoff = cutoff.tz_convert("UTC").tz_localize(None)
            cutoff_norm = cutoff.normalize()
            mask = idx_norm <= cutoff_norm
            df = df.loc[mask]
            idx_norm = idx_norm[mask]
    if df.empty:
        return None, None, None, None, 0
    # Collapse duplicate calendar days (mixed timezones can create multiple rows per day).
    dedup_mask = ~idx_norm.duplicated(keep="last")
    df = df.loc[dedup_mask]
    if df.empty:
        return None, None, None, None, 0
    window = max(1, int(window))
    min_valid = max(1, int(math.ceil(window * float(min_valid_ratio))))
    window_df = df.tail(window)
    valid = window_df.shape[0]
    if valid < min_valid:
        return None, None, None, None, valid
    dollar_tv = window_df["price"] * window_df["volume"]
    dollar_tv = dollar_tv[pd.notna(dollar_tv)]
    price_seg = window_df["price"]
    vol_seg = pd.to_numeric(window_df["volume"], errors="coerce")
    if dollar_tv.empty or dollar_tv.shape[0] < min_valid:
        return None, None, None, None, valid
    adv = float(dollar_tv.median())
    price_med = float(price_seg.median()) if price_seg.size else None
    volume_mean = float(vol_seg.mean()) if vol_seg.size else None
    adv_asof = _coerce_ts(window_df.index.max())
    return adv, price_med, volume_mean, adv_asof, valid


def compute_adv_map(
    tickers: list[str],
    *,
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    warmup_end: pd.Timestamp | None,
    window: int,
    min_valid_ratio: float = 0.7,
) -> pd.DataFrame:
    """
    Compute historical dollar ADV and warmup median price over a warmup window ending at warmup_end.
    Expects price/volume panels already loaded.
    """
    tickers = normalize_symbols(tickers)
    if not tickers:
        return pd.DataFrame(
            columns=[
                "dollar_adv_hist",
                "price_warmup_med",
                "volume_warmup_avg",
                "adv_window",
                "adv_asof",
            ]
        )
    prices_df = _clean_panel(prices_df)
    volumes_df = _clean_panel(volumes_df)
    adv_values: dict[str, float] = {}
    price_meds: dict[str, float] = {}
    volume_means: dict[str, float] = {}
    adv_asof: dict[str, pd.Timestamp | None] = {}
    insufficient: list[str] = []
    min_valid_ratio = clamp01(float(min_valid_ratio), default=0.7)
    for sym in tickers:
        adv, price_med, volume_mean, asof_ts, valid = _adv_and_price_for_symbol(
            sym,
            prices_df,
            volumes_df,
            window,
            warmup_end,
            min_valid_ratio,
        )
        if adv is None or price_med is None or volume_mean is None:
            insufficient.append(sym)
            continue
        adv_values[sym] = adv
        price_meds[sym] = price_med
        volume_means[sym] = volume_mean
        adv_asof[sym] = asof_ts
    df = pd.DataFrame(
        {
            "dollar_adv_hist": pd.Series(adv_values),
            "price_warmup_med": pd.Series(price_meds),
            "volume_warmup_avg": pd.Series(volume_means),
            "adv_window": float(window),
            "adv_asof": pd.Series(adv_asof),
        }
    )
    df.index.name = "ticker"
    logger.info(
        "ADV warmup computed: %d with history, %d lacking sufficient data "
        "(window=%d, min_valid_ratio=%.3f, warmup_end=%s).",
        len(adv_values),
        len(insufficient),
        window,
        float(min_valid_ratio),
        warmup_end,
    )
    return df


def load_price_volume_panels(
    tickers: list[str],
    data_cfg: dict[str, Any],
    *,
    stop_event: threading.Event | None = None,
    auto_adjust: bool = True,
    request_timeout: float | None = None,
    prices_cache_key: str = "raw_prices_cache",
    volume_cache_key: str = "volume_path",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = normalize_symbols(tickers)
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    plan = build_download_plan(data_cfg, n_tickers=len(tickers))
    paths = resolve_artifact_paths(data_cfg=data_cfg)

    prices_df, volumes_df = fetch_price_volume_data(
        tickers,
        plan.start_date,
        plan.end_date,
        plan.interval,
        plan.batch_size,
        plan.pause,
        plan.retries,
        plan.backoff,
        plan.use_threads,
        max_workers=plan.max_workers,
        stop_event=stop_event,
        auto_adjust=auto_adjust,
        show_progress=False,
        request_timeout=request_timeout,
        junk_filter=None,
    )
    prices_df, volumes_df, _ = _retry_missing_history(
        tickers,
        prices_df,
        volumes_df,
        start_date=plan.start_date,
        end_date=plan.end_date,
        interval=plan.interval,
        pause=plan.pause,
        retries=plan.retries,
        backoff=plan.backoff,
        stop_event=stop_event,
        auto_adjust=auto_adjust,
        show_progress=False,
        request_timeout=request_timeout,
        junk_filter=None,
    )
    # Optional raw cache persistence
    raw_prices_path: Any = (
        paths.raw_prices_cache
        if prices_cache_key == "raw_prices_cache"
        else data_cfg.get(prices_cache_key)
    )
    raw_volume_path: Any = (
        paths.volume_path
        if volume_cache_key == "volume_path"
        else data_cfg.get(volume_cache_key)
    )
    try:
        if raw_prices_path:
            pp = (
                raw_prices_path
                if isinstance(raw_prices_path, Path)
                else Path(str(raw_prices_path))
            )
            _atomic_write_pickle(prices_df, pp)
        if raw_volume_path:
            vp = (
                raw_volume_path
                if isinstance(raw_volume_path, Path)
                else Path(str(raw_volume_path))
            )
            _atomic_write_pickle(volumes_df, vp)
    except Exception as exc:
        logger.warning("Failed to persist raw price/volume cache: %s", exc)
    return _clean_panel(prices_df), _clean_panel(volumes_df)


@overload
def load_or_compute_adv_map(
    tickers: list[str],
    cfg: dict[str, Any],
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    warmup_end: pd.Timestamp | None,
    *,
    return_meta: Literal[False] = False,
) -> pd.DataFrame: ...


@overload
def load_or_compute_adv_map(
    tickers: list[str],
    cfg: dict[str, Any],
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    warmup_end: pd.Timestamp | None,
    *,
    return_meta: Literal[True],
) -> tuple[pd.DataFrame, dict[str, Any]]: ...


def load_or_compute_adv_map(
    tickers: list[str],
    cfg: dict[str, Any],
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    warmup_end: pd.Timestamp | None,
    *,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    data_cfg = cfg.get("data", {}) or {}
    artifact_paths = resolve_artifact_paths(data_cfg=data_cfg)
    ttl_days = cfg_float(
        data_cfg,
        "adv_cache_ttl_days",
        cfg_float(
            data_cfg,
            "fundamentals_cache_ttl_days",
            30.0,
            min_value=0.0,
            logger=logger,
            section_name="data",
        ),
        min_value=0.0,
        logger=logger,
        section_name="data",
    )
    adv_path = artifact_paths.adv_csv
    window = cfg_int(
        data_cfg, "adv_window", 30, min_value=1, logger=logger, section_name="data"
    )
    fingerprint = _adv_fingerprint(tickers, data_cfg, window, warmup_end=warmup_end)
    meta_path = adv_path.with_suffix(adv_path.suffix + ".meta.json")
    start_date = str(data_cfg.get("download_start_date") or "")
    end_date = str(data_cfg.get("download_end_date") or "")
    period = str(data_cfg.get("download_period") or "")
    interval = str(data_cfg.get("download_interval", "1d"))
    min_cov_ratio = cfg_float(
        data_cfg,
        "adv_cache_min_coverage_ratio",
        0.8,
        min_value=0.0,
        logger=logger,
        section_name="data",
    )
    min_cov_ratio = clamp01(min_cov_ratio, default=0.8)
    min_valid_ratio = cfg_float(
        data_cfg,
        "adv_min_valid_ratio",
        0.7,
        min_value=0.0,
        logger=logger,
        section_name="data",
    )
    min_valid_ratio = clamp01(min_valid_ratio, default=0.7)

    warmup_end_iso: str | None = None
    warmup_ts = _coerce_ts(warmup_end)
    if warmup_ts is not None and not pd.isna(warmup_ts):
        if getattr(warmup_ts, "tzinfo", None) is not None:
            warmup_ts = warmup_ts.tz_convert("UTC").tz_localize(None)
        warmup_end_iso = warmup_ts.isoformat()

    required_cols = {
        "dollar_adv_hist",
        "price_warmup_med",
        "volume_warmup_avg",
        "adv_window",
        "adv_asof",
    }
    meta_info: dict[str, Any] = {
        "fingerprint": fingerprint,
        "adv_window": int(window),
        "adv_min_valid_ratio": float(min_valid_ratio),
        "download_start_date": start_date,
        "download_end_date": end_date,
        "download_period": period,
        "download_interval": interval,
        "warmup_end": warmup_end_iso,
        "tickers": len(tickers),
        "cache_path": str(adv_path),
        "min_coverage_ratio": float(min_cov_ratio),
    }
    fallback_df: pd.DataFrame | None = None
    fallback_age_days: float | None = None
    if adv_path.exists():
        try:
            age_days = (time.time() - adv_path.stat().st_mtime) / 86400.0
            fallback_age_days = float(age_days)
            fresh_enough = ttl_days <= 0 or age_days <= ttl_days
            meta_fingerprint = None
            expected_tickers = len(tickers)
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta_fingerprint = meta.get("fingerprint")
                    expected_tickers = coerce_int(
                        meta.get("tickers") or expected_tickers or 0,
                        expected_tickers,
                        min_value=0,
                        field_name="adv_meta.tickers",
                    )
                    meta_info.update(meta)
                except Exception:
                    meta_fingerprint = None
            df = pd.read_csv(adv_path, index_col=0)
            df.index = df.index.map(norm_symbol)
            df.index.name = "ticker"
            if "adv_asof" in df.columns:
                df["adv_asof"] = pd.to_datetime(df["adv_asof"], errors="coerce")
            cols_ok = required_cols.issubset(df.columns)
            fp_ok = meta_fingerprint == fingerprint
            coverage_ratio = (
                (float(df.shape[0]) / float(expected_tickers))
                if expected_tickers
                else 1.0
            )
            undercovered = bool(expected_tickers and coverage_ratio < min_cov_ratio)
            if cols_ok and fp_ok and not df.empty and not undercovered:
                fallback_df = df.copy()
                if fresh_enough:
                    logger.info(
                        "Loaded ADV map from cache %s (age=%.1fd, fp match, coverage=%.1f%%).",
                        adv_path,
                        age_days,
                        100.0 * coverage_ratio,
                    )
                    return (df, meta_info) if return_meta else df
            if fresh_enough:
                if df.empty:
                    reason = "empty cache"
                elif not cols_ok:
                    reason = "missing columns"
                elif undercovered:
                    reason = (
                        f"undercoverage coverage={coverage_ratio:.3f} < "
                        f"min_coverage_ratio={min_cov_ratio:.3f} "
                        f"(rows={df.shape[0]}, expected={expected_tickers})"
                    )
                else:
                    reason = "fingerprint mismatch"
                logger.info("ADV cache %s invalid (%s); recomputing.", adv_path, reason)
            else:
                if fallback_df is not None:
                    logger.info(
                        "ADV cache stale (age=%.1fd > ttl=%.1fd); recomputing with safe fallback.",
                        age_days,
                        ttl_days,
                    )
                else:
                    logger.info(
                        "ADV cache stale (age=%.1fd > ttl=%.1fd); recomputing.",
                        age_days,
                        ttl_days,
                    )
        except Exception as exc:
            logger.warning("Failed to load ADV map %s: %s", adv_path, exc)

    df_new = compute_adv_map(
        tickers,
        prices_df=prices_df,
        volumes_df=volumes_df,
        warmup_end=warmup_end,
        window=window,
        min_valid_ratio=min_valid_ratio,
    )
    if df_new.empty:
        # Never overwrite an existing non-empty cache with an empty recomputation:
        # transient vendor/data outages would otherwise poison subsequent runs.
        if fallback_df is not None:
            age_note = ""
            if (
                fallback_age_days is not None
                and ttl_days > 0
                and fallback_age_days > ttl_days
            ):
                age_note = f", stale age={fallback_age_days:.1f}d>ttl={ttl_days:.1f}d"
            logger.warning(
                "Computed ADV map is empty; reusing existing cache at %s (fp match%s).",
                adv_path,
                age_note,
            )
            return (fallback_df, meta_info) if return_meta else fallback_df
        if adv_path.exists():
            logger.error(
                "Computed ADV map is empty and existing cache at %s is not a safe fallback.",
                adv_path,
            )
        else:
            logger.error(
                "Computed ADV map is empty and no cache exists at %s.",
                adv_path,
            )
        raise RuntimeError(
            "Computed ADV map is empty and no safe fallback ADV cache is available."
        )
    try:
        adv_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(adv_path)
        try:
            meta = {
                "fingerprint": fingerprint,
                "adv_window": int(window),
                "adv_min_valid_ratio": float(min_valid_ratio),
                "download_start_date": start_date,
                "download_end_date": end_date,
                "download_period": period,
                "download_interval": interval,
                "tickers": len(tickers),
                "warmup_end": warmup_end_iso,
            }
            meta_info.update(meta)
            meta_path.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
        except Exception as meta_exc:
            logger.warning(
                "Failed to persist ADV cache metadata %s: %s", meta_path, meta_exc
            )
        logger.info("ADV map saved to %s (rows=%d).", adv_path, int(df_new.shape[0]))
    except Exception as exc:
        logger.warning("Failed to persist ADV map to %s: %s", adv_path, exc)
    return (df_new, meta_info) if return_meta else df_new
