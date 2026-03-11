from __future__ import annotations

import datetime as dt
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd
from universe.coercion import cfg_bool, cfg_float, cfg_int, clamp01
from universe.datetime_utils import coerce_date_like_value
from universe.panel_utils import (
    coerce_utc_naive_index,
    collapse_duplicate_index_rows,
    merge_duplicate_columns_prefer_non_null,
)
from universe.symbol_utils import normalize_symbols
from universe.ticker_sets import price_tickers, volume_tickers
from universe.vendor import period_start_from_end

__all__ = [
    "DownloadRequestPlan",
    "build_download_plan",
    "derive_date_range",
    "fetch_unadjusted_panels",
    "normalize_panel_for_universe",
    "price_tickers_with_data",
    "validate_unadjusted_vs_adjusted",
    "volume_tickers_with_data",
]

logger = logging.getLogger("runner_universe")


@dataclass(frozen=True)
class DownloadRequestPlan:
    start_date: Any
    end_date: Any
    interval: str
    batch_size: int
    pause: float
    retries: int
    backoff: float
    use_threads: bool
    max_workers: int | None


def normalize_panel_for_universe(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    interval_s = str(interval or "").lower()
    idx = coerce_utc_naive_index(
        out.index, normalize=interval_s.endswith(("d", "wk", "mo"))
    )
    # Ensure daily bars line up across tickers (Yahoo sometimes returns a mix of tz-aware and tz-naive
    # indices; after tz stripping this can manifest as 00:00 vs 04:00/05:00 timestamps and create
    # duplicate "days" when panels are combined).
    out.index = idx
    out = out.loc[out.index.notna()]
    out = out.sort_index()
    # Merge duplicate timestamps by taking the last non-null value per column.
    # This avoids losing data when different tickers land on different times-of-day pre-normalization.
    out = collapse_duplicate_index_rows(out)
    return merge_duplicate_columns_prefer_non_null(out)


def price_tickers_with_data(prices_df: pd.DataFrame) -> set[str]:
    return price_tickers(prices_df, require_data=True, include_bare_columns=True)


def volume_tickers_with_data(volumes_df: pd.DataFrame) -> set[str]:
    return volume_tickers(volumes_df, require_data=True)


def build_download_plan(
    data_cfg: Mapping[str, Any],
    *,
    n_tickers: int,
) -> DownloadRequestPlan:
    start_date, end_date = derive_date_range(dict(data_cfg))
    interval = str(data_cfg.get("download_interval", "1d"))
    batch_size = cfg_int(data_cfg, "download_batch", 50, min_value=1, logger=logger)
    pause = cfg_float(data_cfg, "download_pause", 1.0, min_value=0.0, logger=logger)
    retries = cfg_int(data_cfg, "download_retries", 3, min_value=0, logger=logger)
    backoff = cfg_float(
        data_cfg,
        "backoff_factor",
        2.0,
        strictly_positive=True,
        logger=logger,
    )

    group_download = cfg_bool(data_cfg, "group_download", False)
    download_threads_cfg = data_cfg.get("download_threads")
    use_threads = int(n_tickers) > 1 and not group_download
    max_workers = None
    if group_download:
        batch_size = max(1, int(n_tickers) if int(n_tickers) > 0 else 1)
        use_threads = False
    if download_threads_cfg is not None:
        try:
            dl_threads = max(1, int(download_threads_cfg))
            if dl_threads == 1:
                use_threads = False
            else:
                max_workers = dl_threads
        except Exception:
            logger.warning(
                "Invalid data.download_threads=%r; falling back to auto.",
                download_threads_cfg,
            )

    return DownloadRequestPlan(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        batch_size=batch_size,
        pause=pause,
        retries=retries,
        backoff=backoff,
        use_threads=use_threads,
        max_workers=max_workers,
    )


def fetch_unadjusted_panels(
    tickers: list[str],
    *,
    download_start_date: Any,
    download_end_date: Any,
    download_interval: str,
    batch_size: int,
    pause: float,
    max_retries: int,
    backoff_factor: float,
    use_threads: bool,
    max_download_workers: int | None,
    stop_event: threading.Event | None,
    progress_bar: bool,
    junk_filter: Callable[[str], bool] | None,
    request_timeout: float | None = None,
    coverage_threshold: float = 0.85,
    fetch_price_volume_data_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]],
    retry_missing_history_fn: Callable[
        ..., tuple[pd.DataFrame, pd.DataFrame, list[str]]
    ],
    normalize_panel_fn: Callable[
        [pd.DataFrame, str], pd.DataFrame
    ] = normalize_panel_for_universe,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    want = normalize_symbols(tickers)
    try:
        coverage_threshold_val = float(coverage_threshold)
    except Exception:
        coverage_threshold_val = 0.85
    coverage_threshold_val = clamp01(coverage_threshold_val, default=0.85)
    meta: dict[str, Any] = {
        "wanted": len(want),
        "coverage_threshold": float(coverage_threshold_val),
        "retried": False,
        "observed": 0,
        "coverage_ratio": 0.0,
    }
    if not want:
        return pd.DataFrame(), pd.DataFrame(), [], meta

    prices_df, volumes_df = fetch_price_volume_data_fn(
        want,
        download_start_date,
        download_end_date,
        download_interval,
        int(batch_size),
        float(pause),
        int(max_retries),
        float(backoff_factor),
        bool(use_threads),
        max_workers=max_download_workers,
        stop_event=stop_event,
        auto_adjust=False,
        show_progress=bool(progress_bar),
        request_timeout=request_timeout,
        junk_filter=junk_filter,
    )
    prices_df, volumes_df, _ = retry_missing_history_fn(
        want,
        prices_df,
        volumes_df,
        start_date=download_start_date,
        end_date=download_end_date,
        interval=download_interval,
        pause=float(pause),
        retries=int(max_retries),
        backoff=float(backoff_factor),
        stop_event=stop_event,
        auto_adjust=False,
        show_progress=bool(progress_bar),
        request_timeout=request_timeout,
        junk_filter=junk_filter,
    )
    prices_df = normalize_panel_fn(prices_df, download_interval)
    volumes_df = normalize_panel_fn(volumes_df, download_interval)

    have_price = price_tickers_with_data(prices_df)
    have_volume = volume_tickers_with_data(volumes_df)
    # A ticker is considered covered only when both close and volume history exist.
    have = have_price & have_volume
    coverage = (len(have) / len(want)) if want else 1.0
    meta["observed"] = int(len(have))
    meta["coverage_ratio"] = float(coverage)

    if coverage < coverage_threshold_val:
        missing_for_retry = sorted(set(want) - have)
        if missing_for_retry:
            meta["retried"] = True
            retry_prices, retry_vols = fetch_price_volume_data_fn(
                missing_for_retry,
                download_start_date,
                download_end_date,
                download_interval,
                1,
                float(pause),
                int(max_retries),
                float(backoff_factor),
                False,
                max_workers=None,
                stop_event=stop_event,
                auto_adjust=False,
                show_progress=bool(progress_bar),
                request_timeout=request_timeout,
                junk_filter=junk_filter,
            )
            retry_prices, retry_vols, _ = retry_missing_history_fn(
                missing_for_retry,
                retry_prices,
                retry_vols,
                start_date=download_start_date,
                end_date=download_end_date,
                interval=download_interval,
                pause=float(pause),
                retries=int(max_retries),
                backoff=float(backoff_factor),
                stop_event=stop_event,
                auto_adjust=False,
                show_progress=bool(progress_bar),
                request_timeout=request_timeout,
                junk_filter=junk_filter,
            )
            if not retry_prices.empty:
                prices_df = pd.concat([prices_df, retry_prices], axis=1)
            if not retry_vols.empty:
                volumes_df = pd.concat([volumes_df, retry_vols], axis=1)
            prices_df = normalize_panel_fn(prices_df, download_interval)
            volumes_df = normalize_panel_fn(volumes_df, download_interval)
            have_price = price_tickers_with_data(prices_df)
            have_volume = volume_tickers_with_data(volumes_df)
            have = have_price & have_volume
            coverage = (len(have) / len(want)) if want else 1.0
            meta["observed"] = int(len(have))
            meta["coverage_ratio"] = float(coverage)

    missing = sorted(set(want) - have)
    return prices_df, volumes_df, missing, meta


def derive_date_range(data_cfg: dict[str, Any]) -> tuple[Any, Any]:
    start = data_cfg.get("download_start_date")
    end = data_cfg.get("download_end_date", "today")
    if start:
        return coerce_date_like_value(start), coerce_date_like_value(end)
    period = data_cfg.get("download_period")
    if not period:
        raise ValueError(
            "data.download_start_date is required (or provide deprecated data.download_period)."
        )

    end_ts = pd.Timestamp(end if end is not None else "today")
    if str(end).strip().lower() in {"today", "now"}:
        end_ts = pd.Timestamp(dt.datetime.now(dt.UTC).date())
    start_ts = period_start_from_end(end_ts=end_ts, period=str(period))
    return start_ts.date().isoformat(), coerce_date_like_value(end)


def validate_unadjusted_vs_adjusted(
    *,
    prices_adjusted: pd.DataFrame,
    prices_unadjusted: pd.DataFrame,
    interval: str,
    min_rows: int = 5,
    strict: bool = False,
) -> None:
    if prices_adjusted is None or prices_unadjusted is None:
        return
    if prices_adjusted.empty or prices_unadjusted.empty:
        return
    common_idx = prices_adjusted.index.intersection(prices_unadjusted.index)
    if common_idx.size < int(min_rows):
        return
    adj = prices_adjusted.loc[common_idx]
    unadj = prices_unadjusted.loc[common_idx]
    tickers = sorted(price_tickers_with_data(adj) & price_tickers_with_data(unadj))
    if not tickers:
        return
    equal_cnt = 0
    compared = 0
    for sym in tickers:
        adj_col = f"{sym}_close" if f"{sym}_close" in adj.columns else sym
        unadj_col = f"{sym}_close" if f"{sym}_close" in unadj.columns else sym
        if adj_col not in adj.columns or unadj_col not in unadj.columns:
            continue
        a = pd.to_numeric(adj[adj_col], errors="coerce")
        b = pd.to_numeric(unadj[unadj_col], errors="coerce")
        mask = a.notna() & b.notna()
        if mask.sum() < int(min_rows):
            continue
        compared += 1
        if (a[mask] == b[mask]).all():
            equal_cnt += 1
    if compared > 0 and equal_cnt == compared:
        msg = (
            "Unadjusted price validation: adjusted and unadjusted closes are identical "
            f"(interval={interval}, compared={compared}). This can be valid when no "
            "dividend/split adjustments fall into the window."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning("%s", msg)
