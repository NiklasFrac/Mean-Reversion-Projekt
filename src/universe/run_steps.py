from __future__ import annotations

import pickle
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import pandas as pd

from universe.adv_outputs import emit_adv_cache_and_filtered_csv
from universe.artifact_keys import (
    KEY_PRICES,
    KEY_PRICES_CANONICAL,
    KEY_PRICES_UNADJUSTED,
    KEY_PRICES_UNADJUSTED_WARMUP,
    KEY_VOLUMES,
    KEY_VOLUMES_CANONICAL,
    KEY_VOLUMES_UNADJUSTED,
    set_artifact_path,
)
from universe.coercion import cfg_bool, safe_float
from universe.downloads import (
    build_download_plan,
    price_tickers_with_data,
    volume_tickers_with_data,
)
from universe.monitoring import logger, stage_timer
from universe.storage import ArtifactPaths, resolve_artifact_paths

__all__ = ["RunnerDeps", "download_and_persist_history"]


@dataclass(frozen=True)
class RunnerDeps:
    fetch_price_volume_data: Callable[..., tuple[pd.DataFrame, pd.DataFrame]]
    retry_missing_history: Callable[..., tuple[pd.DataFrame, pd.DataFrame, list[str]]]
    normalize_panel: Callable[[pd.DataFrame, str], pd.DataFrame]
    fetch_unadjusted_panels: Callable[
        ..., tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]
    ]
    validate_unadjusted_vs_adjusted: Callable[..., None]
    build_data_quality_report: Callable[..., dict[str, Any]]
    write_universe_csv: Callable[[pd.DataFrame, Path], None]
    write_universe_ext_csv: Callable[..., None]
    artifact_targets: Callable[..., tuple[Path, Path, Path | None, Path | None]]
    atomic_write_pickle: Callable[[object, Path], None]
    norm_symbol: Callable[[str], str]


@dataclass
class DownloadSettings:
    allow_incomplete_history: bool
    adjust_dividends: bool
    strict_unadjusted_validation: bool
    group_download: bool
    start_date: Any
    end_date: Any
    interval: str
    batch_size: int
    pause: float
    retries: int
    backoff: float
    use_threads: bool
    max_workers: int | None
    progress_bar: bool
    request_timeout: float | None


@dataclass
class DownloadResult:
    prices_adj: pd.DataFrame
    volumes_adj: pd.DataFrame
    missing_adj: list[str]
    prices_unadj: pd.DataFrame
    volumes_unadj: pd.DataFrame
    missing_unadj: list[str]
    unadjusted_meta: dict[str, Any]


def _panel_symbol_from_col_name(name: str) -> str:
    for suffix in ("_open", "_high", "_low", "_close"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _slice_panel_to_tickers(
    panel: pd.DataFrame,
    *,
    tickers: list[str],
    norm_symbol_fn: Callable[[str], str],
) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame() if panel is None else panel
    wanted = {norm_symbol_fn(str(t)) for t in tickers if str(t).strip()}
    if not wanted:
        return panel.iloc[:, 0:0]
    keep_cols = [
        col
        for col in panel.columns
        if norm_symbol_fn(_panel_symbol_from_col_name(str(col))) in wanted
    ]
    return panel.loc[:, keep_cols]


def _build_download_settings(
    *,
    data_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    tickers_final: list[str],
    derive_date_range_fn: Callable[[dict[str, Any]], tuple[Any, Any]],
) -> DownloadSettings:
    allow_incomplete_history = cfg_bool(data_cfg, "allow_incomplete_history", False)
    adjust_dividends = cfg_bool(data_cfg, "adjust_dividends", True)
    strict_unadjusted_validation = cfg_bool(
        data_cfg, "strict_unadjusted_validation", False
    )
    group_download = cfg_bool(data_cfg, "group_download", False)

    if "download_start_date" in data_cfg or "download_period" in data_cfg:
        date_cfg = dict(data_cfg)
    else:
        # Keep compatibility with injected derive_date_range_fn in tests by
        # allowing externally supplied start/end derivation when config omits both.
        start_compat, end_compat = derive_date_range_fn(dict(data_cfg))
        date_cfg = dict(data_cfg)
        date_cfg["download_start_date"] = start_compat
        date_cfg["download_end_date"] = end_compat

    plan = build_download_plan(date_cfg, n_tickers=len(tickers_final))

    progress_bar = cfg_bool(runtime_cfg, "progress_bar", True)
    request_timeout_raw = runtime_cfg.get("request_timeout")
    request_timeout: float | None = None
    if request_timeout_raw is not None:
        parsed_timeout = safe_float(request_timeout_raw)
        if parsed_timeout is None:
            logger.warning(
                "Invalid runtime.request_timeout=%r; disabling vendor timeout.",
                request_timeout_raw,
            )
        elif parsed_timeout > 0:
            request_timeout = float(parsed_timeout)

    return DownloadSettings(
        allow_incomplete_history=allow_incomplete_history,
        adjust_dividends=adjust_dividends,
        strict_unadjusted_validation=strict_unadjusted_validation,
        group_download=group_download,
        start_date=plan.start_date,
        end_date=plan.end_date,
        interval=plan.interval,
        batch_size=plan.batch_size,
        pause=plan.pause,
        retries=plan.retries,
        backoff=plan.backoff,
        use_threads=plan.use_threads,
        max_workers=plan.max_workers,
        progress_bar=progress_bar,
        request_timeout=request_timeout,
    )


def _download_panels(
    *,
    deps: RunnerDeps,
    tickers_final: list[str],
    settings: DownloadSettings,
    stop_event: threading.Event | None,
    extra_out: MutableMapping[str, Any],
) -> DownloadResult:
    prices_adj = pd.DataFrame()
    volumes_adj = pd.DataFrame()
    missing_adj: list[str] = []

    prices_unadj = pd.DataFrame()
    volumes_unadj = pd.DataFrame()
    missing_unadj: list[str] = []
    unadj_meta: dict[str, Any] = {}

    if not tickers_final:
        return DownloadResult(
            prices_adj=prices_adj,
            volumes_adj=volumes_adj,
            missing_adj=missing_adj,
            prices_unadj=prices_unadj,
            volumes_unadj=volumes_unadj,
            missing_unadj=missing_unadj,
            unadjusted_meta=unadj_meta,
        )

    auto_adjust = bool(settings.adjust_dividends)
    with stage_timer("download_prices"):
        prices_adj, volumes_adj = deps.fetch_price_volume_data(
            tickers_final,
            settings.start_date,
            settings.end_date,
            settings.interval,
            settings.batch_size,
            settings.pause,
            settings.retries,
            settings.backoff,
            settings.use_threads,
            max_workers=settings.max_workers,
            stop_event=stop_event,
            auto_adjust=auto_adjust,
            show_progress=settings.progress_bar,
            request_timeout=settings.request_timeout,
            junk_filter=None,
        )
        prices_adj, volumes_adj, missing_adj = deps.retry_missing_history(
            tickers_final,
            prices_adj,
            volumes_adj,
            start_date=settings.start_date,
            end_date=settings.end_date,
            interval=settings.interval,
            pause=settings.pause,
            retries=settings.retries,
            backoff=settings.backoff,
            stop_event=stop_event,
            auto_adjust=auto_adjust,
            show_progress=settings.progress_bar,
            request_timeout=settings.request_timeout,
            junk_filter=None,
        )
        prices_adj = deps.normalize_panel(prices_adj, settings.interval)
        volumes_adj = deps.normalize_panel(volumes_adj, settings.interval)
        have = price_tickers_with_data(prices_adj) & volume_tickers_with_data(
            volumes_adj
        )
        missing_after_norm = sorted(set(tickers_final) - have)
        if missing_after_norm != missing_adj:
            logger.warning(
                "Adjusted panels missing %d tickers after normalization (previously %d); "
                "using post-normalization missing list (examples: %s).",
                len(missing_after_norm),
                len(missing_adj),
                ", ".join(missing_after_norm[:5]),
            )
            missing_adj = missing_after_norm

    if settings.adjust_dividends:
        with stage_timer("download_prices_unadjusted"):
            prices_unadj, volumes_unadj, missing_unadj, unadj_meta = (
                deps.fetch_unadjusted_panels(
                    tickers_final,
                    download_start_date=settings.start_date,
                    download_end_date=settings.end_date,
                    download_interval=settings.interval,
                    batch_size=settings.batch_size,
                    pause=settings.pause,
                    max_retries=settings.retries,
                    backoff_factor=settings.backoff,
                    use_threads=settings.use_threads,
                    max_download_workers=settings.max_workers,
                    stop_event=stop_event,
                    progress_bar=settings.progress_bar,
                    request_timeout=settings.request_timeout,
                    junk_filter=None,
                )
            )
            extra_out["unadjusted_download"] = unadj_meta
            extra_out["data_quality"] = deps.build_data_quality_report(
                prices_unadjusted=prices_unadj,
                volumes_reported=volumes_unadj,
            )
            deps.validate_unadjusted_vs_adjusted(
                prices_adjusted=prices_adj,
                prices_unadjusted=prices_unadj,
                interval=settings.interval,
                strict=settings.strict_unadjusted_validation,
            )

    for name, panel in (("prices", prices_adj), ("volumes", volumes_adj)):
        if panel is None or panel.empty:
            continue
        idx = pd.to_datetime(panel.index, errors="coerce")
        if idx.duplicated().any() or not idx.is_monotonic_increasing:
            raise RuntimeError(
                f"Non-monotonic index in {name} panel after normalization."
            )

    return DownloadResult(
        prices_adj=prices_adj,
        volumes_adj=volumes_adj,
        missing_adj=missing_adj,
        prices_unadj=prices_unadj,
        volumes_unadj=volumes_unadj,
        missing_unadj=missing_unadj,
        unadjusted_meta=unadj_meta,
    )


def _apply_missing_history_policy(
    *,
    deps: RunnerDeps,
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    tickers_final: list[str],
    missing_all: list[str],
    allow_incomplete_history: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not missing_all:
        return df_universe, df_fundamentals, tickers_final
    if not allow_incomplete_history:
        raise RuntimeError(
            f"Missing history for {len(missing_all)} tickers (examples: {missing_all[:5]})."
        )
    drop = {deps.norm_symbol(t) for t in missing_all}
    keep = [t for t in tickers_final if deps.norm_symbol(t) not in drop]
    df_universe = df_universe.loc[
        [t for t in df_universe.index if deps.norm_symbol(t) in set(keep)]
    ]
    df_fundamentals = df_fundamentals.reindex(df_universe.index)
    tickers_final = [deps.norm_symbol(t) for t in df_universe.index.tolist()]
    # Final universe csv artifacts are written in runner_universe.main only after
    # this function returns successfully, so failed downstream steps cannot leave
    # partially committed final outputs.
    return df_universe, df_fundamentals, tickers_final


def _persist_unadjusted_panels(
    *,
    data_cfg: Mapping[str, Any],
    prices_unadj: pd.DataFrame,
    volumes_unadj: pd.DataFrame,
    artifacts: MutableMapping[str, str],
    atomic_write_pickle_fn: Callable[[object, Path], None],
    artifact_paths: ArtifactPaths | None = None,
) -> None:
    # Persist filtered unadjusted panels (separate from warmup cache).
    paths = artifact_paths or resolve_artifact_paths(data_cfg=data_cfg)
    warmup_path = paths.raw_prices_unadj_warmup_cache
    raw_unadj = paths.raw_prices_unadj_cache
    atomic_write_pickle_fn(prices_unadj, raw_unadj)
    raw_vol_unadj = paths.raw_volume_unadj_cache
    atomic_write_pickle_fn(volumes_unadj, raw_vol_unadj)
    set_artifact_path(
        artifacts,
        key=KEY_PRICES_UNADJUSTED_WARMUP,
        path=warmup_path,
    )
    set_artifact_path(
        artifacts,
        key=KEY_PRICES_UNADJUSTED,
        path=raw_unadj,
    )
    set_artifact_path(
        artifacts,
        key=KEY_VOLUMES_UNADJUSTED,
        path=raw_vol_unadj,
    )


def _persist_adjusted_panels(
    *,
    deps: RunnerDeps,
    data_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    prices_adj: pd.DataFrame,
    volumes_adj: pd.DataFrame,
    artifacts: MutableMapping[str, str],
) -> None:
    def _same_path(a: Path, b: Path) -> bool:
        try:
            return a.resolve(strict=False) == b.resolve(strict=False)
        except Exception:
            return str(a) == str(b)

    use_hashed_artifacts = cfg_bool(runtime_cfg, "use_hashed_artifacts", True)
    if prices_adj.empty and volumes_adj.empty:
        return
    price_bytes = pickle.dumps(prices_adj, protocol=pickle.HIGHEST_PROTOCOL)
    vol_bytes = pickle.dumps(volumes_adj, protocol=pickle.HIGHEST_PROTOCOL)
    prices_path, vols_path, mirror_prices, mirror_vols = deps.artifact_targets(
        hashed=use_hashed_artifacts,
        data_cfg=dict(data_cfg),
        price_bytes=price_bytes,
        vol_bytes=vol_bytes,
    )
    prices_path.parent.mkdir(parents=True, exist_ok=True)
    vols_path.parent.mkdir(parents=True, exist_ok=True)
    deps.atomic_write_pickle(prices_adj, prices_path)
    deps.atomic_write_pickle(volumes_adj, vols_path)
    if mirror_prices is not None and not _same_path(prices_path, mirror_prices):
        mirror_prices.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(prices_path), str(mirror_prices))
    if mirror_vols is not None and not _same_path(vols_path, mirror_vols):
        mirror_vols.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(vols_path), str(mirror_vols))
    set_artifact_path(artifacts, key=KEY_PRICES, path=prices_path)
    set_artifact_path(artifacts, key=KEY_VOLUMES, path=vols_path)
    set_artifact_path(artifacts, key=KEY_PRICES_CANONICAL, path=mirror_prices)
    set_artifact_path(artifacts, key=KEY_VOLUMES_CANONICAL, path=mirror_vols)


def download_and_persist_history(
    *,
    deps: RunnerDeps,
    df_universe: pd.DataFrame,
    df_fundamentals: pd.DataFrame,
    tickers_final: list[str],
    data_cfg: Mapping[str, Any],
    universe_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    stop_event: threading.Event | None,
    extra_out: MutableMapping[str, Any],
    derive_date_range_fn: Callable[[dict[str, Any]], tuple[Any, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Path | None, Path | None]:
    artifact_paths = resolve_artifact_paths(
        universe_cfg=universe_cfg,
        data_cfg=data_cfg,
        runtime_cfg=runtime_cfg,
    )
    settings = _build_download_settings(
        data_cfg=data_cfg,
        runtime_cfg=runtime_cfg,
        tickers_final=tickers_final,
        derive_date_range_fn=derive_date_range_fn,
    )

    extra_out["data_policy"] = {
        "adjust_dividends": settings.adjust_dividends,
        "allow_incomplete_history": settings.allow_incomplete_history,
        "download_start_date": settings.start_date,
        "download_end_date": settings.end_date,
        "download_interval": settings.interval,
        "download_batch": settings.batch_size,
        "download_threads": data_cfg.get("download_threads"),
        "group_download": settings.group_download,
        "download_pause": settings.pause,
        "download_retries": settings.retries,
        "backoff_factor": settings.backoff,
        "request_timeout_sec": settings.request_timeout,
        # Universe panels are normalized to UTC and persisted as tz-naive timestamps.
        # Downstream consumers should interpret naive timestamps as UTC unless
        # explicitly overridden.
        "raw_index_naive_tz": "UTC",
        "raw_index_timezone": "UTC",
        "raw_index_is_tz_naive": True,
        "unadjusted_coverage_definition": "price_and_volume",
        "market_cap_asof": "run_time_snapshot",
        "filters_asof": {
            "price": "warmup_median_no_row_fallback",
            "volume": "warmup_mean_no_row_fallback",
            "dollar_adv": "warmup_hist_if_column_present_else_snapshot_no_row_fallback",
            "market_cap": "run_time_snapshot",
        },
    }

    res = _download_panels(
        deps=deps,
        tickers_final=tickers_final,
        settings=settings,
        stop_event=stop_event,
        extra_out=extra_out,
    )

    missing_all = sorted(set(res.missing_adj) | set(res.missing_unadj))
    df_universe, df_fundamentals, tickers_final = _apply_missing_history_policy(
        deps=deps,
        df_universe=df_universe,
        df_fundamentals=df_fundamentals,
        tickers_final=tickers_final,
        missing_all=missing_all,
        allow_incomplete_history=settings.allow_incomplete_history,
    )
    # Keep summary counts consistent when symbols are dropped after history checks.
    extra_out["n_filtered"] = int(df_universe.shape[0])

    prices_adj = _slice_panel_to_tickers(
        res.prices_adj, tickers=tickers_final, norm_symbol_fn=deps.norm_symbol
    )
    volumes_adj = _slice_panel_to_tickers(
        res.volumes_adj, tickers=tickers_final, norm_symbol_fn=deps.norm_symbol
    )
    prices_unadj = _slice_panel_to_tickers(
        res.prices_unadj, tickers=tickers_final, norm_symbol_fn=deps.norm_symbol
    )
    volumes_unadj = _slice_panel_to_tickers(
        res.volumes_unadj, tickers=tickers_final, norm_symbol_fn=deps.norm_symbol
    )

    if settings.adjust_dividends:
        _persist_unadjusted_panels(
            data_cfg=data_cfg,
            prices_unadj=prices_unadj,
            volumes_unadj=volumes_unadj,
            artifacts=extra_out.setdefault("artifacts", {}),
            atomic_write_pickle_fn=deps.atomic_write_pickle,
            artifact_paths=artifact_paths,
        )

    _persist_adjusted_panels(
        deps=deps,
        data_cfg=data_cfg,
        runtime_cfg=runtime_cfg,
        prices_adj=prices_adj,
        volumes_adj=volumes_adj,
        artifacts=extra_out.setdefault("artifacts", {}),
    )

    adv_csv_path, adv_csv_filtered_path = emit_adv_cache_and_filtered_csv(
        data_cfg=data_cfg,
        universe_cfg=universe_cfg,
        tickers_final=tickers_final,
        artifacts=extra_out.setdefault("artifacts", {}),
        norm_symbol_fn=deps.norm_symbol,
        atomic_write_pickle_fn=deps.atomic_write_pickle,
        artifact_paths=artifact_paths,
    )

    return (
        df_universe,
        df_fundamentals,
        tickers_final,
        adv_csv_path,
        adv_csv_filtered_path,
    )
