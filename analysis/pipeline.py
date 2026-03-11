"""Main analysis pipeline orchestrating loading, numerics, bootstrap, and persistence."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from analysis.atomic_io import (
    _atomic_write_bytes,
    _atomic_write_text,
    save_json,
    save_parquet,
    save_results,
)
from analysis.bootstrap_fdr import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bootstrap_pair_stats,
)
from analysis.config_io import (
    deep_merge,
    dict_hash_sha256,
    file_sha256,
    get_git_sha,
    load_config,
)
from analysis.logging_config import logger
from analysis.mp_helpers import _bootstrap_batch_worker, _worker_init_for_mp
from analysis.numerics import compute_log_returns, compute_shrink_corr
from analysis.pairs import list_high_pairs_vectorized
from analysis.preprocess import load_filled_data
from analysis.prometheus_metrics import _PROM, init_prometheus, prom_observe
from analysis.rolling import rolling_pair_metrics_fast
from analysis.threading_control import set_thread_limits
from analysis.utils import _guard, _pct_label, _stage


def _dq_summary(
    pos_frac: pd.Series,
    nan_frac: pd.Series,
    zero_count: pd.Series,
    dropped_cols: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    def _stat(s: pd.Series) -> dict[str, Optional[float]]:
        if s.empty:
            return {"min": None, "mean": None, "median": None, "max": None}
        return {
            "min": float(s.min()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "max": float(s.max()),
        }

    zero_nonzero = int((zero_count > 0).sum()) if not zero_count.empty else 0
    zero_max = float(zero_count.max()) if not zero_count.empty else None

    out: dict[str, Any] = {
        "pos_frac": _stat(pos_frac),
        "nan_frac": _stat(nan_frac),
        "zero_count": {"max": zero_max, "nonzero": zero_nonzero},
    }
    if dropped_cols is not None:
        out["dropped_cols"] = list(dropped_cols)
    return out


def _utc_run_id(prefix: str, cfg_hash: str) -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{ts}-{cfg_hash[:8]}"


def _prepare_run_artifacts(
    *,
    pairs_out: Path,
    cfg_path: str | Path | None,
    cfg: dict[str, Any],
    cfg_hash: str,
) -> dict[str, Any]:
    """
    Prepare immutable, per-run artifact locations and persist a config snapshot.

    Design:
    - "Latest mutable" stays at pairs_out (existing behavior).
    - Additionally, every run writes a run-scoped copy under <pairs_out.parent>/by_run/<run_id>/...
      so audits are reproducible even if runs/data is overwritten later.
    """
    run_id = _utc_run_id("ANL", cfg_hash)
    run_root = pairs_out.parent / "by_run" / run_id
    inputs_dir = run_root / "inputs"
    outputs_dir = run_root / "outputs"
    run_pairs_out = outputs_dir / pairs_out.name

    # Persist resolved config as JSON for exact reproducibility.
    resolved_cfg_path = inputs_dir / "config_resolved.json"
    resolved_cfg_text = json.dumps(cfg, indent=2, default=str)
    _atomic_write_text(resolved_cfg_path, resolved_cfg_text)
    resolved_cfg_sha256 = file_sha256(resolved_cfg_path)

    source_copy_path: Optional[Path] = None
    source_copy_sha256: Optional[str] = None
    if cfg_path is not None:
        try:
            src_p = Path(cfg_path)
            if src_p.exists() and src_p.is_file():
                suffix = src_p.suffix or ".yaml"
                source_copy_path = inputs_dir / f"config_source{suffix}"
                _atomic_write_bytes(source_copy_path, src_p.read_bytes())
                source_copy_sha256 = file_sha256(source_copy_path)
        except Exception:
            source_copy_path = None
            source_copy_sha256 = None

    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "run_pairs_out": run_pairs_out,
        "config_snapshot": {
            "resolved_config_json": str(resolved_cfg_path),
            "resolved_config_sha256": resolved_cfg_sha256,
            "source_config_path": str(cfg_path) if cfg_path else None,
            "source_config_copy": str(source_copy_path) if source_copy_path else None,
            "source_config_copy_sha256": source_copy_sha256,
        },
    }


@dataclass
class RunStats:
    n_tickers: int
    n_initial_pairs: int
    n_candidates: int
    n_selected: int
    runtime_seconds: float
    seed: Optional[int] = None


def _coerce_ts_like_to_index_tz(ts_like: Any, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Parse a timestamp-like value and align it to the timezone domain of `index`."""
    ts = pd.Timestamp(ts_like)
    idx_tz = getattr(index, "tz", None)
    if idx_tz is None:
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("UTC").tz_localize(None)
    if ts.tzinfo is None:
        return ts.tz_localize(idx_tz)
    return ts.tz_convert(idx_tz)


def _resolve_analysis_time_boundary(
    da_cfg: dict[str, Any],
    *,
    preferred_key: str,
    legacy_key: str,
) -> Any:
    preferred = da_cfg.get(preferred_key, None)
    legacy = da_cfg.get(legacy_key, None)

    if preferred not in (None, ""):
        if legacy not in (None, "") and legacy != preferred:
            logger.warning(
                "Both data_analysis.%s and deprecated data_analysis.%s are set; using %s.",
                preferred_key,
                legacy_key,
                preferred_key,
            )
        return preferred

    if legacy not in (None, ""):
        logger.warning(
            "data_analysis.%s is deprecated; use data_analysis.%s.",
            legacy_key,
            preferred_key,
        )
        return legacy

    return None


def _resolve_max_candidates(da_cfg: dict[str, Any]) -> tuple[Optional[int], bool]:
    """
    Resolve candidate cap from config.

    Returns:
      (max_candidates, disabled)
    Rules:
      - disable_max_candidates=true disables capping.
      - max_candidates=None or <=0 also disables capping.
      - otherwise max_candidates must be int-like and >0.
    """
    if bool(da_cfg.get("disable_max_candidates", False)):
        return None, True

    raw_max = da_cfg.get("max_candidates", 500)
    if raw_max is None:
        return None, True

    try:
        max_cand = int(raw_max)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "data_analysis.max_candidates must be an integer, null, or <= 0 to disable."
        ) from e

    if max_cand <= 0:
        return None, True
    return max_cand, False


def main(
    cfg_path: str | Path | None = None,
    quick: bool = False,
    overrides: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    t0 = time.time()
    if cfg_path is None and overrides is not None:
        # Library-mode: treat overrides as the full config to keep the API deterministic and
        # independent of repo-local default YAMLs / environment variables.
        cfg = overrides
    else:
        cfg = load_config(cfg_path) if cfg_path is not None else load_config(None)
        if overrides:
            cfg = deep_merge(cfg, overrides)

    init_prometheus(cfg)
    timings: dict[str, float] = {}

    perf_cfg = cfg.get("perf", {}) or {}
    set_thread_limits(
        blas_threads=cast(Optional[int], perf_cfg.get("blas_threads", None)),
        numexpr_threads=cast(Optional[int], perf_cfg.get("numexpr_threads", None)),
    )

    da_cfg = cfg.get("data_analysis", {}) or {}
    if "use_mutliprocessing" in da_cfg and not da_cfg.get("use_multiprocessing", None):
        logger.warning(
            "Config key 'use_mutliprocessing' is deprecated; use 'use_multiprocessing' instead."
        )
    use_mp_flag = bool(
        da_cfg.get("use_multiprocessing", da_cfg.get("use_mutliprocessing", False))
    )
    start_method = str(da_cfg.get("mp_start_method", "spawn"))
    n_jobs_val = da_cfg.get("n_jobs", None)
    n_jobs = int(n_jobs_val) if n_jobs_val is not None else 0
    cpu_cnt = int(os.cpu_count() or 1)
    eff_workers = n_jobs if n_jobs > 0 else cpu_cnt

    bs_batch_size = int((da_cfg.get("bootstrap", {}) or {}).get("batch_size", 16))
    if bs_batch_size <= 0:
        bs_batch_size = 16

    data_cfg = cfg.get("data", {})
    default_primary_path = Path("backtest/data/filled_data.pkl")
    default_panel_path = Path("runs/data/processed/filled_prices_panel_exec.parquet")

    prices_path_cfg = data_cfg.get("prices_path")
    primary_path = (
        Path(prices_path_cfg).expanduser() if prices_path_cfg else default_primary_path
    )
    explicit_primary = prices_path_cfg is not None

    panel_path_cfg = data_cfg.get("panel_prices_path")
    panel_path = (
        Path(panel_path_cfg).expanduser() if panel_path_cfg else default_panel_path
    )

    prefer_panel = bool(data_cfg.get("prefer_panel_prices", False))
    panel_pref_mode = str(data_cfg.get("panel_preference_mode", "fallback")).lower()
    price_field_cfg = data_cfg.get("price_field")
    panel_field_cfg = data_cfg.get("panel_field", price_field_cfg or "close")

    selected_path = primary_path
    selected_field_preference: Optional[str]
    if isinstance(price_field_cfg, str) and price_field_cfg:
        selected_field_preference = price_field_cfg.lower()
    else:
        selected_field_preference = None

    prefer_panel_now = (
        prefer_panel
        and panel_path.exists()
        and (panel_pref_mode == "override" or panel_pref_mode == "fallback")
    )
    if prefer_panel_now:
        selected_path = panel_path
        if isinstance(panel_field_cfg, str) and panel_field_cfg:
            selected_field_preference = panel_field_cfg.lower()
        else:
            selected_field_preference = "close"
        logger.info(
            "Resolved prices_path to panel artefact %s (field=%s, mode=%s, prefer_panel_prices=%s)",
            selected_path,
            selected_field_preference,
            panel_pref_mode,
            prefer_panel,
        )
    elif not primary_path.exists() and not panel_path.exists():
        logger.warning(
            "Configured prices_path %s does not exist and panel_prices_path %s not found; attempting %s anyway.",
            primary_path,
            panel_path,
            primary_path,
        )
    elif explicit_primary and not primary_path.exists() and panel_path.exists():
        logger.warning(
            "Explicit prices_path %s not found; falling back to panel_prices_path %s",
            primary_path,
            panel_path,
        )
        selected_path = panel_path
        if isinstance(panel_field_cfg, str) and panel_field_cfg:
            selected_field_preference = panel_field_cfg.lower()
        else:
            selected_field_preference = "close"

    filled_path = selected_path
    pairs_out = Path(data_cfg.get("pairs_path", "backtest/data/filtered_pairs.pkl"))
    ts_col_cfg = data_cfg.get("timestamp_column")
    ts_unit = data_cfg.get("timestamp_unit")

    cfg_hash = dict_hash_sha256(cfg)
    run_artifacts = _prepare_run_artifacts(
        pairs_out=pairs_out,
        cfg_path=cfg_path,
        cfg=cfg,
        cfg_hash=cfg_hash,
    )
    run_pairs_out: Path = run_artifacts["run_pairs_out"]

    rolling_window = int(da_cfg.get("rolling_window", 504))
    rolling_step = int(da_cfg.get("rolling_step", 21))
    thr_corr = float(da_cfg.get("pair_corr_threshold", 0.8))
    thr1 = float(da_cfg.get("pct_thr1", 0.7))
    thr2 = float(da_cfg.get("pct_thr2", 0.8))
    pct_thr1_cut = float(
        da_cfg.get("pair_pct_threshold1", da_cfg.get("pair_pct_threshold", 0.7))
    )
    pct_thr2_cut = float(
        da_cfg.get("pair_pct_threshold2", da_cfg.get("pair_pct_threshold", 0.7))
    )
    mean_min_cfg = da_cfg.get("pair_mean_corr_min", None)
    if mean_min_cfg is not None:
        logger.warning(
            "data_analysis.pair_mean_corr_min is deprecated and ignored "
            "(paper-aligned selection relies on bootstrap/FDR + persistence thresholds)."
        )
    bs_cfg = da_cfg.get("bootstrap", {}) or {}
    n_resamples = int(bs_cfg.get("n_resamples", 200))
    null_mean = float(bs_cfg.get("null_mean_corr", 0.7))
    block_size = int(bs_cfg.get("block_size", 21))
    ci_level = float(bs_cfg.get("ci_level", 0.95))
    fdr_alpha = float(da_cfg.get("fdr_alpha", 0.05))
    enable_bootstrap = bool(da_cfg.get("enable_bootstrap", True))
    enable_hypothesis_test = bool(da_cfg.get("enable_hypothesis_test", True))
    enable_fdr = bool(da_cfg.get("enable_fdr", True))
    fdr_method = str(da_cfg.get("fdr_method", "BH")).upper()
    rng_seed = int(da_cfg.get("rng_seed", 42))
    max_candidates, max_candidates_disabled = _resolve_max_candidates(da_cfg)
    train_start = _resolve_analysis_time_boundary(
        da_cfg,
        preferred_key="train_start_local",
        legacy_key="train_start_utc",
    )
    train_cutoff = _resolve_analysis_time_boundary(
        da_cfg,
        preferred_key="train_cutoff_local",
        legacy_key="train_cutoff_utc",
    )
    train_start_ts = None
    train_cutoff_ts = None

    rcfg = da_cfg.get("returns_cleaning", {})
    rc_min_pos = float(rcfg.get("min_positive_frac", 0.99))
    rc_max_nan_c = float(rcfg.get("max_nan_frac_cols", 0.01))
    rc_row_pol = str(rcfg.get("drop_policy_rows", "any"))
    rc_min_std = float(rcfg.get("min_return_std", 0.0))

    pd_tol = float(da_cfg.get("pd_tol", 1e-12))
    pd_floor = float(da_cfg.get("pd_floor", 1e-8))
    persist_intermediates = bool(da_cfg.get("persist_intermediates", True))

    if quick:
        logger.info("QUICK mode active")
        n_resamples = min(50, n_resamples)
        if "rolling_window" not in da_cfg:
            rolling_window = min(252, rolling_window)
        if "rolling_step" not in da_cfg:
            rolling_step = max(5, rolling_step // 4)

    _guard(0.0 <= thr_corr <= 1.0, "pair_corr_threshold must be in [0,1]")
    _guard(
        0.0 <= thr1 <= 1.0 and 0.0 <= thr2 <= 1.0, "pct_thr1/pct_thr2 must be in [0,1]"
    )
    _guard(pd_floor > 0.0, "pd_floor must be > 0")
    _guard(pd_tol >= 0.0, "pd_tol must be >= 0")
    _guard(
        (0.0 <= pct_thr1_cut <= 1.0) or (pct_thr1_cut >= 50.0),
        "pair_pct_threshold1 in [0,1] (fraction) or as percentage >= 50",
    )
    _guard(
        (0.0 <= pct_thr2_cut <= 1.0) or (pct_thr2_cut >= 50.0),
        "pair_pct_threshold2 in [0,1] (fraction) or as percentage >= 50",
    )
    _guard(
        rolling_window >= 2 and rolling_step >= 1,
        "rolling_window >= 2, rolling_step >= 1",
    )
    _guard(rolling_step <= rolling_window, "rolling_step must be <= rolling_window")
    _guard(n_resamples >= 50, "n_resamples too small (>= 50 recommended)")
    _guard(0.0 < fdr_alpha <= 1.0, "fdr_alpha must be in (0,1]")
    _guard(block_size >= 1, "bootstrap.block_size must be >= 1")
    _guard(0.0 < ci_level < 1.0, "bootstrap.ci_level must be in (0,1)")
    _guard(fdr_method in {"BH", "BY"}, "fdr_method must be 'BH' or 'BY'")

    if not enable_bootstrap:
        if enable_hypothesis_test:
            logger.warning(
                "data_analysis.enable_bootstrap=false disables hypothesis testing as well."
            )
            enable_hypothesis_test = False
        if enable_fdr:
            logger.warning(
                "data_analysis.enable_bootstrap=false disables FDR adjustment as well."
            )
            enable_fdr = False
    elif not enable_hypothesis_test and enable_fdr:
        logger.warning(
            "data_analysis.enable_fdr=true ignored because enable_hypothesis_test=false."
        )
        enable_fdr = False

    np.random.seed(rng_seed)

    with _stage(timings, "load_prices"):
        df_prices = load_filled_data(
            filled_path,
            ts_col_cfg=ts_col_cfg,
            ts_unit=ts_unit,
            field_preference=selected_field_preference,
        )
    index_tz_str = str(df_prices.index.tz) if df_prices.index.tz is not None else None
    if train_start:
        with _stage(timings, "apply_train_start"):
            ts0 = _coerce_ts_like_to_index_tz(
                train_start, cast(pd.DatetimeIndex, df_prices.index)
            )
            train_start_ts = ts0
            df_prices = df_prices.loc[ts0:]
            logger.info(
                "Applied train_start=%s -> prices rows=%d",
                ts0.isoformat(),
                df_prices.shape[0],
            )
    if train_cutoff:
        with _stage(timings, "apply_cutoff"):
            ts = _coerce_ts_like_to_index_tz(
                train_cutoff, cast(pd.DatetimeIndex, df_prices.index)
            )
            train_cutoff_ts = ts
            df_prices = df_prices.loc[:ts]
            logger.info(
                "Applied train_cutoff=%s -> prices rows=%d",
                ts.isoformat(),
                df_prices.shape[0],
            )
    if train_start_ts is not None and train_cutoff_ts is not None:
        _guard(train_start_ts <= train_cutoff_ts, "train_start must be <= train_cutoff")
    _guard(
        not df_prices.empty, "No filled price data available after cutoff/start filters"
    )

    input_hash = file_sha256(filled_path) if filled_path.exists() else None

    with _stage(timings, "compute_returns"):
        pos_frac_before = (df_prices > 0).mean(axis=0)
        nan_frac_before = df_prices.isna().mean(axis=0)
        zero_count_before = (df_prices == 0).sum(axis=0)

        df_log = compute_log_returns(
            df_prices,
            min_positive_frac=rc_min_pos,
            max_nan_frac_cols=rc_max_nan_c,
            drop_policy_rows=rc_row_pol,
            min_return_std=rc_min_std,
        )
        _guard(not df_log.empty, "Log returns empty after preprocessing")

    _guard(
        df_log.shape[0] >= rolling_window,
        f"Not enough rows ({df_log.shape[0]}) for rolling_window={rolling_window}",
    )

    aligned_prices = (
        df_prices.loc[df_log.index, df_log.columns]
        if not df_log.empty
        else df_prices.iloc[0:0]
    )
    pos_frac_after = (aligned_prices > 0).mean(axis=0)
    nan_frac_after = aligned_prices.isna().mean(axis=0)
    zero_count_after = (aligned_prices == 0).sum(axis=0)
    dropped_cols_list = sorted(
        set(map(str, df_prices.columns)) - set(map(str, df_log.columns))
    )

    pct_cut1_abs = (pct_thr1_cut * 100.0) if pct_thr1_cut <= 1.0 else pct_thr1_cut
    pct_cut2_abs = (pct_thr2_cut * 100.0) if pct_thr2_cut <= 1.0 else pct_thr2_cut
    logger.info(
        "pair_pct_threshold1 evaluated as %.2f%% | pair_pct_threshold2 as %.2f%%",
        pct_cut1_abs,
        pct_cut2_abs,
    )

    with _stage(timings, "global_corr"):
        corr_shrink = compute_shrink_corr(df_log, pd_tol=pd_tol, pd_floor=pd_floor)
    with _stage(timings, "list_pairs"):
        pairs = list_high_pairs_vectorized(corr_shrink, threshold=thr_corr)
        initial_pairs_count = len(pairs)

    with _stage(timings, "rolling_metrics"):
        candidate_pairs = [(a, b) for a, b, _ in pairs]
        df_pair_summary, window_cors = rolling_pair_metrics_fast(
            df_log,
            candidate_pairs,
            window=rolling_window,
            step=rolling_step,
            thr1=thr1,
            thr2=thr2,
            pd_tol=pd_tol,
            pd_floor=pd_floor,
        )
    if not df_pair_summary.empty:
        k_windows_by_pair = {
            str(p): int(np.sum(~np.isnan(np.asarray(cors, dtype=float))))
            for p, cors in window_cors.items()
        }
        df_pair_summary["k_windows"] = (
            df_pair_summary["pair"]
            .astype(str)
            .map(k_windows_by_pair)
            .fillna(0)
            .astype(int)
        )
        df_pair_summary_valid = df_pair_summary.loc[
            df_pair_summary["k_windows"] > 0
        ].copy()
    else:
        df_pair_summary_valid = df_pair_summary.copy()

    try:
        n_windows = 0
        if window_cors:
            any_pair = next(iter(window_cors.values()))
            n_windows = int(len(any_pair))
    except Exception:
        n_windows = 0

    if df_pair_summary_valid.empty:
        runtime = time.time() - t0
        stats_empty = RunStats(
            n_tickers=df_prices.shape[1],
            n_initial_pairs=initial_pairs_count,
            n_candidates=0,
            n_selected=0,
            runtime_seconds=runtime,
            seed=rng_seed,
        )
        raw_rows = int(df_prices.shape[0] - 1)
        dropped_rows = max(0, raw_rows - int(df_log.shape[0]))
        dropped_cols = max(0, int(df_prices.shape[1] - df_log.shape[1]))
        meta_out: dict[str, Any] = {
            "analysis_schema_version": 1,
            "cfg_path": str(cfg_path) if cfg_path else None,
            "cfg_hash": cfg_hash,
            "git_sha": get_git_sha(),
            "run": {
                "run_id": run_artifacts["run_id"],
                "run_root": run_artifacts["run_root"],
            },
            "outputs": {
                "run_scoped_pairs_path": str(run_pairs_out),
                "latest_pairs_path": str(pairs_out),
            },
            "config": run_artifacts["config_snapshot"],
            "stats": asdict(stats_empty),
            "thresholds": {
                "corr": thr_corr,
                "pct_thr1_cut": pct_cut1_abs,
                "pct_thr2_cut": pct_cut2_abs,
                "pct_thr1": thr1,
                "pct_thr2": thr2,
                "null_mean": null_mean,
                "fdr_alpha": fdr_alpha,
            },
            "rolling": {"window": rolling_window, "step": rolling_step},
            "bootstrap": {
                "n_resamples": n_resamples,
                "seed": rng_seed,
                "block_size": block_size,
                "batch_size": bs_batch_size,
                "ci_level": ci_level,
            },
            "data_analysis": {
                "use_multiprocessing": use_mp_flag,
                "mp_start_method": start_method,
                "effective_workers": eff_workers,
                "n_jobs": n_jobs,
                "max_candidates": max_candidates,
                "max_candidates_disabled": max_candidates_disabled,
                "enable_bootstrap": enable_bootstrap,
                "enable_hypothesis_test": enable_hypothesis_test,
                "enable_fdr": enable_fdr,
                "fdr_method": fdr_method if enable_fdr else None,
                "significance_mode": (
                    f"fdr_{fdr_method.lower()}"
                    if enable_hypothesis_test and enable_fdr
                    else (
                        "pvalue"
                        if enable_hypothesis_test
                        else "disabled"
                    )
                ),
            },
            "inputs": [
                {
                    "prices_path": str(filled_path),
                    "sha256": input_hash,
                    "rows": int(df_prices.shape[0]),
                    "cols": int(df_prices.shape[1]),
                }
            ],
            "metrics": {
                "n_windows": 0,
                "dropped_rows_after_returns": dropped_rows,
                "dropped_cols_after_cleaning": dropped_cols,
            },
            "dq": {
                "before": _dq_summary(
                    pos_frac_before, nan_frac_before, zero_count_before
                ),
                "after": _dq_summary(
                    pos_frac_after, nan_frac_after, zero_count_after, dropped_cols_list
                ),
            },
            "time": {
                "tz_policy": "preserve_input_index_tz",
                "index_tz": index_tz_str,
                "train_start_config": train_start,
                "train_cutoff_config": train_cutoff,
                "train_start_effective": train_start_ts.isoformat()
                if train_start_ts
                else None,
                "train_cutoff_effective": train_cutoff_ts.isoformat()
                if train_cutoff_ts
                else None,
            },
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": __import__("sklearn").__version__,
                "scipy": __import__("scipy").__version__,
            },
            "timings": timings,
            "timestamp": time.time(),
        }
        # Immutable per-run snapshot + "latest mutable" (existing behavior).
        save_results(pd.DataFrame(), run_pairs_out, meta_out)
        if run_pairs_out.resolve() != pairs_out.resolve():
            save_results(pd.DataFrame(), pairs_out, meta_out)
        prom_observe(timings, 0)
        ka = int(
            ((cfg.get("monitoring") or {}).get("prometheus") or {}).get(
                "keep_alive_sec", 0
            )
        )
        if ka > 0 and _PROM and os.environ.get("PROM_KEEPALIVE") == "1":
            logger.info("Keeping Prometheus metrics server alive for %ds...", ka)
            time.sleep(ka)
        logger.info("No pairs found - wrote empty DataFrame")
        return pd.DataFrame()

    if max_candidates_disabled:
        candidates = df_pair_summary_valid
        logger.info(
            "Candidate cap disabled; evaluating all %d rolling-valid pairs.",
            len(candidates),
        )
    else:
        candidates = df_pair_summary_valid.head(cast(int, max_candidates))

    with _stage(timings, "bootstrap"):
        pvals: list[float] = [1.0] * len(candidates)
        rows_out: list[dict[str, Any]] = [{} for _ in range(len(candidates))]
        base_seed = rng_seed
        thr1_label = _pct_label(thr1)
        thr2_label = _pct_label(thr2)

        def _fill_row(i: int, pair_name: str, boot: dict[str, Any]) -> None:
            row_i = candidates.iloc[i]
            n_resamples_eff = boot.get("n_resamples_eff", n_resamples)
            block_size_eff = boot.get("block_size_eff", block_size)
            ci_level_eff = boot.get("ci_level", ci_level)
            pvals[i] = float(boot.get("pval", 1.0))
            rows_out[i] = {
                "pair": pair_name,
                "mean_corr_raw": float(row_i["mean_corr_raw"]),
                thr1_label + "_raw": float(row_i[f"{thr1_label}_raw"]),
                thr2_label + "_raw": float(row_i.get(f"{thr2_label}_raw", 0.0)),
                "mean_corr": float(row_i["mean_corr"]),
                thr1_label: float(row_i[thr1_label]),
                thr2_label: float(
                    row_i.get(thr2_label, row_i.get(f"{thr2_label}_raw", 0.0))
                ),
                "boot_mean": float(boot.get("mean_corr", np.nan)),
                "boot_mean_ci_low": float(
                    boot.get("mean_corr_ci", (np.nan, np.nan))[0]
                ),
                "boot_mean_ci_high": float(
                    boot.get("mean_corr_ci", (np.nan, np.nan))[1]
                ),
                "boot_pct1": float(boot.get("pct_above_thr1", np.nan)),
                "boot_pct1_ci_low": float(boot.get("pct1_ci", (np.nan, np.nan))[0]),
                "boot_pct1_ci_high": float(boot.get("pct1_ci", (np.nan, np.nan))[1]),
                "pval": float(boot.get("pval", 1.0)),
                "k_windows": int(boot.get("k_windows", 0) or 0),
                "n_resamples_eff": int(
                    n_resamples if n_resamples_eff is None else n_resamples_eff
                ),
                "block_size_eff": int(
                    block_size if block_size_eff is None else block_size_eff
                ),
                "ci_level": float(ci_level if ci_level_eff is None else ci_level_eff),
            }

        def _default_boot_for_idx(i: int) -> dict[str, Any]:
            row_i = candidates.iloc[i]
            return {
                "mean_corr": float("nan"),
                "pct_above_thr1": float(row_i.get(f"{thr1_label}_raw", 0.0)),
                "mean_corr_ci": (np.nan, np.nan),
                "pct1_ci": (np.nan, np.nan),
                "pval": 1.0,
                "k_windows": 0,
                "n_resamples_eff": int(n_resamples),
                "block_size_eff": int(block_size),
                "ci_level": float(ci_level),
            }

        def _bootstrap_disabled_for_idx(i: int) -> dict[str, Any]:
            row_i = candidates.iloc[i]
            cors = window_cors.get(str(row_i["pair"]))
            k_windows = (
                int(np.sum(~np.isnan(np.asarray(cors, dtype=float))))
                if cors is not None
                else 0
            )
            return {
                "mean_corr": float("nan"),
                "pct_above_thr1": float(row_i.get(f"{thr1_label}_raw", np.nan)),
                "mean_corr_ci": (np.nan, np.nan),
                "pct1_ci": (np.nan, np.nan),
                "pval": np.nan,
                "k_windows": k_windows,
                "n_resamples_eff": 0,
                "block_size_eff": 0,
                "ci_level": float(ci_level),
            }

        def _do_sequential() -> None:
            for i in range(len(candidates)):
                row = candidates.iloc[i]
                pair_name = str(row["pair"])
                cors = window_cors.get(pair_name)
                if cors is None or not np.any(~np.isnan(np.asarray(cors))):
                    boot = _default_boot_for_idx(i)
                else:
                    h = int(
                        hashlib.sha256(
                            (pair_name + str(base_seed)).encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    seed = (base_seed ^ h) & 0x7FFFFFFF
                    boot = bootstrap_pair_stats(
                        np.asarray(cors, dtype=float),
                        thr1=thr1,
                        null_mean=null_mean,
                        n_resamples=n_resamples,
                        rng_seed=seed,
                        block_size=block_size,
                        ci_level=ci_level,
                    )
                _fill_row(i, pair_name, boot)

        def _do_parallel_batched() -> None:
            try:
                mp_ctx = mp.get_context(start_method)
            except ValueError:
                mp_ctx = mp.get_context()

            max_workers = max(1, int(eff_workers))
            if max_workers <= 1 or len(candidates) <= 1:
                _do_sequential()
                return

            blas_thr_cfg = cast(Optional[int], perf_cfg.get("blas_threads", None))
            blas_thr_workers = blas_thr_cfg if blas_thr_cfg is not None else 1
            ne_thr = cast(Optional[int], perf_cfg.get("numexpr_threads", None))

            tasks: list[tuple[int, str, Optional[np.ndarray]]] = []
            for i in range(len(candidates)):
                row = candidates.iloc[i]
                pair_name = str(row["pair"])
                cors = window_cors.get(pair_name)
                tasks.append(
                    (
                        i,
                        pair_name,
                        np.asarray(cors, dtype=float) if cors is not None else None,
                    )
                )

            batches: list[list[tuple[int, str, Optional[np.ndarray]]]] = [
                tasks[i : i + bs_batch_size]
                for i in range(0, len(tasks), bs_batch_size)
            ]

            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
                initializer=_worker_init_for_mp,
                initargs=(blas_thr_workers, ne_thr),
            ) as ex:
                futs: list[Any] = []
                fut2batch: dict[Any, list[tuple[int, str, Optional[np.ndarray]]]] = {}
                for batch in batches:
                    fut = ex.submit(
                        _bootstrap_batch_worker,
                        batch,
                        thr1,
                        null_mean,
                        n_resamples,
                        base_seed,
                        block_size,
                        ci_level,
                    )
                    futs.append(fut)
                    fut2batch[fut] = batch

                for fut in as_completed(futs):
                    try:
                        results = fut.result()
                    except Exception as e:
                        batch = fut2batch.get(fut, [])
                        pair_preview = (
                            ", ".join(p for _, p, _ in batch[:5]) or "<unknown>"
                        )
                        raise RuntimeError(
                            "Bootstrap batch failed for "
                            f"{len(batch)} pair(s): {pair_preview}"
                        ) from e
                    for idx_i, pair_name_i, boot in results:
                        if 0 <= idx_i < len(candidates):
                            _fill_row(idx_i, pair_name_i, boot)

            for i in range(len(candidates)):
                if not rows_out[i]:
                    _fill_row(
                        i, str(candidates.iloc[i]["pair"]), _default_boot_for_idx(i)
                    )

        if not enable_bootstrap:
            for i in range(len(candidates)):
                pair_name = str(candidates.iloc[i]["pair"])
                _fill_row(i, pair_name, _bootstrap_disabled_for_idx(i))
        elif use_mp_flag:
            _do_parallel_batched()
        else:
            _do_sequential()

    with _stage(timings, "fdr_select"):
        thr2_label = _pct_label(thr2)
        significance_mode = "disabled"
        if enable_hypothesis_test:
            if enable_fdr:
                decisions = (
                    benjamini_yekutieli(pvals, alpha=fdr_alpha)
                    if fdr_method == "BY"
                    else benjamini_hochberg(pvals, alpha=fdr_alpha)
                )
                significance_mode = f"fdr_{fdr_method.lower()}"
            else:
                decisions = [
                    bool(np.isfinite(p)) and float(p) <= float(fdr_alpha)
                    for p in pvals
                ]
                significance_mode = "pvalue"
        else:
            decisions = [True] * len(rows_out)
        selected: list[dict[str, Any]] = []
        selected_mask = [False] * len(rows_out)
        thr1_label = _pct_label(thr1)
        for i, (dec, r) in enumerate(zip(decisions, rows_out)):
            if not dec:
                continue
            if r[thr1_label + "_raw"] < pct_cut1_abs:
                continue
            if r.get(thr2_label + "_raw", pct_cut2_abs) < pct_cut2_abs:
                continue
            selected_mask[i] = True
            selected.append(
                {
                    "pair": r["pair"],
                    "mean_corr": round(r["mean_corr"], 4),
                    thr1_label: round(r[thr1_label], 2),
                    thr2_label: round(
                        r.get(thr2_label, r.get(thr2_label + "_raw", 0.0)), 2
                    ),
                    "pval": round(r["pval"], 6),
                    "ci_mean_low": r["boot_mean_ci_low"],
                    "ci_mean_high": r["boot_mean_ci_high"],
                }
            )
        df_selected = pd.DataFrame(
            selected,
            columns=[
                "pair",
                "mean_corr",
                thr1_label,
                thr2_label,
                "pval",
                "ci_mean_low",
                "ci_mean_high",
            ],
        )
        if not df_selected.empty:
            df_selected = df_selected.sort_values(
                "mean_corr", ascending=False
            ).reset_index(drop=True)

    runtime = time.time() - t0
    stats = RunStats(
        n_tickers=df_prices.shape[1],
        n_initial_pairs=initial_pairs_count,
        n_candidates=len(candidates),
        n_selected=len(df_selected),
        runtime_seconds=runtime,
        seed=rng_seed,
    )

    try:
        n_windows = 0
        if window_cors:
            any_pair = next(iter(window_cors.values()))
            n_windows = int(len(any_pair))
    except Exception:
        n_windows = 0

    raw_rows = int(df_prices.shape[0] - 1)
    dropped_rows = max(0, raw_rows - int(df_log.shape[0]))
    dropped_cols = max(0, int(df_prices.shape[1] - df_log.shape[1]))

    meta: dict[str, Any] = {
        "analysis_schema_version": 1,
        "cfg_path": str(cfg_path) if cfg_path else None,
        "cfg_hash": cfg_hash,
        "git_sha": get_git_sha(),
        "run": {
            "run_id": run_artifacts["run_id"],
            "run_root": run_artifacts["run_root"],
        },
        "outputs": {
            "run_scoped_pairs_path": str(run_pairs_out),
            "latest_pairs_path": str(pairs_out),
        },
        "config": run_artifacts["config_snapshot"],
        "stats": asdict(stats),
        "thresholds": {
            "corr": thr_corr,
            "pct_thr1_cut": pct_cut1_abs,
            "pct_thr2_cut": pct_cut2_abs,
            "pct_thr1": thr1,
            "pct_thr2": thr2,
            "null_mean": null_mean,
            "fdr_alpha": fdr_alpha,
            "pd_tol": pd_tol,
            "pd_floor": pd_floor,
        },
        "rolling": {"window": rolling_window, "step": rolling_step},
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": rng_seed,
            "block_size": block_size,
            "batch_size": bs_batch_size,
            "ci_level": ci_level,
        },
        "data_analysis": {
                "use_multiprocessing": use_mp_flag,
                "mp_start_method": start_method,
                "effective_workers": eff_workers,
                "n_jobs": n_jobs,
                "max_candidates": max_candidates,
                "max_candidates_disabled": max_candidates_disabled,
                "persist_intermediates": persist_intermediates,
                "enable_bootstrap": enable_bootstrap,
                "enable_hypothesis_test": enable_hypothesis_test,
                "enable_fdr": enable_fdr,
                "fdr_method": fdr_method if enable_fdr else None,
                "significance_mode": significance_mode,
            },
        "inputs": [
            {
                "prices_path": str(filled_path),
                "sha256": input_hash,
                "rows": int(df_prices.shape[0]),
                "cols": int(df_prices.shape[1]),
            }
        ],
        "metrics": {
            "n_windows": n_windows,
            "dropped_rows_after_returns": dropped_rows,
            "dropped_cols_after_cleaning": dropped_cols,
        },
        "dq": {
            "before": _dq_summary(pos_frac_before, nan_frac_before, zero_count_before),
            "after": _dq_summary(
                pos_frac_after, nan_frac_after, zero_count_after, dropped_cols_list
            ),
        },
        "time": {
            "tz_policy": "preserve_input_index_tz",
            "index_tz": index_tz_str,
            "train_start_config": train_start,
            "train_cutoff_config": train_cutoff,
            "train_start_effective": train_start_ts.isoformat()
            if train_start_ts
            else None,
            "train_cutoff_effective": train_cutoff_ts.isoformat()
            if train_cutoff_ts
            else None,
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "timings": timings,
        "timestamp": time.time(),
    }

    with _stage(timings, "save_artifacts"):
        # Immutable per-run snapshot + "latest mutable" (existing behavior).
        save_results(df_selected, run_pairs_out, meta)
        if run_pairs_out.resolve() != pairs_out.resolve():
            save_results(df_selected, pairs_out, meta)

        if persist_intermediates:
            out_dir = run_pairs_out.parent
            artifacts: dict[str, Any] = {}

            # 1) Dense return matrix R (paper Step 1 output).
            returns_path = out_dir / "returns_dense.parquet"
            save_parquet(df_log, returns_path, compression="zstd")
            artifacts["returns_dense"] = {
                "path": str(returns_path),
                "sha256": file_sha256(returns_path),
            }

            # 2) Global candidate pairs P after static correlation thresholding (paper Step 3).
            df_global_pairs = pd.DataFrame(pairs, columns=["left", "right", "corr"])
            global_pairs_path = out_dir / "pairs_global.parquet"
            save_parquet(df_global_pairs, global_pairs_path, compression="zstd")
            artifacts["pairs_global"] = {
                "path": str(global_pairs_path),
                "sha256": file_sha256(global_pairs_path),
                "rows": int(df_global_pairs.shape[0]),
            }

            # 3) Rolling summary statistics for all global candidates (paper Step 4 output summary).
            rolling_summary_path = out_dir / "rolling_summary.parquet"
            save_parquet(df_pair_summary, rolling_summary_path, compression="zstd")
            artifacts["rolling_summary"] = {
                "path": str(rolling_summary_path),
                "sha256": file_sha256(rolling_summary_path),
                "rows": int(df_pair_summary.shape[0]),
            }

            # 4) Rolling window metadata (paper Step 4 windows).
            windows = [
                (s, s + rolling_window)
                for s in range(0, len(df_log) - rolling_window + 1, rolling_step)
            ]
            window_rows: list[dict[str, Any]] = []
            for wi, (s, e) in enumerate(windows):
                start_ts = df_log.index[s]
                end_ts = df_log.index[e - 1]
                window_rows.append(
                    {
                        "window_idx": wi,
                        "start_pos": int(s),
                        "end_pos_exclusive": int(e),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "n_obs": int(e - s),
                    }
                )
            windows_df = pd.DataFrame(window_rows)
            windows_path = out_dir / "rolling_windows.parquet"
            save_parquet(windows_df, windows_path, compression="zstd")
            artifacts["rolling_windows"] = {
                "path": str(windows_path),
                "sha256": file_sha256(windows_path),
            }

            # 5) Top-M list P_boot (paper Step 5) + rolling series ρ_p(k) used for bootstrap.
            boot_candidates_path = out_dir / "pairs_boot_candidates.parquet"
            save_parquet(candidates, boot_candidates_path, compression="zstd")
            artifacts["pairs_boot_candidates"] = {
                "path": str(boot_candidates_path),
                "sha256": file_sha256(boot_candidates_path),
                "rows": int(candidates.shape[0]),
            }

            if not candidates.empty and window_rows:
                cand_names = candidates["pair"].astype(str).tolist()
                cors_stack = np.vstack(
                    [
                        np.asarray(
                            window_cors.get(name, np.full(n_windows, np.nan)),
                            dtype=float,
                        )
                        for name in cand_names
                    ]
                )
                rolling_idx = pd.Index(
                    [r["end_ts"] for r in window_rows], name="window_end_ts"
                )
                df_rolling = pd.DataFrame(
                    cors_stack.T, index=rolling_idx, columns=cand_names
                )
            else:
                df_rolling = pd.DataFrame()
            rolling_cors_path = out_dir / "rolling_cors_candidates.parquet"
            save_parquet(df_rolling, rolling_cors_path, compression="zstd")
            artifacts["rolling_cors_candidates"] = {
                "path": str(rolling_cors_path),
                "sha256": file_sha256(rolling_cors_path),
                "shape": list(df_rolling.shape),
            }

            # 6) Bootstrap results + FDR decisions (paper Steps 6–8).
            df_boot = pd.DataFrame(rows_out)
            df_boot["significance_reject"] = list(map(bool, decisions))
            df_boot["fdr_reject"] = list(map(bool, decisions))
            df_boot["selected"] = list(map(bool, selected_mask))
            bootstrap_path = out_dir / "bootstrap_results.parquet"
            save_parquet(df_boot, bootstrap_path, compression="zstd")
            artifacts["bootstrap_results"] = {
                "path": str(bootstrap_path),
                "sha256": file_sha256(bootstrap_path),
                "rows": int(df_boot.shape[0]),
            }

            save_json(artifacts, out_dir / "artifacts_index.json")
            artifacts["artifacts_index"] = {
                "path": str(out_dir / "artifacts_index.json"),
                "sha256": file_sha256(out_dir / "artifacts_index.json"),
            }
            meta["artifacts"] = artifacts

    meta["timings"] = dict(timings)
    try:
        # Keep meta JSON in sync for both output locations.
        for out_p in (run_pairs_out, pairs_out):
            meta_path = out_p.with_suffix(".meta.json")
            _atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))
    except Exception:
        logger.debug("Meta JSON rewrite failed", exc_info=True)

    prom_observe(timings, len(df_selected))
    logger.info("Saved filtered pairs: %d (runtime=%.1fs)", len(df_selected), runtime)

    ka = int(
        ((cfg.get("monitoring") or {}).get("prometheus") or {}).get("keep_alive_sec", 0)
    )
    if ka > 0 and _PROM and os.environ.get("PROM_KEEPALIVE") == "1":
        logger.info("Keeping Prometheus metrics server alive for %ds...", ka)
        time.sleep(ka)

    return df_selected


__all__ = ["RunStats", "main"]
