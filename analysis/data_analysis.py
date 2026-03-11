#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Data Analysis for pair filtering (modularized facade).

The pipeline, numerics, bootstrap/FDR, I/O, and Prometheus logic now live in
dedicated modules to keep responsibilities narrow while preserving the original
public API and CLI behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from analysis import numerics as _numerics
from analysis.atomic_io import (
    _atomic_write_bytes,
    _atomic_write_text,
    _fsync_dir,
    save_results,
)
from analysis.bootstrap_fdr import (
    _stationary_bootstrap_indices,
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
from analysis.constants import PRICE_COL_CANDIDATES
from analysis.logging_config import logger
from analysis.mp_helpers import (
    _bootstrap_batch_worker,
    _bootstrap_worker,
    _worker_init_for_mp,
)
from analysis.numerics import _nearest_positive_definite, compute_log_returns
from analysis.pairs import (
    hierarchical_clustering_from_corr,
    list_high_pairs_vectorized,
    pairs_df_from_corr,
)
from analysis.pipeline import RunStats, main
from analysis.preprocess import (
    _dedup_str_columns,
    ensure_utc_index,
    load_filled_data,
    select_price_columns,
)
from analysis.prometheus_metrics import (
    _PROM,
    _PROM_REG,
    _PROM_RUN_ID,
    init_prometheus,
    prom_observe,
)
from analysis.rolling import rolling_pair_metrics_fast
from analysis.threading_control import set_thread_limits
from analysis.utils import _canon_pair, _guard, _pct_label, _stage, parse_pair


def compute_shrink_corr(
    returns,
    fix_pd: bool = True,
    *,
    pd_tol: float = 1e-12,
    pd_floor: float = 1e-8,
):
    return _numerics.compute_shrink_corr(
        returns,
        fix_pd=fix_pd,
        pd_fix_fn=_nearest_positive_definite,
        pd_tol=pd_tol,
        pd_floor=pd_floor,
    )


__all__ = [
    "PRICE_COL_CANDIDATES",
    "_PROM_REG",
    "_PROM",
    "_PROM_RUN_ID",
    "init_prometheus",
    "prom_observe",
    "deep_merge",
    "file_sha256",
    "dict_hash_sha256",
    "get_git_sha",
    "load_config",
    "set_thread_limits",
    "_guard",
    "_stage",
    "_pct_label",
    "_canon_pair",
    "ensure_utc_index",
    "_dedup_str_columns",
    "select_price_columns",
    "load_filled_data",
    "_nearest_positive_definite",
    "compute_shrink_corr",
    "compute_log_returns",
    "list_high_pairs_vectorized",
    "pairs_df_from_corr",
    "hierarchical_clustering_from_corr",
    "rolling_pair_metrics_fast",
    "_stationary_bootstrap_indices",
    "bootstrap_pair_stats",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "parse_pair",
    "_atomic_write_bytes",
    "_atomic_write_text",
    "save_results",
    "_fsync_dir",
    "np",
    "Path",
    "LedoitWolf",
    "_worker_init_for_mp",
    "_bootstrap_worker",
    "_bootstrap_batch_worker",
    "RunStats",
    "main",
    "_parse_cli",
]


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data analysis runner for pair filtering")
    p.add_argument("--cfg", type=str, default=None, help="Path to YAML config")
    p.add_argument(
        "--quick", action="store_true", help="Quick/dev mode (fewer resamples)"
    )
    p.add_argument(
        "--override-prices", type=str, help="Override data.prices_path in config"
    )
    p.add_argument(
        "--override-out", type=str, help="Override data.pairs_path in config"
    )
    p.add_argument("--n-jobs", type=int, help="Override data_analysis.n_jobs")
    p.add_argument(
        "--max-candidates", type=int, help="Override data_analysis.max_candidates"
    )
    p.add_argument(
        "--disable-max-candidates",
        action="store_true",
        help="Disable candidate capping (ignore max_candidates)",
    )
    p.add_argument(
        "--n-resamples", type=int, help="Override data_analysis.bootstrap.n_resamples"
    )
    p.add_argument(
        "--disable-bootstrap",
        action="store_true",
        help="Disable bootstrap statistics and downstream significance gating",
    )
    p.add_argument(
        "--disable-hypothesis-test",
        action="store_true",
        help="Disable p-value-based hypothesis testing",
    )
    p.add_argument(
        "--disable-fdr",
        action="store_true",
        help="Disable FDR adjustment and use unadjusted p-values when hypothesis testing stays enabled",
    )
    p.add_argument("--dry-run", action="store_true", help="Show merged config & exit")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    logger.setLevel(getattr(logging, args.log_level.upper(), logger.level))
    overrides_cfg: dict[str, Any] = {}
    if args.override_prices:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data": {"prices_path": args.override_prices}}
        )
    if args.override_out:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data": {"pairs_path": args.override_out}}
        )
    if args.n_jobs is not None:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"n_jobs": args.n_jobs}}
        )
    if args.max_candidates is not None:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"max_candidates": args.max_candidates}}
        )
    if args.disable_max_candidates:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"disable_max_candidates": True}}
        )
    if args.n_resamples is not None:
        overrides_cfg = deep_merge(
            overrides_cfg,
            {"data_analysis": {"bootstrap": {"n_resamples": args.n_resamples}}},
        )
    if args.disable_bootstrap:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"enable_bootstrap": False}}
        )
    if args.disable_hypothesis_test:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"enable_hypothesis_test": False}}
        )
    if args.disable_fdr:
        overrides_cfg = deep_merge(
            overrides_cfg, {"data_analysis": {"enable_fdr": False}}
        )

    if args.dry_run:
        try:
            cfg_preview = load_config(args.cfg) if args.cfg else load_config(None)
            cfg_preview = deep_merge(cfg_preview, overrides_cfg)
            print(json.dumps(cfg_preview, indent=2, default=str))
        except Exception:
            logger.exception("Failed to load/merge config for dry-run")
        raise SystemExit(0)

    try:
        main(cfg_path=args.cfg, quick=args.quick, overrides=overrides_cfg)
    except Exception:
        logger.exception("data_analysis failed", exc_info=True)
        raise
