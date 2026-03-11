from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from analysis.config_io import deep_merge as deep_merge
from analysis.config_io import load_config as da_load_config
from analysis.pipeline import main as da_main

logger = logging.getLogger("runner_analysis")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analysis Runner (pair filtering)")
    p.add_argument("--cfg", type=str, default=None, help="Path to YAML/JSON config")
    p.add_argument(
        "--quick", action="store_true", help="Quick/dev mode (fewer resamples)"
    )

    # Common overrides
    p.add_argument(
        "--override-prices", type=str, default=None, help="Override data.prices_path"
    )
    p.add_argument(
        "--override-out", type=str, default=None, help="Override data.pairs_path"
    )
    p.add_argument(
        "--n-jobs", type=int, default=None, help="Override data_analysis.n_jobs"
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Override data_analysis.max_candidates",
    )
    p.add_argument(
        "--disable-max-candidates",
        action="store_true",
        help="Disable candidate capping (ignores max_candidates)",
    )
    p.add_argument(
        "--n-resamples", type=int, default=None, help="Override bootstrap n_resamples"
    )
    p.add_argument(
        "--block-size", type=int, default=None, help="Override bootstrap block_size"
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
    p.add_argument(
        "--train-cutoff-local",
        type=str,
        default=None,
        help="Override data_analysis.train_cutoff_local (input index timezone, typically America/New_York)",
    )
    p.add_argument(
        "--train-cutoff-utc",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    # Data-quality overrides for returns cleaning
    p.add_argument(
        "--min-positive-frac",
        type=float,
        default=None,
        help="returns_cleaning.min_positive_frac",
    )
    p.add_argument(
        "--max-nan-frac-cols",
        type=float,
        default=None,
        help="returns_cleaning.max_nan_frac_cols",
    )
    p.add_argument(
        "--drop-policy-rows",
        type=str,
        default=None,
        help="returns_cleaning.drop_policy_rows",
    )

    p.add_argument(
        "--dry-run", action="store_true", help="Print merged config and exit"
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args(argv)


def _load_yaml_like(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _resolve_cfg_path(cfg_arg: str | None) -> tuple[Path | None, str]:
    if cfg_arg:
        p = Path(cfg_arg)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        return p, str(p)

    for env_key in ("ANALYSIS_CONFIG", "BACKTEST_CONFIG", "STRAT_CONFIG"):
        v = os.getenv(env_key)
        if v:
            p = Path(v)
            if not p.exists():
                raise FileNotFoundError(f"{env_key} points to missing config: {p}")
            return p, f"{env_key}={p}"

    # Local dev default (used by tests)
    cfg_dir = Path.cwd() / "runs" / "configs"
    for name in ("config_analysis.yaml", "config.yaml"):
        local = cfg_dir / name
        if local.exists():
            return local, str(local)

    return None, "<analysis.load_config()>"


def resolve_full_yaml(cfg_arg: str | None) -> tuple[dict[str, Any], str]:
    cfg_path, src = _resolve_cfg_path(cfg_arg)
    if cfg_path is not None:
        return _load_yaml_like(cfg_path), src
    return da_load_config(None), src


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    if getattr(args, "override_prices", None):
        overrides = deep_merge(
            overrides, {"data": {"prices_path": args.override_prices}}
        )
    if getattr(args, "override_out", None):
        overrides = deep_merge(overrides, {"data": {"pairs_path": args.override_out}})

    if getattr(args, "n_jobs", None) is not None:
        overrides = deep_merge(
            overrides, {"data_analysis": {"n_jobs": int(args.n_jobs)}}
        )
    if getattr(args, "max_candidates", None) is not None:
        overrides = deep_merge(
            overrides, {"data_analysis": {"max_candidates": int(args.max_candidates)}}
        )
    if bool(getattr(args, "disable_max_candidates", False)):
        overrides = deep_merge(
            overrides, {"data_analysis": {"disable_max_candidates": True}}
        )

    bs: dict[str, Any] = {}
    if getattr(args, "n_resamples", None) is not None:
        bs["n_resamples"] = int(args.n_resamples)
    if getattr(args, "block_size", None) is not None:
        bs["block_size"] = int(args.block_size)
    if bs:
        overrides = deep_merge(overrides, {"data_analysis": {"bootstrap": bs}})

    da_flags: dict[str, Any] = {}
    if bool(getattr(args, "disable_bootstrap", False)):
        da_flags["enable_bootstrap"] = False
    if bool(getattr(args, "disable_hypothesis_test", False)):
        da_flags["enable_hypothesis_test"] = False
    if bool(getattr(args, "disable_fdr", False)):
        da_flags["enable_fdr"] = False
    if da_flags:
        overrides = deep_merge(overrides, {"data_analysis": da_flags})

    train_cutoff_local = getattr(args, "train_cutoff_local", None)
    train_cutoff_legacy = getattr(args, "train_cutoff_utc", None)
    if train_cutoff_local:
        overrides = deep_merge(
            overrides, {"data_analysis": {"train_cutoff_local": train_cutoff_local}}
        )
    elif train_cutoff_legacy:
        logger.warning("--train-cutoff-utc is deprecated; use --train-cutoff-local.")
        overrides = deep_merge(
            overrides, {"data_analysis": {"train_cutoff_local": train_cutoff_legacy}}
        )

    rc: dict[str, Any] = {}
    if getattr(args, "min_positive_frac", None) is not None:
        rc["min_positive_frac"] = float(args.min_positive_frac)
    if getattr(args, "max_nan_frac_cols", None) is not None:
        rc["max_nan_frac_cols"] = float(args.max_nan_frac_cols)
    if getattr(args, "drop_policy_rows", None):
        rc["drop_policy_rows"] = str(args.drop_policy_rows)
    if rc:
        overrides = deep_merge(overrides, {"data_analysis": {"returns_cleaning": rc}})

    return overrides


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv) if argv is not None else parse_args()
    logger.setLevel(getattr(logging, str(args.log_level).upper(), logger.level))

    try:
        base_cfg, cfg_src = resolve_full_yaml(args.cfg)
        cfg_path, _src2 = _resolve_cfg_path(args.cfg)
    except Exception:
        logger.exception("Failed to resolve config")
        return 2

    overrides = build_overrides(args)
    merged_cfg = deep_merge(base_cfg, overrides) if overrides else base_cfg

    if args.dry_run:
        print(json.dumps(merged_cfg, indent=2, default=str))
        logger.info("Config source: %s", cfg_src)
        return 0

    try:
        effective_overrides: dict[str, Any] | None
        if cfg_path is None:
            # If config was resolved via analysis.load_config() fallback, pass the fully merged
            # configuration to keep runtime behavior consistent with --dry-run.
            effective_overrides = merged_cfg
        else:
            effective_overrides = overrides or None
        da_main(
            cfg_path=str(cfg_path) if cfg_path is not None else None,
            quick=bool(args.quick),
            overrides=effective_overrides,
        )
    except Exception:
        logger.exception("Analysis pipeline failed")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
