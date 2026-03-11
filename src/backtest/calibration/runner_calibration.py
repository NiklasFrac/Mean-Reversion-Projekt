"""
Calibration runner for the LOB liquidity model (free-data heuristic).

This runner performs a non-circular parameter search over `execution.lob.liq_model`
using external plausibility targets (stylized facts / broad bounds).

Usage:
  python -m backtest.runner_calibration --cfg runs/configs/config_backtest.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import yaml

from backtest.borrow.context import build_borrow_context
from backtest.calibration.metrics import (
    derive_exec_metrics,
    report_by_adv_decile,
    report_by_participation,
    summarize_metrics,
)
from backtest.calibration.score import DEFAULT_TARGETS, ScoreBreakdown, score_trades
from backtest.utils.common.io import load_yaml_dict as _load_yaml
from backtest.utils.common.merge import deep_merge as _deep_merge
from backtest.utils.common.prices import as_price_map as _as_price_map
from backtest.config.yaml_cfg_extractors import _extract_execution_cfg_for_yaml
from backtest.loader import prepare_pairs_data
from backtest.utils.run.data import _pair_prefilter_inputs
from backtest.run.runtime import load_runtime_context
from backtest.utils.run.strategy import _build_strategy
from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg
from backtest.windowing.eval import (
    remap_cfg_for_named_eval,
    slice_frame,
    walkforward_window_splits,
)
from backtest.windowing.walkforward import generate_walkforward_windows_from_cfg
from backtest.utils.tz import to_naive_local, utc_now

logger = logging.getLogger("backtest.runner_calibration")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _default_out_dir() -> Path:
    ts = utc_now().strftime("CAL-%Y%m%dT%H%M%SZ")
    return Path("runs/results/calibration") / ts


def _liq_model_patch(
    cfg: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, Any]:
    _ = cfg
    return {"execution": {"lob": {"liq_model": dict(params)}}}


def _extract_base_liq_model_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ex_norm = _extract_execution_cfg_for_yaml(cfg)
    lob = ex_norm.get("lob")
    if not isinstance(lob, Mapping):
        return {}
    liq = lob.get("liq_model")
    return dict(liq) if isinstance(liq, Mapping) else {}


def _patch_liq_model(cfg: dict[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    patch = _liq_model_patch(cfg, params)
    return _deep_merge(cfg, patch)


def _coerce_windows(
    cfg: Mapping[str, Any], prices_idx: pd.DatetimeIndex
) -> dict[str, dict[str, Any]]:
    """
    Resolve calibration windows.

    If cfg['calibration']['splits'] exists, use it. Otherwise fall back to backtest.splits:
      - analysis <- backtest.splits.analysis
      - train   <- backtest.splits.train
      - val     <- backtest.splits.test
      - test    <- [backtest.splits.test.end + 1 session, prices_idx.max()]
    """
    bt = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    bt_splits = bt.get("splits") if isinstance(bt.get("splits"), dict) else {}
    wf = bt.get("walkforward", {}) if isinstance(bt.get("walkforward"), dict) else {}

    cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration"), dict) else {}
    splits = cal.get("splits") if isinstance(cal.get("splits"), dict) else {}
    if isinstance(splits, dict) and all(
        k in splits for k in ("analysis", "train", "val", "test")
    ):
        return cast(dict[str, dict[str, Any]], splits)

    # If walkforward is enabled and explicit backtest.splits are missing, synthesize them from the dataset calendar.
    if (not bt_splits) and bool(wf.get("enabled", False)):
        cal_idx = (
            pd.DatetimeIndex(pd.to_datetime(prices_idx, errors="coerce"))
            .dropna()
            .sort_values()
            .unique()
        )
        windows, _ = generate_walkforward_windows_from_cfg(calendar=cal_idx, cfg=cfg)
        if not windows:
            raise KeyError(
                "walkforward.enabled but produced 0 windows; cannot infer calibration windows."
            )
        widx = int((cal.get("wf_window", 0) or 0))
        widx = int(max(0, min(widx, len(windows) - 1)))
        w = windows[widx]
        bt_splits = walkforward_window_splits(cal_idx, w)

    if (
        not isinstance(bt_splits, dict)
        or "train" not in bt_splits
        or "test" not in bt_splits
    ):
        raise KeyError("Need backtest.splits.{train,test} (or calibration.splits)")

    # Default: reuse existing windows, then extend a holdout to the end of the dataset.
    analysis = bt_splits.get("analysis") or {
        "start": str(prices_idx.min().date()),
        "end": str(prices_idx.min().date()),
    }
    train = bt_splits["train"]
    val = bt_splits["test"]

    # Define a test window after val.end until last available date.
    last = to_naive_local(prices_idx.max())
    v_end = pd.Timestamp(val["end"])
    v_end = to_naive_local(v_end)
    test_start = (v_end + pd.Timedelta(days=1)).date().isoformat()
    test_end = pd.Timestamp(last).date().isoformat()

    return {
        "analysis": dict(analysis),
        "train": dict(train),
        "val": dict(val),
        "test": {"start": str(test_start), "end": str(test_end)},
    }


@dataclass(frozen=True)
class Trial:
    params: dict[str, Any]
    score: float
    score_breakdown: ScoreBreakdown
    train_summary: dict[str, Any]
    val_summary: dict[str, Any]


def _iter_grid(rng: np.random.Generator, *, n: int) -> list[dict[str, Any]]:
    """
    Produce a compact but expressive grid (randomized) over the most important parameters.
    """
    depth = np.array([2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3], dtype=float)
    spread_adv = np.array([6.0, 10.0, 15.0, 22.0, 32.0], dtype=float)
    spread_floor = np.array([0.25, 0.5, 0.75, 1.0], dtype=float)
    gamma = np.array([0.7, 1.0, 1.3], dtype=float)

    combos: list[dict[str, Any]] = []
    for _ in range(int(max(1, n))):
        combos.append(
            {
                "depth_frac_of_adv_shares": float(rng.choice(depth)),
                "spread_adv_mult": float(rng.choice(spread_adv)),
                "spread_floor_bps": float(rng.choice(spread_floor)),
                "depth_gamma": float(rng.choice(gamma)),
            }
        )
    # de-dup
    seen = set()
    out: list[dict[str, Any]] = []
    for c in combos:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate LOB liquidity model (free-data)"
    )
    ap.add_argument(
        "--cfg", type=Path, default=Path("runs/configs/config_backtest.yaml")
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--trials", type=int, default=20, help="Number of randomized grid trials"
    )
    ap.add_argument(
        "--quick", action="store_true", help="Reduce pairs for faster iteration"
    )
    ap.add_argument(
        "--wf-window",
        type=int,
        default=None,
        help="Walkforward window index (overrides calibration.wf_window)",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument(
        "--seed",
        type=int,
        default=424242,
        help="Calibration RNG seed (independent of backtest seed)",
    )
    ap.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Override backtest.initial_capital for calibration runs (strategy sizing depends on this).",
    )
    args = ap.parse_args(argv)

    logger.setLevel(getattr(logging, str(args.log_level).upper(), logger.level))

    if not args.cfg.exists():
        logger.error("Config not found: %s", args.cfg)
        return 2

    cfg_in = _load_yaml(args.cfg)
    cfg = dict(cfg_in)
    if args.capital is not None:
        cfg = _deep_merge(cfg, {"backtest": {"initial_capital": float(args.capital)}})
    if args.wf_window is not None:
        cal_cfg = (
            dict(cfg.get("calibration") or {})
            if isinstance(cfg.get("calibration"), dict)
            else {}
        )
        cal_cfg["wf_window"] = int(args.wf_window)
        cfg["calibration"] = cal_cfg

    out_dir = args.out or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_in.json").write_text(
        json.dumps(cfg_in, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    runtime = load_runtime_context(cfg, out_dir=out_dir)
    cfg = runtime.cfg
    prices_panel = runtime.prices_panel
    prices_close = runtime.prices
    pairs = runtime.pairs
    adv_map = runtime.adv_map
    (out_dir / "config_effective.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    cal_splits = _coerce_windows(cfg, cast(pd.DatetimeIndex, prices_close.index))
    (out_dir / "calibration_splits.json").write_text(
        json.dumps(cal_splits, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    cfg_strategy = dict(cfg)
    bt_strategy = (
        dict(cfg_strategy.get("backtest") or {})
        if isinstance(cfg_strategy.get("backtest"), dict)
        else {}
    )
    strategy_splits = (
        bt_strategy.get("splits") if isinstance(bt_strategy.get("splits"), dict) else {}
    )
    if (
        not isinstance(strategy_splits, dict)
        or "train" not in strategy_splits
        or "test" not in strategy_splits
    ):
        synth_train = dict(cal_splits.get("analysis") or cal_splits.get("train") or {})
        train_start = (cal_splits.get("train") or {}).get("start")
        test_end = (cal_splits.get("test") or {}).get("end")
        if not (
            synth_train.get("start")
            and synth_train.get("end")
            and train_start
            and test_end
        ):
            raise KeyError(
                "Cannot synthesize backtest.splits for calibration strategy bootstrap."
            )
        bt_strategy["splits"] = {
            "train": synth_train,
            "test": {"start": str(train_start), "end": str(test_end)},
        }
        cfg_strategy["backtest"] = bt_strategy
        logger.info(
            "Calibration: synthesized strategy splits train=[%s..%s], test=[%s..%s]",
            bt_strategy["splits"]["train"]["start"],
            bt_strategy["splits"]["train"]["end"],
            bt_strategy["splits"]["test"]["start"],
            bt_strategy["splits"]["test"]["end"],
        )

    # Build a single cached portfolio to avoid re-running strategy per trial.
    # Use analysis.start .. test.end slice (max window) so rolling features are available.
    a0 = cal_splits["analysis"]["start"]
    t_end = cal_splits["test"]["end"]
    logger.info("Preparing cached portfolio on slice [%s .. %s]", a0, t_end)
    prices_close_slice = slice_frame(prices_close, start=a0, end=t_end)

    disable_prefilter, _prefilter_range = _pair_prefilter_inputs(cfg)

    pairs_data = prepare_pairs_data(
        prices_close_slice,
        pairs,
        adv_map=adv_map,
        verbose=False,
        disable_prefilter=disable_prefilter,
        attach_prices_df=True,
        prefilter_range=_prefilter_range,
    )
    if args.quick and len(pairs_data) > 40:
        keys = list(pairs_data.keys())[:40]
        pairs_data = {k: pairs_data[k] for k in keys}
        logger.info("Quick mode: limiting pairs_data to %d pairs", len(pairs_data))

    borrow_ctx = build_borrow_context(cfg_strategy)
    strat = _build_strategy(cfg_strategy, borrow_ctx=borrow_ctx)
    portfolio = strat(pairs_data)

    # Trials
    rng = np.random.default_rng(int(args.seed))
    grid = _iter_grid(rng, n=int(args.trials))
    logger.info("Running %d calibration trials", len(grid))

    trials: list[Trial] = []

    def _run_eval(eval_split: str, cfg_trial: dict[str, Any]) -> pd.DataFrame:
        cfg_eval = remap_cfg_for_named_eval(
            cfg_trial,
            splits=cal_splits,
            eval_split=eval_split,
            order={
                "train": ("analysis", "train"),
                "val": ("train", "val"),
                "test": ("val", "test"),
            },
        )
        stats, trades = backtest_portfolio_with_yaml_cfg(
            portfolio=portfolio,
            price_data=_as_price_map(prices_close_slice),
            market_data_panel=prices_panel,
            adv_map=adv_map,
            yaml_cfg=cfg_eval,
            borrow_ctx=borrow_ctx,
        )
        _ = stats  # stats currently unused for calibration; keep for future extensions
        return trades

    base_liq = _extract_base_liq_model_cfg(cfg)

    for i, patch_params in enumerate(grid, start=1):
        params = dict(base_liq) if isinstance(base_liq, dict) else {}
        params.update(patch_params)
        cfg_trial = _patch_liq_model(cfg, params)

        logger.info("Trial %d/%d: %s", i, len(grid), patch_params)
        try:
            tr_trades = _run_eval("train", cfg_trial)
            va_trades = _run_eval("val", cfg_trial)
        except Exception as e:
            logger.warning("Trial failed: %s", e)
            bd = ScoreBreakdown(
                score=1e9, penalties={"exception": 1e9}, notes={"error": str(e)}
            )
            trials.append(
                Trial(
                    params=params,
                    score=bd.score,
                    score_breakdown=bd,
                    train_summary={"n_trades": 0},
                    val_summary={"n_trades": 0},
                )
            )
            continue

        tr_df = derive_exec_metrics(tr_trades)
        va_df = derive_exec_metrics(va_trades)

        # Score = train score + 0.5*val score (stability)
        bd_tr = score_trades(tr_df, targets=DEFAULT_TARGETS)
        bd_va = score_trades(va_df, targets=DEFAULT_TARGETS)
        score = float(bd_tr.score + 0.5 * bd_va.score)
        bd = ScoreBreakdown(
            score=score,
            penalties={
                "train": float(bd_tr.score),
                "val_half": float(0.5 * bd_va.score),
                **{f"tr.{k}": float(v) for k, v in bd_tr.penalties.items()},
                **{f"va.{k}": float(v) for k, v in bd_va.penalties.items()},
            },
            notes={"train": bd_tr.notes, "val": bd_va.notes},
        )

        tr_sum = summarize_metrics(
            tr_df, outlier_bps_threshold=DEFAULT_TARGETS.outlier_bps_threshold
        ).__dict__
        va_sum = summarize_metrics(
            va_df, outlier_bps_threshold=DEFAULT_TARGETS.outlier_bps_threshold
        ).__dict__

        trials.append(
            Trial(
                params=params,
                score=score,
                score_breakdown=bd,
                train_summary=tr_sum,
                val_summary=va_sum,
            )
        )

    # Select best
    trials_sorted = sorted(trials, key=lambda t: float(t.score))
    best = trials_sorted[0] if trials_sorted else None
    if best is None:
        logger.error("No successful trials")
        return 1

    # Write trials table
    rows: list[dict[str, Any]] = []
    for t in trials_sorted:
        row: dict[str, Any] = {"score": float(t.score)}
        row.update({f"p.{k}": v for k, v in t.params.items()})
        row.update({f"train.{k}": v for k, v in t.train_summary.items()})
        row.update({f"val.{k}": v for k, v in t.val_summary.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "trials.csv", index=False)

    # Best artifacts
    (out_dir / "score_breakdown.json").write_text(
        json.dumps(
            best.score_breakdown.__dict__, indent=2, ensure_ascii=False, default=str
        ),
        encoding="utf-8",
    )
    (out_dir / "best_patch.yaml").write_text(
        yaml.safe_dump(_liq_model_patch(cfg, best.params), sort_keys=False),
        encoding="utf-8",
    )

    # Validation on holdout test
    logger.info("Validating best params on holdout test split")
    cfg_best = _patch_liq_model(cfg, best.params)
    te_trades = _run_eval("test", cfg_best)
    te_df = derive_exec_metrics(te_trades)

    # Reports
    for name, dfm in (
        ("train", derive_exec_metrics(_run_eval("train", cfg_best))),
        ("val", derive_exec_metrics(_run_eval("val", cfg_best))),
        ("test", te_df),
    ):
        report_by_adv_decile(dfm).to_csv(
            out_dir / f"report_{name}_by_adv_decile.csv", index=False
        )
        report_by_participation(dfm).to_csv(
            out_dir / f"report_{name}_by_participation.csv", index=False
        )
        summary = summarize_metrics(
            dfm, outlier_bps_threshold=DEFAULT_TARGETS.outlier_bps_threshold
        ).__dict__
        (out_dir / f"summary_{name}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    logger.info("Calibration done: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
