from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from backtest.config.cfg import AppConfig, config_to_dict, load_config
from backtest.optimize.runner import load_bo_trials
from backtest.reporting.report_bundle import (
    debug_dir as report_debug_dir,
    write_core_report,
)
from backtest.runner.portfolio import write_pnl_concentration_report
from backtest.runner.runtime import (
    limit_runtime_pairs,
    load_runtime_context,
)
from backtest.runner.single import run_single_backtest
from backtest.runner.walkforward_run import run_walkforward_backtest
from backtest.utils.io import write_json
from backtest.utils.tz import utc_now

logger = logging.getLogger("backtest.runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _train_refits_for_report(art: Any) -> list[dict[str, Any]]:
    if art.train_refit is None:
        return []
    refit = art.train_refit
    items: list[dict[str, Any]] = [
        {
            "train_start": refit.train_start,
            "train_end": refit.train_end,
            "n_pairs": refit.n_pairs,
            "wf_i": refit.wf_i,
            "stats": refit.stats,
            "trades": refit.trades,
            "equity": refit.equity,
            "summary": refit.summary,
        }
    ]
    return items


def run(cfg: AppConfig, *, out_dir: Path, quick: bool = False) -> dict[str, Any]:
    cfg_base = cfg
    reporting_cfg = cfg.reporting
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_context(cfg_base, out_dir=out_dir)
    cfg_base = runtime.cfg
    prices_panel = runtime.prices_panel
    prices = runtime.prices
    pairs = runtime.pairs
    adv_map = runtime.adv_map
    write_json(out_dir / "config_effective.json", config_to_dict(cfg_base))

    if quick and len(pairs) > 250:
        runtime = limit_runtime_pairs(runtime, limit=250)
        prices_panel = runtime.prices_panel
        prices = runtime.prices
        pairs = runtime.pairs
        adv_map = runtime.adv_map
        cfg_base = runtime.cfg
        logger.info("Quick mode: limiting pairs to %d", len(pairs))

    if bool(cfg_base.backtest.walkforward.enabled):
        return run_walkforward_backtest(
            cfg_base=cfg_base,
            out_dir=out_dir,
            reporting_cfg=reporting_cfg,
            runtime=runtime,
            quick=quick,
        )

    debug_out = report_debug_dir(out_dir) if reporting_cfg.debug_enabled else None
    art = run_single_backtest(
        cfg_eff=cfg_base,
        out_dir=out_dir,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        quick=quick,
        return_raw_trades=False,
        debug_out=debug_out,
    )
    write_json(out_dir / "config_effective.json", config_to_dict(art.cfg_eff))
    report = write_core_report(
        out_dir,
        reporting_cfg=reporting_cfg,
        test_eq=art.test_equity,
        test_trades=art.trades,
        train_refits=_train_refits_for_report(art),
        cv_scores=art.bo_run.selected_cv_scores,
        bo_trials=load_bo_trials(art.bo_run.bo_out),
        window_rows=None,
    )
    write_pnl_concentration_report(Path(report["report_dir"]), art.trades)
    return {
        "out_dir": str(out_dir),
        "report_dir": report["report_dir"],
        "n_pairs": art.n_pairs,
        "n_trades": art.n_trades,
    }


def _default_out_dir(_cfg_path: Path) -> Path:
    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return Path("runs/results/performance") / f"BT-{ts}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backtest runner (lob or light execution)"
    )
    parser.add_argument(
        "--cfg", type=Path, default=Path("runs/configs/config_backtest.yaml")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: runs/results/performance/BT-...)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer pairs / smaller windows)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logger.setLevel(getattr(logging, str(args.log_level).upper(), logger.level))

    if not args.cfg.exists():
        logger.error("Config not found: %s", args.cfg)
        return 2

    cfg = load_config(args.cfg)
    out_dir = args.out or _default_out_dir(args.cfg)

    try:
        res = run(cfg, out_dir=out_dir, quick=bool(args.quick))
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        return 1

    logger.info("Done: %s", res.get("out_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
