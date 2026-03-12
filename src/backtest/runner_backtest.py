from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from backtest import pit_guard as _pit_guard
from backtest.borrow.context import build_borrow_context
from backtest.risk_policy import build_risk_policy
from backtest.reporting.report_bundle import (
    debug_dir as _debug_dir,
    load_reporting_config,
    write_core_report,
)
from backtest.reporting.tearsheet import summarize_stats
from backtest.utils.reporting import equity_from_stats as _equity_from_stats
from backtest.utils.run import strategy as _strategy_utils
from backtest.run.bo_runner import BORunResult, run_bo_if_enabled
from backtest.utils.run.data import _as_price_map
from backtest.utils.run.overfit import _write_walkforward_overfit_summary
from backtest.run.pit_guard import _pit_guard_window_runner
from backtest.run.runtime import (
    build_runtime_calendar,
    limit_runtime_pairs,
    load_runtime_context,
)
from backtest.run.train_only import (
    TrainRefitArtifacts,
    run_train_refit,
    write_train_refit_debug,
)
from backtest.utils.run.trades import (
    _apply_global_positions_ledger,
    _collect_portfolio_intents,
    _portfolio_from_trades,
)
from backtest.run.window_execution import (
    execute_window_backtest,
    prepare_pairs_data_for_cfg,
)
from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg
from backtest.simulators.stateful import rescale_trades_stateful
from backtest.utils.tz import align_ts_to_index, utc_now
from backtest.windowing.walkforward import generate_walkforward_windows_from_cfg


def _missing_pit_guard(name: str):
    raise ImportError(f"backtest.validation.pit_guard missing required symbol: {name}")


assert_no_future_dependency = getattr(
    _pit_guard,
    "assert_no_future_dependency",
    lambda **_k: _missing_pit_guard("assert_no_future_dependency"),
)
assert_no_future_dependency_walkforward = getattr(
    _pit_guard,
    "assert_no_future_dependency_walkforward",
    lambda **_k: _missing_pit_guard("assert_no_future_dependency_walkforward"),
)
pit_guard_config_from_cfg = getattr(
    _pit_guard,
    "pit_guard_config_from_cfg",
    lambda *_a, **_k: _missing_pit_guard("pit_guard_config_from_cfg"),
)
sanitize_cfg_for_pit = getattr(
    _pit_guard,
    "sanitize_cfg_for_pit",
    lambda *_a, **_k: _missing_pit_guard("sanitize_cfg_for_pit"),
)

logger = logging.getLogger("backtest.runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_validate_cfg_strict = _strategy_utils._validate_cfg_strict
_build_strategy = _strategy_utils._build_strategy


@dataclass(frozen=True)
class SingleRunArtifacts:
    cfg_eff: dict[str, Any]
    bo_run: BORunResult
    pairs_data: dict[str, Any]
    borrow_ctx: Any
    stats: pd.DataFrame
    trades: pd.DataFrame
    raw_trades: pd.DataFrame
    orders: pd.DataFrame
    test_equity: pd.Series
    test_summary: dict[str, Any]
    train_refit: TrainRefitArtifacts | None
    portfolio: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs_data))

    @property
    def n_trades(self) -> int:
        return int(len(self.trades)) if isinstance(self.trades, pd.DataFrame) else 0


def _align_ts_to_index_tz(ts: Any, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return align_ts_to_index(ts, idx)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_debug_window(
    debug_out: Path,
    *,
    art: SingleRunArtifacts,
    splits: Mapping[str, Any] | None = None,
) -> None:
    debug_out.mkdir(parents=True, exist_ok=True)
    _json_dump(debug_out / "config_effective.json", art.cfg_eff)
    art.stats.to_csv(debug_out / "stats.csv", index=False)
    art.trades.to_csv(debug_out / "trades.csv", index=False)
    entry_intents = art.stats.attrs.get("entry_intents_df")
    if not isinstance(entry_intents, pd.DataFrame) or entry_intents.empty:
        if (
            isinstance(art.raw_trades, pd.DataFrame)
            and not art.raw_trades.empty
            and "signal_date" in art.raw_trades.columns
        ):
            entry_intents = art.raw_trades
    if isinstance(entry_intents, pd.DataFrame) and not entry_intents.empty:
        entry_intents.to_csv(debug_out / "entry_intents.csv", index=False)
    state_transitions = art.stats.attrs.get("state_transitions_df")
    if isinstance(state_transitions, pd.DataFrame) and not state_transitions.empty:
        state_transitions.to_csv(debug_out / "state_transitions.csv", index=False)
    if isinstance(art.orders, pd.DataFrame) and not art.orders.empty:
        art.orders.to_csv(debug_out / "orders.csv", index=False)
    if splits is not None:
        _json_dump(debug_out / "walkforward_window.json", {"splits": dict(splits)})
    if art.bo_run.bo_res is not None:
        bo_debug = debug_out / "bo"
        bo_debug.mkdir(parents=True, exist_ok=True)
        _json_dump(bo_debug / "bo_best.json", art.bo_run.bo_res)
        if (
            art.bo_run.selected_cv_scores is not None
            and not art.bo_run.selected_cv_scores.empty
        ):
            art.bo_run.selected_cv_scores.to_csv(
                bo_debug / "selected_cv_scores.csv", index=False
            )
        if art.bo_run.bo_out is not None:
            _copy_if_exists(
                art.bo_run.bo_out / "bo_trials.csv", bo_debug / "bo_trials.csv"
            )
            _copy_if_exists(
                art.bo_run.bo_out / "bo_best.json", bo_debug / "bo_best_source.json"
            )
    if art.train_refit is not None:
        write_train_refit_debug(debug_out / "train_refit", art.train_refit)
    try:
        from backtest.reporting.pnl_breakdown import generate_pnl_breakdown

        generate_pnl_breakdown({}, debug_out / "performance")
    except Exception:
        logger.warning("Debug pnl breakdown failed.", exc_info=True)


def _run_optional_overfit(
    *,
    cfg_eff: Mapping[str, Any],
    out_dir: Path,
    bo_run: BORunResult,
    eq: pd.Series,
) -> None:
    of_cfg = (
        cfg_eff.get("overfit", {}) if isinstance(cfg_eff.get("overfit"), dict) else {}
    )
    if not bool(of_cfg.get("enabled", False)):
        return
    try:
        from backtest.overfit import analyze_bo_trials

        trials_list: list[str] = []
        if bo_run.bo_out is not None:
            trials_path = bo_run.bo_out / "bo_trials.csv"
            if trials_path.exists():
                trials_list = [str(trials_path)]
        if not trials_list:
            return
        aggregate = str(of_cfg.get("aggregate", "median")).strip().lower()
        if aggregate not in {"median", "mean", "max"}:
            aggregate = "median"
        component_filter = of_cfg.get(
            "component_filter", of_cfg.get("components", None)
        )
        if component_filter is None:
            bo_res = bo_run.bo_res if isinstance(bo_run.bo_res, Mapping) else {}
            theta_markov = (
                bo_res.get("theta_markov_hat")
                if isinstance(bo_res.get("theta_markov_hat"), Mapping)
                else {}
            )
            theta_sig = (
                bo_res.get("theta_sig_hat")
                if isinstance(bo_res.get("theta_sig_hat"), Mapping)
                else {}
            )
            if theta_markov:
                component_filter = "theta_markov"
            elif theta_sig:
                component_filter = "theta_sig"
        out_relpath = str(of_cfg.get("out_relpath", "overfit_summary.json"))
        out_path = Path(out_relpath)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
        analyze_bo_trials(
            trials_list,
            aggregate=cast(Any, aggregate),
            equity_curve=eq,
            out_path=out_path,
            trading_days=int(of_cfg.get("trading_days", 252)),
            component_filter=component_filter,
            metric_filter=of_cfg.get("metric_filter", of_cfg.get("metric", None)),
        )
    except Exception:
        logger.warning("Overfit analysis failed.", exc_info=True)


def _portfolio_has_intents(portfolio: Mapping[str, Any] | None) -> bool:
    intents = _collect_portfolio_intents(portfolio)
    return isinstance(intents, pd.DataFrame) and not intents.empty


def _namespace_intent_portfolio(
    portfolio: Mapping[str, Any] | None,
    *,
    wf_i: int,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    prefix = f"WF-{int(wf_i):03d}::"
    for key, meta in (portfolio or {}).items():
        if not isinstance(meta, Mapping):
            continue
        meta_local = dict(meta)
        intents = meta.get("intents")
        if isinstance(intents, pd.DataFrame):
            meta_local["intents"] = intents.copy()
        state = meta.get("state")
        if isinstance(state, Mapping):
            state_local = dict(state)
            state_local["window_id"] = int(wf_i)
            meta_local["state"] = state_local
        out[f"{prefix}{key}"] = meta_local
    return out


def _run_once(
    *,
    cfg_eff: dict[str, Any],
    out_dir: Path,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    quick: bool = False,
    return_raw_trades: bool = False,
    debug_out: Path | None = None,
) -> SingleRunArtifacts:
    _validate_cfg_strict(cfg_eff)

    pairs_data = prepare_pairs_data_for_cfg(
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        cfg=cfg_eff,
        adv_map=adv_map,
    )
    bo_run = run_bo_if_enabled(
        cfg_eff=cfg_eff,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        out_dir=out_dir,
        quick=quick,
        pairs_data=pairs_data,
        persist_quick_budget=True,
    )
    cfg_eff = bo_run.cfg_eff
    pairs_data = bo_run.pairs_data or pairs_data

    window_run = execute_window_backtest(
        cfg=cfg_eff,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        adv_map=adv_map,
        pairs_data=pairs_data,
    )
    stats = window_run.stats
    trades = window_run.trades
    raw_trades = (
        window_run.raw_trades
        if return_raw_trades or debug_out is not None
        else pd.DataFrame()
    )
    eq = _equity_from_stats(stats)
    test_summary_df = summarize_stats(eq, trades_df=trades)
    test_summary = (
        test_summary_df.to_dict(orient="records")[0]
        if not test_summary_df.empty
        else {}
    )

    train_refit = None
    try:
        train_refit = run_train_refit(
            cfg_eff=cfg_eff,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=window_run.pairs_data,
            adv_map=adv_map,
            borrow_ctx=window_run.borrow_ctx,
        )
    except Exception:
        logger.warning("Train refit failed.", exc_info=True)

    art = SingleRunArtifacts(
        cfg_eff=cfg_eff,
        bo_run=bo_run,
        pairs_data=window_run.pairs_data,
        borrow_ctx=window_run.borrow_ctx,
        stats=stats,
        trades=trades,
        raw_trades=raw_trades,
        orders=window_run.orders,
        test_equity=eq,
        test_summary=test_summary,
        train_refit=train_refit,
        portfolio=getattr(window_run, "portfolio", {}) or {},
    )

    if debug_out is not None:
        _write_debug_window(debug_out, art=art)
    _run_optional_overfit(cfg_eff=cfg_eff, out_dir=out_dir, bo_run=bo_run, eq=eq)
    return art


def run(cfg: dict[str, Any], *, out_dir: Path, quick: bool = False) -> dict[str, Any]:
    cfg_base = dict(cfg)
    reporting_cfg = load_reporting_config(cfg_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = load_runtime_context(cfg_base, out_dir=out_dir)
    cfg_base = runtime.cfg
    prices_panel = runtime.prices_panel
    prices = runtime.prices
    pairs = runtime.pairs
    adv_map = runtime.adv_map
    _json_dump(out_dir / "config_effective.json", cfg_base)

    if quick and len(pairs) > 250:
        runtime = limit_runtime_pairs(runtime, limit=250)
        prices_panel = runtime.prices_panel
        prices = runtime.prices
        pairs = runtime.pairs
        adv_map = runtime.adv_map
        cfg_base = runtime.cfg
        logger.info("Quick mode: limiting pairs to %d", len(pairs))

    pg = pit_guard_config_from_cfg(cfg_base)
    if pg.enabled:
        cfg_pit = sanitize_cfg_for_pit(cfg_base, pg)
        borrow_ctx_pit = build_borrow_context(cfg_pit)
        availability_long = (
            getattr(borrow_ctx_pit, "availability_long", None)
            if borrow_ctx_pit is not None
            else None
        )
        bt_pit = (
            cfg_pit.get("backtest", {})
            if isinstance(cfg_pit.get("backtest"), dict)
            else {}
        )
        wf_pit = (
            bt_pit.get("walkforward", {})
            if isinstance(bt_pit.get("walkforward"), dict)
            else {}
        )
        mode = str(pg.mode or "auto").strip().lower()
        if mode == "auto":
            mode = "walkforward" if bool(wf_pit.get("enabled", False)) else "single"
        if mode == "walkforward":
            assert_no_future_dependency_walkforward(
                prices=prices,
                prices_panel=prices_panel,
                pairs=pairs,
                cfg=cfg_pit,
                runner=cast(Any, _pit_guard_window_runner),
                adv_map=adv_map,
                borrow_ctx=borrow_ctx_pit,
                availability_long=availability_long,
                availability_scope=pg.availability_scope,
                pg=pg,
            )
        else:
            assert_no_future_dependency(
                prices=prices,
                prices_panel=prices_panel,
                pairs=pairs,
                cfg=cfg_pit,
                runner=cast(Any, _pit_guard_window_runner),
                adv_map=adv_map,
                borrow_ctx=borrow_ctx_pit,
                availability_long=availability_long,
                availability_scope=pg.availability_scope,
                pg=pg,
            )

    bt = (
        cfg_base.get("backtest", {})
        if isinstance(cfg_base.get("backtest"), dict)
        else {}
    )
    wf = bt.get("walkforward", {}) if isinstance(bt.get("walkforward"), dict) else {}
    wf_enabled = bool(wf.get("enabled", False))

    if not wf_enabled:
        debug_out = _debug_dir(out_dir) if reporting_cfg.debug_enabled else None
        art = _run_once(
            cfg_eff=dict(cfg_base),
            out_dir=out_dir,
            prices=prices,
            prices_panel=prices_panel,
            pairs=pairs,
            adv_map=adv_map,
            quick=quick,
            return_raw_trades=False,
            debug_out=debug_out,
        )
        _json_dump(out_dir / "config_effective.json", art.cfg_eff)
        report = write_core_report(
            out_dir,
            reporting_cfg=reporting_cfg,
            test_eq=art.test_equity,
            test_trades=art.trades
            if isinstance(art.trades, pd.DataFrame)
            else pd.DataFrame(),
            train_refits=[cast(Mapping[str, Any], art.train_refit)]
            if art.train_refit is not None
            else [],
            cv_scores=art.bo_run.selected_cv_scores,
            window_rows=None,
        )
        return {
            "out_dir": str(out_dir),
            "report_dir": report["report_dir"],
            "n_pairs": art.n_pairs,
            "n_trades": art.n_trades,
        }

    calendar = build_runtime_calendar(runtime)
    windows, wf_meta = generate_walkforward_windows_from_cfg(
        calendar=calendar, cfg=cfg_base
    )
    if not windows:
        raise ValueError(
            "Walkforward enabled but produced 0 windows (check backtest.range and months params)."
        )

    debug_root = _debug_dir(out_dir) if reporting_cfg.debug_enabled else None
    if debug_root is not None:
        _json_dump(
            debug_root / "walkforward_plan.json",
            {"meta": wf_meta, "windows": [w.as_dict() for w in windows]},
        )

    carry_trades_parts: list[pd.DataFrame] = []
    carry_intent_parts: list[pd.DataFrame] = []
    carry_intent_portfolio: dict[str, Any] = {}
    use_global_intent_sim = False
    wf_sizing_params: dict[int, dict[str, float]] = {}
    global_test_start = pd.Timestamp(windows[0].test_start)
    global_test_end = pd.Timestamp(windows[-1].test_end)
    bt0 = (
        cfg_base.get("backtest", {})
        if isinstance(cfg_base.get("backtest"), dict)
        else {}
    )
    rolling_capital = float(bt0.get("initial_capital", 1_000_000.0))

    window_rows: list[dict[str, Any]] = []
    train_refits: list[Mapping[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []

    for w in windows:
        wf_i = int(w.i)
        cfg_eff = deepcopy(cfg_base)
        cfg_eff.setdefault("backtest", {})
        bt2 = (
            cfg_eff.get("backtest", {})
            if isinstance(cfg_eff.get("backtest"), dict)
            else {}
        )
        splits = w.as_splits()
        test_split = splits.get("test", {})
        if isinstance(test_split, dict):
            test_split = dict(test_split)
            test_split["entry_end"] = str(w.test_end.date())
            test_split["exit_end"] = str(global_test_end.date())
            splits["test"] = test_split
        bt2["splits"] = splits
        bt2["initial_capital"] = float(rolling_capital)
        cfg_eff["backtest"] = bt2

        risk_cfg = (
            cfg_eff.get("risk", {}) if isinstance(cfg_eff.get("risk"), dict) else {}
        )
        exec_cfg = (
            cfg_eff.get("execution", {})
            if isinstance(cfg_eff.get("execution"), dict)
            else {}
        )
        sizing_policy = build_risk_policy(
            risk_cfg=risk_cfg,
            backtest_cfg=bt2 if isinstance(bt2, dict) else {},
            execution_cfg=exec_cfg,
        ).sizing
        wf_sizing_params[wf_i] = {
            "risk_per_trade": float(sizing_policy.risk_per_trade),
            "max_trade_pct": float(sizing_policy.max_trade_pct),
            "max_participation": float(sizing_policy.max_participation),
        }

        with tempfile.TemporaryDirectory(prefix=f"wf_{wf_i:03d}_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            debug_out = (
                debug_root / f"WF-{wf_i:03d}" if debug_root is not None else None
            )
            art = _run_once(
                cfg_eff=cfg_eff,
                out_dir=tmp_dir,
                prices=prices,
                prices_panel=prices_panel,
                pairs=pairs,
                adv_map=adv_map,
                quick=quick,
                return_raw_trades=True,
                debug_out=debug_out,
            )
            if debug_out is not None:
                _json_dump(
                    debug_out / "walkforward_window.json",
                    {"i": wf_i, "splits": splits, "truncated": bool(w.truncated)},
                )

            cap_start = float(rolling_capital)
            cap_end = cap_start
            if isinstance(art.test_equity, pd.Series) and not art.test_equity.empty:
                cap_end = float(
                    pd.to_numeric(art.test_equity.iloc[-1], errors="coerce")
                )
                if pd.notna(cap_end) and cap_end > 0.0:
                    rolling_capital = float(cap_end)
                else:
                    cap_end = cap_start

            if _portfolio_has_intents(art.portfolio):
                use_global_intent_sim = True
                carry_intent_portfolio.update(
                    _namespace_intent_portfolio(art.portfolio, wf_i=wf_i)
                )
                intents_raw = _collect_portfolio_intents(art.portfolio)
                if isinstance(intents_raw, pd.DataFrame) and not intents_raw.empty:
                    intents_raw = intents_raw.copy()
                    intents_raw.insert(0, "wf_i", wf_i)
                    carry_intent_parts.append(intents_raw)
            elif isinstance(art.raw_trades, pd.DataFrame) and not art.raw_trades.empty:
                raw = art.raw_trades.copy()
                raw.insert(0, "wf_i", wf_i)
                carry_trades_parts.append(raw)

            row = {
                "wf_i": wf_i,
                "train_start": str(w.train_start.date()),
                "train_end": str(w.train_end.date()),
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "truncated": bool(w.truncated),
                "n_pairs": art.n_pairs,
                "n_trades": art.n_trades,
                "capital_start": float(cap_start),
                "capital_end": float(cap_end),
                "capital_return": float(
                    (cap_end / cap_start - 1.0) if cap_start > 0 else 0.0
                ),
            }
            row.update(art.test_summary)
            window_rows.append(row)

            if art.train_refit is not None:
                train_refits.append(
                    {
                        "wf_i": wf_i,
                        "train_start": art.train_refit.train_start,
                        "train_end": art.train_refit.train_end,
                        "n_pairs": art.train_refit.n_pairs,
                        "equity": art.train_refit.equity,
                        "summary": art.train_refit.summary,
                    }
                )

            if (
                art.bo_run.selected_cv_scores is not None
                and not art.bo_run.selected_cv_scores.empty
            ):
                df_cv = art.bo_run.selected_cv_scores.copy()
                df_cv.insert(0, "wf_i", wf_i)
                cv_frames.append(df_cv)

    if use_global_intent_sim:
        carry_portfolio = dict(carry_intent_portfolio)
        if debug_root is not None and carry_intent_parts:
            pd.concat(carry_intent_parts, ignore_index=True).to_csv(
                debug_root / "walkforward_entry_intents.csv", index=False
            )
    else:
        carry_trades_df = (
            pd.concat(carry_trades_parts, ignore_index=True)
            if carry_trades_parts
            else pd.DataFrame()
        )
        carry_trades_df, blocked_df, ledger_report = _apply_global_positions_ledger(
            carry_trades_df
        )

        borrow_ctx_local = build_borrow_context(cfg_base)
        bt_cfg_local = (
            cfg_base.get("backtest", {})
            if isinstance(cfg_base.get("backtest"), dict)
            else {}
        )
        carry_trades_df, stateful_report = rescale_trades_stateful(
            carry_trades_df,
            price_data=_as_price_map(prices),
            initial_capital=float(bt0.get("initial_capital", 1_000_000.0)),
            wf_params=wf_sizing_params,
            borrow_ctx=borrow_ctx_local,
            settlement_lag_bars=int(bt_cfg_local.get("settlement_lag_bars", 0) or 0),
        )

        if debug_root is not None:
            _json_dump(debug_root / "walkforward_ledger_report.json", ledger_report)
            _json_dump(debug_root / "walkforward_stateful_report.json", stateful_report)
            if blocked_df is not None and not blocked_df.empty:
                blocked_df.to_csv(
                    debug_root / "walkforward_blocked_trades.csv", index=False
                )

        carry_portfolio = _portfolio_from_trades(carry_trades_df)
    cfg_global = deepcopy(cfg_base)
    cfg_global.setdefault("backtest", {})
    bt_global = (
        cfg_global.get("backtest", {})
        if isinstance(cfg_global.get("backtest"), dict)
        else {}
    )
    train_end = (global_test_start - pd.Timedelta(days=1)).normalize()
    train_start = (train_end - pd.Timedelta(days=1)).normalize()
    bt_global["splits"] = {
        "train": {"start": str(train_start.date()), "end": str(train_end.date())},
        "test": {
            "start": str(global_test_start.date()),
            "end": str(global_test_end.date()),
        },
    }
    cfg_global["backtest"] = bt_global

    borrow_ctx = build_borrow_context(cfg_global)
    carry_stats, carry_trades = backtest_portfolio_with_yaml_cfg(
        portfolio=carry_portfolio,
        price_data=_as_price_map(prices),
        market_data_panel=prices_panel,
        adv_map=adv_map,
        yaml_cfg=cfg_global,
        borrow_ctx=borrow_ctx,
    )
    eq_global = _equity_from_stats(carry_stats)

    if window_rows and isinstance(eq_global.index, pd.DatetimeIndex):
        synced_rows: list[dict[str, Any]] = []
        capital = float(bt0.get("initial_capital", 1_000_000.0))
        for row in window_rows:
            row_out = dict(row)
            test_end_raw = row_out.get("test_end")
            cap_end = capital
            if test_end_raw is not None:
                try:
                    cut_ts = _align_ts_to_index_tz(
                        pd.Timestamp(test_end_raw),
                        cast(pd.DatetimeIndex, eq_global.index),
                    )
                    eq_cut = eq_global.loc[eq_global.index <= cut_ts]
                    if not eq_cut.empty:
                        cap_end = float(eq_cut.iloc[-1])
                except Exception:
                    cap_end = capital
            row_out["capital_start"] = capital
            row_out["capital_end"] = cap_end
            row_out["capital_return"] = float(
                (cap_end / capital - 1.0) if capital > 0 else 0.0
            )
            if pd.notna(cap_end) and cap_end > 0:
                capital = cap_end
            synced_rows.append(row_out)
        window_rows = synced_rows

    if debug_root is not None:
        pd.DataFrame(window_rows).to_csv(
            debug_root / "test_window_summary_debug.csv", index=False
        )
        entry_intents_global = carry_stats.attrs.get("entry_intents_df")
        if (
            isinstance(entry_intents_global, pd.DataFrame)
            and not entry_intents_global.empty
        ):
            entry_intents_global.to_csv(
                debug_root / "walkforward_entry_intents_global.csv", index=False
            )
        state_transitions_global = carry_stats.attrs.get("state_transitions_df")
        if (
            isinstance(state_transitions_global, pd.DataFrame)
            and not state_transitions_global.empty
        ):
            state_transitions_global.to_csv(
                debug_root / "walkforward_state_transitions.csv", index=False
            )
        if isinstance(carry_trades, pd.DataFrame) and not carry_trades.empty:
            carry_trades.to_csv(debug_root / "walkforward_trades.csv", index=False)

    cv_scores = (
        pd.concat(cv_frames, ignore_index=True)
        if cv_frames
        else pd.DataFrame(
            columns=["wf_i", "fold_id", "score", "selection_metric", "component"]
        )
    )
    report = write_core_report(
        out_dir,
        reporting_cfg=reporting_cfg,
        test_eq=eq_global,
        test_trades=carry_trades
        if isinstance(carry_trades, pd.DataFrame)
        else pd.DataFrame(),
        train_refits=train_refits,
        cv_scores=cv_scores,
        window_rows=pd.DataFrame(window_rows),
    )

    of_cfg = (
        cfg_base.get("overfit", {}) if isinstance(cfg_base.get("overfit"), dict) else {}
    )
    if bool(of_cfg.get("enabled", False)) and debug_root is not None:
        try:
            _write_walkforward_overfit_summary(
                debug_root,
                windows=windows,
                out_relpath=str(of_cfg.get("out_relpath", "overfit_summary.json")),
            )
        except Exception:
            logger.warning("Walkforward overfit summary failed.", exc_info=True)

    return {
        "out_dir": str(out_dir),
        "report_dir": report["report_dir"],
        "n_windows": int(len(windows)),
    }


def _default_out_dir(cfg_path: Path) -> Path:
    ts = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return Path("runs/results/performance") / f"BT-{ts}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Backtest runner (lob or light execution)")
    ap.add_argument(
        "--cfg", type=Path, default=Path("runs/configs/config_backtest.yaml")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: runs/results/performance/BT-...)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer pairs / smaller windows)",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args(argv)

    logger.setLevel(getattr(logging, str(args.log_level).upper(), logger.level))

    if not args.cfg.exists():
        logger.error("Config not found: %s", args.cfg)
        return 2

    from backtest.utils.common.io import load_yaml_dict as _load_yaml

    cfg = _load_yaml(args.cfg)
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
