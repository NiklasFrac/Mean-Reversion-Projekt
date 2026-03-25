from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from backtest.borrow.context import build_borrow_context
from backtest.config.cfg import AppConfig, config_to_dict, parse_config
from backtest.optimize.runner import load_bo_trials
from backtest.reporting.common import equity_from_stats
from backtest.reporting.report_bundle import (
    ReportingConfig,
    debug_dir as report_debug_dir,
    write_core_report,
)
from backtest.risk.policy import build_risk_policy
from backtest.runner.portfolio import (
    collect_portfolio_intents,
    write_pnl_concentration_report,
)
from backtest.runner.runtime import RuntimeContext, build_runtime_calendar
from backtest.runner.single import run_single_backtest
from backtest.simulators.engine import backtest_portfolio_with_yaml_cfg
from backtest.simulators.performance import compute_drawdowns
from backtest.simulators.stateful import replay_trades_mtm, rescale_trades_stateful
from backtest.utils.io import write_json
from backtest.utils.prices import as_price_map
from backtest.utils.tz import align_ts_to_index
from backtest.windowing.plan import generate_walkforward_windows_from_cfg

__all__ = ["apply_global_positions_ledger", "run_walkforward_backtest"]

logger = logging.getLogger("backtest.runner.walkforward_run")


def apply_global_positions_ledger(
    trades_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if trades_df is None or trades_df.empty:
        return trades_df, pd.DataFrame(), {"kept": 0, "blocked": 0}
    df = trades_df.copy()
    if "entry_date" not in df.columns or "exit_date" not in df.columns:
        return (
            df,
            pd.DataFrame(),
            {"kept": int(len(df)), "blocked": 0, "warning": "missing entry/exit"},
        )
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    df = df.dropna(subset=["entry_date", "exit_date"])
    if df.empty:
        return df, pd.DataFrame(), {"kept": 0, "blocked": 0}

    def _pair_key(row: pd.Series) -> str:
        pair = row.get("pair")
        if pair is not None and str(pair).strip():
            return str(pair)
        y = row.get("y_symbol") or row.get("t1_symbol") or row.get("leg1_symbol")
        x = row.get("x_symbol") or row.get("t2_symbol") or row.get("leg2_symbol")
        if y and x:
            return f"{str(y).upper()}-{str(x).upper()}"
        return "PAIR"

    df["_ledger_pair"] = df.apply(_pair_key, axis=1)
    df = df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
    open_until: dict[str, pd.Timestamp] = {}
    keep_mask = []
    blocked_rows: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        pair = str(row.get("_ledger_pair", "PAIR"))
        entry = pd.Timestamp(row["entry_date"])
        exit_ts = pd.Timestamp(row["exit_date"])
        last_exit = open_until.get(pair)
        if last_exit is not None and entry <= last_exit:
            keep_mask.append(False)
            blocked_rows.append(pos)
            continue
        keep_mask.append(True)
        open_until[pair] = exit_ts

    kept = df.loc[keep_mask].drop(columns=["_ledger_pair"])
    blocked = (
        df.loc[blocked_rows].drop(columns=["_ledger_pair"])
        if blocked_rows
        else pd.DataFrame()
    )
    return kept, blocked, {"kept": int(len(kept)), "blocked": int(len(blocked))}
def _portfolio_has_intents(portfolio: Mapping[str, Any] | None) -> bool:
    intents = collect_portfolio_intents(portfolio)
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


def _stats_from_equity_series(eq: pd.Series) -> pd.DataFrame:
    if eq is None or eq.empty:
        return pd.DataFrame(columns=["equity", "returns", "drawdown", "drawdown_pct"])
    returns = pd.to_numeric(eq, errors="coerce").pct_change().fillna(0.0)
    drawdown, _max_dd, _, _ = compute_drawdowns(eq)
    return pd.DataFrame(
        {
            "equity": pd.to_numeric(eq, errors="coerce"),
            "returns": pd.to_numeric(returns, errors="coerce"),
            "drawdown": pd.to_numeric(drawdown, errors="coerce"),
            "drawdown_pct": pd.to_numeric(drawdown, errors="coerce"),
        }
    )


def run_walkforward_backtest(
    *,
    cfg_base: AppConfig,
    out_dir: Path,
    reporting_cfg: ReportingConfig,
    runtime: RuntimeContext,
    quick: bool = False,
) -> dict[str, Any]:
    prices_panel = runtime.prices_panel
    prices = runtime.prices
    pairs = runtime.pairs
    adv_map = runtime.adv_map

    calendar = build_runtime_calendar(runtime)
    windows, wf_meta = generate_walkforward_windows_from_cfg(
        calendar=calendar, cfg=cfg_base
    )
    if not windows:
        raise ValueError(
            "Walkforward enabled but produced 0 windows (check backtest.range and months params)."
        )

    debug_root = report_debug_dir(out_dir) if reporting_cfg.debug_enabled else None
    if debug_root is not None:
        write_json(
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
    bt0 = cfg_base.backtest
    rolling_capital = float(bt0.initial_capital)

    window_rows: list[dict[str, Any]] = []
    train_refits: list[Mapping[str, Any]] = []
    cv_frames: list[pd.DataFrame] = []
    bo_trial_frames: list[pd.DataFrame] = []

    for window in windows:
        wf_i = int(window.i)
        splits = dict(window.as_splits())
        splits["test"] = {
            **splits["test"],
            "entry_end": str(window.test_end.date()),
            "exit_end": str(global_test_end.date()),
        }
        cfg_eff = parse_config(
            {
                **config_to_dict(cfg_base),
                "backtest": {
                    **config_to_dict(cfg_base)["backtest"],
                    "splits": splits,
                    "initial_capital": float(rolling_capital),
                },
            }
        )

        sizing_policy = build_risk_policy(
            risk_cfg=config_to_dict(cfg_eff)["risk"],
            backtest_cfg=config_to_dict(cfg_eff)["backtest"],
            execution_cfg=config_to_dict(cfg_eff)["execution"],
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
            art = run_single_backtest(
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
                write_json(
                    debug_out / "walkforward_window.json",
                    {"i": wf_i, "splits": splits, "truncated": bool(window.truncated)},
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
                intents_raw = collect_portfolio_intents(art.portfolio)
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
                "train_start": str(window.train_start.date()),
                "train_end": str(window.train_end.date()),
                "test_start": str(window.test_start.date()),
                "test_end": str(window.test_end.date()),
                "truncated": bool(window.truncated),
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
                        "trades": art.train_refit.trades,
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
            df_trials = load_bo_trials(art.bo_run.bo_out)
            if not df_trials.empty:
                df_trials.insert(0, "wf_i", wf_i)
                bo_trial_frames.append(df_trials)

    carry_stats = pd.DataFrame()
    carry_trades = pd.DataFrame()
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
        carry_trades_df, blocked_df, ledger_report = apply_global_positions_ledger(
            carry_trades_df
        )

        borrow_ctx_local = build_borrow_context(cfg_base.borrow)
        carry_trades_df, stateful_report = rescale_trades_stateful(
            carry_trades_df,
            price_data=as_price_map(prices),
            initial_capital=float(bt0.initial_capital),
            wf_params=wf_sizing_params,
            borrow_ctx=borrow_ctx_local,
            settlement_lag_bars=int(cfg_base.backtest.settlement_lag_bars),
        )

        if debug_root is not None:
            write_json(debug_root / "walkforward_ledger_report.json", ledger_report)
            write_json(debug_root / "walkforward_stateful_report.json", stateful_report)
            if blocked_df is not None and not blocked_df.empty:
                blocked_df.to_csv(
                    debug_root / "walkforward_blocked_trades.csv", index=False
                )

        replay_eq, carry_trades, replay_report = replay_trades_mtm(
            carry_trades_df,
            price_data=as_price_map(prices),
            initial_capital=float(bt0.initial_capital),
            borrow_ctx=borrow_ctx_local,
            settlement_lag_bars=int(cfg_base.backtest.settlement_lag_bars),
        )
        carry_stats = _stats_from_equity_series(replay_eq)
        carry_stats.attrs.update(replay_report)
        eq_global = replay_eq
        if debug_root is not None:
            write_json(debug_root / "walkforward_replay_report.json", replay_report)

    if use_global_intent_sim:
        cfg_global_dict = config_to_dict(cfg_base)
        train_end = (global_test_start - pd.Timedelta(days=1)).normalize()
        train_start = (train_end - pd.Timedelta(days=1)).normalize()
        cfg_global = parse_config(
            {
                **cfg_global_dict,
                "backtest": {
                    **cfg_global_dict["backtest"],
                    "splits": {
                        "train": {
                            "start": str(train_start.date()),
                            "end": str(train_end.date()),
                        },
                        "test": {
                            "start": str(global_test_start.date()),
                            "end": str(global_test_end.date()),
                        },
                    },
                },
            }
        )

        borrow_ctx = build_borrow_context(cfg_global.borrow)
        result = backtest_portfolio_with_yaml_cfg(
            portfolio=carry_portfolio,
            price_data=as_price_map(prices),
            market_data_panel=prices_panel,
            adv_map=adv_map,
            yaml_cfg=config_to_dict(cfg_global),
            borrow_ctx=borrow_ctx,
        )
        if isinstance(result, tuple):
            carry_stats, carry_trades = result
        else:
            carry_stats = result.stats
            carry_trades = result.trades
        eq_global = equity_from_stats(carry_stats)

    if window_rows and isinstance(eq_global.index, pd.DatetimeIndex):
        synced_rows: list[dict[str, Any]] = []
        capital = float(bt0.initial_capital)
        for row in window_rows:
            row_out = dict(row)
            test_end_raw = row_out.get("test_end")
            cap_end = capital
            if test_end_raw is not None:
                try:
                    cut_ts = align_ts_to_index(
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
    bo_trials = (
        pd.concat(bo_trial_frames, ignore_index=True)
        if bo_trial_frames
        else pd.DataFrame()
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
        bo_trials=bo_trials,
        window_rows=pd.DataFrame(window_rows),
    )
    write_pnl_concentration_report(
        Path(report["report_dir"]),
        carry_trades if isinstance(carry_trades, pd.DataFrame) else pd.DataFrame(),
    )

    return {
        "out_dir": str(out_dir),
        "report_dir": report["report_dir"],
        "n_windows": int(len(windows)),
    }
