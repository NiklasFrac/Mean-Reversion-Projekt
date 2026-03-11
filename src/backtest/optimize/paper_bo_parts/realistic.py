from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.borrow.context import build_borrow_context
from backtest.config.cfg import make_config_from_yaml
from backtest.loader import as_price_mapping
from backtest.reporting.tearsheet import summarize_stats
from backtest.utils.reporting import equity_from_stats as _equity_from_stats
from backtest.simulators.engine import backtest_portfolio
from backtest.simulators.performance import compute_performance
from backtest.strat.registry import build_strategy_instance, list_strategy_keys

from .core import BAD_SCORE, _log_trial
from .cv import (
    BOCVConfig,
    _aggregate_scores,
    _build_cv_folds,
    _parse_bo_mode,
    _split_consecutive_indices,
)

logger = logging.getLogger("backtest.paper_bo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _build_strategy(cfg: Mapping[str, Any], *, borrow_ctx: Any) -> Any:
    scfg = cfg.get("strategy", {}) if isinstance(cfg.get("strategy"), Mapping) else {}
    name = str(scfg.get("name") or "baseline").strip()
    params = scfg.get("params", {}) if isinstance(scfg.get("params"), Mapping) else {}
    try:
        return build_strategy_instance(
            dict(cfg),
            borrow_ctx=borrow_ctx,
            params=dict(params) if params else {},
            name=name,
        )
    except KeyError:
        keys = list_strategy_keys()
        raise KeyError(
            f"Unknown strategy.name={name!r} for realistic BO. Available: {keys}"
        ) from None


def _candidate_bo_payload(theta: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(theta.get("theta_sig_hat"), Mapping) or isinstance(
        theta.get("theta_markov_hat"), Mapping
    ):
        out: dict[str, Any] = {}
        if isinstance(theta.get("theta_sig_hat"), Mapping):
            out["theta_sig_hat"] = dict(
                cast(Mapping[str, Any], theta.get("theta_sig_hat"))
            )
        if isinstance(theta.get("theta_markov_hat"), Mapping):
            out["theta_markov_hat"] = dict(
                cast(Mapping[str, Any], theta.get("theta_markov_hat"))
            )
        return out
    return {"theta_sig_hat": dict(theta)}


def apply_bo_params_to_cfg(
    cfg: dict[str, Any], bo_best: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Patch a backtest config dict with the BO output.
    """
    out = dict(cfg)
    sig = dict(out.get("signal") or {})
    markov = dict(out.get("markov_filter") or {})
    if "stage1" in bo_best or "stage2" in bo_best:
        raise ValueError(
            "Legacy stage-based bo_best payload is no longer supported. "
            "Expected keys: theta_sig_hat, optional theta_markov_hat, score."
        )

    try:
        th = bo_best.get("theta_sig_hat")
        if isinstance(th, Mapping):
            sig["entry_z"] = float(
                cast(Any, th.get("entry_z", sig.get("entry_z", 2.0)))
            )
            sig["exit_z"] = float(cast(Any, th.get("exit_z", sig.get("exit_z", 0.5))))
            sig["stop_z"] = float(cast(Any, th.get("stop_z", sig.get("stop_z", 2.0))))
    except Exception:
        pass

    try:
        th_markov = bo_best.get("theta_markov_hat")
        if isinstance(th_markov, Mapping):
            p_min = float(
                cast(
                    Any,
                    th_markov.get(
                        "min_revert_prob", markov.get("min_revert_prob", 0.5)
                    ),
                )
            )
            markov["min_revert_prob"] = float(np.clip(p_min, 0.0, 1.0))

            h_raw = th_markov.get(
                "horizon_days", th_markov.get("horizon", markov.get("horizon_days", 10))
            )
            horizon_days = max(1, int(round(float(cast(Any, h_raw)))))
            markov["horizon_days"] = int(horizon_days)
        if markov:
            out["markov_filter"] = markov
    except Exception:
        pass

    out["signal"] = sig
    return out


@dataclass(frozen=True)
class RealisticConfig:
    metric: str


def _parse_realistic_cfg(cfg: Mapping[str, Any]) -> RealisticConfig:
    mode = _parse_bo_mode(cfg)
    if mode != "realistic":
        return RealisticConfig(metric="sharpe")

    bo = cfg.get("bo", {}) if isinstance(cfg.get("bo"), Mapping) else {}
    realistic = (
        bo.get("realistic", {}) if isinstance(bo.get("realistic"), Mapping) else {}
    )
    metric = str(realistic.get("metric", "sharpe")).strip().lower()
    if metric not in {"sharpe", "cagr", "calmar"}:
        metric = "sharpe"
    return RealisticConfig(metric=metric)


def _metric_from_summary(summary: pd.DataFrame, metric: str) -> float:
    if summary is None or summary.empty:
        return BAD_SCORE
    row = summary.to_dict(orient="records")[0] if not summary.empty else {}
    val = row.get(metric)
    try:
        out = float(cast(Any, val))
        return out if np.isfinite(out) else BAD_SCORE
    except Exception:
        return BAD_SCORE


def _score_realistic_segment(
    *,
    cfg: Mapping[str, Any],
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame,
    pairs_data: Mapping[str, Any],
    adv_map: Mapping[str, float] | None,
    theta: Mapping[str, Any],
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    initial_capital: float,
) -> tuple[pd.Series | None, dict[str, Any]]:
    try:
        if train_dates.empty or test_dates.empty:
            return None, {"n_pairs": 0, "n_trades": 0, "reason": "empty_fold"}

        cfg_local = apply_bo_params_to_cfg(dict(cfg), _candidate_bo_payload(theta))
        cfg_local["_bo_train_dates"] = [str(pd.Timestamp(ts)) for ts in train_dates]
        cfg_local["_bo_require_execution_hooks"] = True

        bt = dict(cfg_local.get("backtest") or {})
        bt["splits"] = {
            "train": {
                "start": str(pd.Timestamp(train_dates.min())),
                "end": str(pd.Timestamp(train_dates.max())),
            },
            "test": {
                "start": str(pd.Timestamp(test_dates.min())),
                "end": str(pd.Timestamp(test_dates.max())),
                "entry_end": str(pd.Timestamp(test_dates.max())),
                "exit_end": str(pd.Timestamp(test_dates.max())),
            },
        }
        cfg_local["backtest"] = bt

        borrow_ctx = build_borrow_context(cfg_local)
        strat = _build_strategy(cfg_local, borrow_ctx=borrow_ctx)
        portfolio = strat(dict(pairs_data))
        if not portfolio:
            return None, {"n_pairs": 0, "n_trades": 0, "reason": "empty_portfolio"}

        cfg_obj = replace(
            make_config_from_yaml(dict(cfg)),
            splits=cast(dict[str, dict[str, Any]], bt["splits"]),
            raw_yaml=dict(cfg_local),
        )
        stats, trades = backtest_portfolio(
            portfolio=portfolio,
            price_data=as_price_mapping(prices),
            cfg=cfg_obj,
            borrow_ctx=borrow_ctx,
            market_data_panel=prices_panel,
            adv_map=adv_map,
        )
        if (
            bool(getattr(stats, "attrs", {}).get("exec_lob_enabled", False))
            and isinstance(trades, pd.DataFrame)
            and not trades.empty
        ):
            required_cols = (
                "exec_entry_vwap_y",
                "exec_entry_vwap_x",
                "exec_exit_vwap_y",
                "exec_exit_vwap_x",
            )
            missing = [col for col in required_cols if col not in trades.columns]
            if missing:
                raise RuntimeError(f"LOB execution annotations missing: {missing}")
        eq = pd.to_numeric(_equity_from_stats(stats), errors="coerce").dropna()
        if eq.empty:
            return pd.Series(dtype=float), {
                "n_pairs": int(len(portfolio)),
                "n_trades": 0,
                "reason": "empty_equity",
            }
        pnl = eq.diff().fillna(float(eq.iloc[0]) - float(initial_capital)).astype(float)
        n_trades = int(len(trades)) if isinstance(trades, pd.DataFrame) else 0
        return pnl, {"n_pairs": int(len(portfolio)), "n_trades": n_trades}
    except Exception as e:
        logger.warning("Realistic BO candidate failed: %s", e, exc_info=True)
        return None, {
            "n_pairs": 0,
            "n_trades": 0,
            "reason": "exception",
            "error": str(e),
        }


def _score_realistic_candidate(
    *,
    cfg: Mapping[str, Any],
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame,
    pairs_data: Mapping[str, Any],
    adv_map: Mapping[str, float] | None,
    theta: Mapping[str, Any],
    train_dates: pd.DatetimeIndex,
    eval_segments: list[pd.DatetimeIndex],
    metric: str,
    initial_capital: float,
) -> tuple[float, dict[str, Any]]:
    pnl_parts: list[pd.Series] = []
    n_pairs = 0
    n_trades = 0
    last_reason: str | None = None
    last_error: str | None = None

    for seg in eval_segments:
        pnl_seg, meta = _score_realistic_segment(
            cfg=cfg,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=pairs_data,
            adv_map=adv_map,
            theta=theta,
            train_dates=train_dates,
            test_dates=seg,
            initial_capital=initial_capital,
        )
        n_pairs = max(n_pairs, int(meta.get("n_pairs", 0) or 0))
        n_trades += int(meta.get("n_trades", 0) or 0)
        last_reason = cast(str | None, meta.get("reason"))
        last_error = cast(str | None, meta.get("error"))
        if isinstance(pnl_seg, pd.Series) and not pnl_seg.empty:
            pnl_parts.append(
                pd.to_numeric(pnl_seg, errors="coerce").dropna().astype(float)
            )

    if not pnl_parts:
        return BAD_SCORE, {
            "n_pairs": n_pairs,
            "n_trades": n_trades,
            "reason": last_reason,
            "error": last_error,
        }

    pnl = pd.concat(pnl_parts).sort_index()
    if isinstance(pnl.index, pd.DatetimeIndex):
        pnl = pnl.groupby(level=0).sum().sort_index()
    perf = compute_performance(pnl, float(initial_capital))
    eq = cast(pd.Series, perf.get("equity", pd.Series(dtype=float)))
    summary = summarize_stats(eq)
    score = _metric_from_summary(summary, metric)
    return score, {"n_pairs": n_pairs, "n_trades": n_trades}


def _fold_score_realistic(
    *,
    cfg: Mapping[str, Any],
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame,
    pairs_data: Mapping[str, Any],
    adv_map: Mapping[str, float] | None,
    calendar: pd.DatetimeIndex,
    cv: BOCVConfig,
    seed: int,
    component: str,
    out_dir: Path,
    params_for_log: Mapping[str, Any],
    theta: Mapping[str, Any],
    metric: str,
    initial_capital: float,
) -> float:
    if not cv.enabled:
        score, _meta = _score_realistic_candidate(
            cfg=cfg,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=pairs_data,
            adv_map=adv_map,
            theta=theta,
            train_dates=calendar,
            eval_segments=[calendar],
            metric=metric,
            initial_capital=initial_capital,
        )
        _log_trial(out_dir, component=component, params=params_for_log, score=score)
        return score if np.isfinite(score) else BAD_SCORE

    folds = _build_cv_folds(calendar, cv=cv, seed=seed)
    if not folds:
        score, _meta = _score_realistic_candidate(
            cfg=cfg,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=pairs_data,
            adv_map=adv_map,
            theta=theta,
            train_dates=calendar,
            eval_segments=[calendar],
            metric=metric,
            initial_capital=initial_capital,
        )
        _log_trial(out_dir, component=component, params=params_for_log, score=score)
        return score if np.isfinite(score) else BAD_SCORE

    scores: list[float] = []
    for j, (train_idx, test_idx) in enumerate(folds):
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_dates = calendar[train_idx]
        eval_segments = [calendar[seg] for seg in _split_consecutive_indices(test_idx)]
        score, _meta = _score_realistic_candidate(
            cfg=cfg,
            prices=prices,
            prices_panel=prices_panel,
            pairs_data=pairs_data,
            adv_map=adv_map,
            theta=theta,
            train_dates=train_dates,
            eval_segments=eval_segments,
            metric=metric,
            initial_capital=initial_capital,
        )
        if np.isfinite(score) and score != BAD_SCORE:
            scores.append(float(score))
            _log_trial(
                out_dir,
                component=component,
                params=params_for_log,
                fold_id=j,
                oos_score=float(score),
            )

    agg = _aggregate_scores(scores, mode=cv.aggregate, trim_pct=cv.trim_pct)
    _log_trial(out_dir, component=component, params=params_for_log, score=agg)
    return agg if np.isfinite(agg) else BAD_SCORE
