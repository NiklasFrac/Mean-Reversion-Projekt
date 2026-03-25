from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.config.cfg import AppConfig, config_to_dict
from backtest.utils.io import write_json
from backtest.utils.tz import utc_now

from .cv import _fold_score_from_pnl, _fold_score_with_refit
from .fast_objective import (
    _portfolio_pnl_equal_weight,
    _precompute_spreads,
    _simulate_stage_pnl,
)
from .inputs import TrainInputs, build_train_inputs
from .params import (
    MarkovParams,
    SignalParams,
    markov_bo_defaults,
    markov_search_space,
    resolve_bo_config,
    signal_search_space,
)
from .realistic_objective import _fold_score_realistic
from .search import BAD_SCORE, _bayes_optimize, _safe_int

logger = logging.getLogger("backtest.optimize.optimizer")


def _score_fast_candidate(
    *,
    per_pair_prices: Mapping[str, Mapping[str, Any]],
    cal: pd.DatetimeIndex,
    init_cap: float,
    cv: Any,
    seed: int,
    out_dir: Path,
    z_default: int,
    hmax0: int,
    cool0: int,
    cfg: Mapping[str, Any],
    spreads: Mapping[str, pd.Series],
    component: str,
    params_for_log: Mapping[str, Any],
    entry_z: float,
    exit_z: float,
    stop_z: float,
    markov_overrides: Mapping[str, Any] | None = None,
) -> float:
    if cv.enabled:
        return _fold_score_with_refit(
            per_pair_prices=per_pair_prices,
            calendar=cal,
            initial_capital=init_cap,
            cv=cv,
            seed=seed,
            component=component,
            out_dir=out_dir,
            params_for_log=params_for_log,
            z_window=int(z_default),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            max_hold_days=hmax0,
            cooldown_days=cool0,
            cfg=cfg,
            markov_overrides=markov_overrides,
        )

    pnl_by_pair = _simulate_stage_pnl(
        spreads=spreads,
        per_pair_prices=per_pair_prices,
        z_window=z_default,
        entry_z=float(entry_z),
        exit_z=float(exit_z),
        stop_z=float(stop_z),
        max_hold_days=hmax0,
        cooldown_days=cool0,
        cfg=cfg,
        calendar=cal,
        markov_overrides=markov_overrides,
    )
    pnl = _portfolio_pnl_equal_weight(pnl_by_pair, cal)
    return _fold_score_from_pnl(
        pnl,
        calendar=cal,
        initial_capital=init_cap,
        cv=cv,
        seed=seed,
        component=component,
        out_dir=out_dir,
        params_for_log=params_for_log,
    )


def run_bo_conservative(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None = None,
    pairs: Mapping[str, Any],
    pairs_data: Mapping[str, Any] | None = None,
    cfg: AppConfig,
    adv_map: Mapping[str, float] | None = None,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run BO on the training window using the active objective mode and the
    current search-space schema.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = _safe_int(cfg.runtime.seed, 42)
    resolved = resolve_bo_config(cfg)
    mode = resolved.mode
    cv = resolved.cv
    markov_defaults = markov_bo_defaults(cfg)
    cfg_map = config_to_dict(cfg)

    train_inputs: TrainInputs = build_train_inputs(
        prices=prices, pairs=pairs, pairs_data=pairs_data, cfg=cfg
    )
    per_pair_prices = train_inputs.per_pair_prices
    cal = train_inputs.calendar

    if mode == "realistic":
        if prices_panel is None:
            raise ValueError("bo.mode='realistic' requires prices_panel.")
        if pairs_data is None:
            raise ValueError("bo.mode='realistic' requires pairs_data.")
        if not pairs_data:
            raise ValueError("bo.mode='realistic' requires non-empty pairs_data.")

    z_default = int(cfg.spread_zscore.z_window)
    if z_default <= 1:
        raise ValueError("spread_zscore.z_window must be > 1 for BO.")

    spreads: dict[str, pd.Series] = {}
    if mode == "fast":
        spreads = _precompute_spreads(
            per_pair_prices, cfg=cfg_map, z_window_for_beta=int(z_default)
        )
        if not spreads:
            raise ValueError("BO failed: could not precompute spreads for any pair")

    entry0 = float(cfg.signal.entry_z)
    exit0 = float(cfg.signal.exit_z)
    stop0 = float(cfg.signal.stop_z)
    hmax0 = int(cfg.signal.max_hold_days)
    cool0 = int(cfg.signal.cooldown_days)
    init_cap = float(cfg.backtest.initial_capital)

    theta_space = signal_search_space(
        cfg,
        entry0=entry0,
        exit0=exit0,
        stop0=stop0,
    )

    cache_theta: dict[str, float] = {}

    def _obj_theta(
        entry_z: float,
        exit_z: float,
        stop_z: float,
    ) -> float:
        ez = float(entry_z)
        xz = float(exit_z)
        sz = float(stop_z)
        if not (0.0 < xz < ez):
            return BAD_SCORE
        if sz <= ez:
            return BAD_SCORE
        key = json.dumps(
            {"e": ez, "x": xz, "s": sz},
            sort_keys=True,
        )
        if key in cache_theta:
            return float(cache_theta[key])
        if mode == "realistic":
            sc = _fold_score_realistic(
                cfg=cfg_map,
                prices=prices,
                prices_panel=cast(pd.DataFrame, prices_panel),
                pairs_data=cast(Mapping[str, Any], pairs_data),
                adv_map=adv_map,
                calendar=cal,
                cv=cv,
                seed=seed,
                component="theta_sig",
                out_dir=out_dir,
                params_for_log={
                    "entry_z": ez,
                    "exit_z": xz,
                    "stop_z": sz,
                },
                theta={
                    "entry_z": ez,
                    "exit_z": xz,
                    "stop_z": sz,
                },
                metric=resolved.realistic_metric,
                initial_capital=init_cap,
            )
        else:
            sc = _score_fast_candidate(
                per_pair_prices=per_pair_prices,
                cal=cal,
                init_cap=init_cap,
                cv=cv,
                seed=seed,
                out_dir=out_dir,
                z_default=z_default,
                hmax0=hmax0,
                cool0=cool0,
                cfg=cfg_map,
                spreads=spreads,
                component="theta_sig",
                params_for_log={
                    "entry_z": ez,
                    "exit_z": xz,
                    "stop_z": sz,
                },
                entry_z=ez,
                exit_z=xz,
                stop_z=sz,
            )
        cache_theta[key] = float(sc)
        return float(sc)

    pbounds = {
        "entry_z": theta_space.entry_z,
        "exit_z": theta_space.exit_z,
        "stop_z": theta_space.stop_z,
    }

    if all(abs(pbounds[key][0] - pbounds[key][1]) < 1e-12 for key in pbounds):
        theta_sig = SignalParams(
            entry_z=float(pbounds["entry_z"][0]),
            exit_z=float(pbounds["exit_z"][0]),
            stop_z=float(pbounds["stop_z"][0]),
        )
        theta_score = _obj_theta(
            theta_sig.entry_z,
            theta_sig.exit_z,
            theta_sig.stop_z,
        )
    else:
        best_theta, theta_score = _bayes_optimize(
            out_dir=out_dir,
            stage="theta_sig",
            pbounds=pbounds,
            objective=_obj_theta,
            seed=seed,
            init_points=int(resolved.signal_budget.init_points),
            n_iter=int(resolved.signal_budget.n_iter),
            patience=int(resolved.signal_budget.patience),
        )
        theta_sig = SignalParams(
            entry_z=float(best_theta.get("entry_z", entry0)),
            exit_z=float(best_theta.get("exit_z", exit0)),
            stop_z=float(best_theta.get("stop_z", stop0)),
        )

    theta_sig_score = float(theta_score)
    theta_hat = theta_sig.as_dict()
    theta_markov_hat: dict[str, Any] | None = None
    theta_markov_score: float | None = None

    if markov_defaults.enabled:
        theta_markov_space = markov_search_space(cfg, markov_defaults)
        cache_markov: dict[str, float] = {}

        def _obj_markov(min_revert_prob: float, horizon_days: float) -> float:
            p_min = float(np.clip(float(min_revert_prob), 0.0, 1.0))
            horizon = max(1, int(round(float(horizon_days))))
            key = json.dumps(
                {"p": p_min, "h": horizon},
                sort_keys=True,
            )
            if key in cache_markov:
                return float(cache_markov[key])

            if mode == "realistic":
                sc = _fold_score_realistic(
                    cfg=cfg_map,
                    prices=prices,
                    prices_panel=cast(pd.DataFrame, prices_panel),
                    pairs_data=cast(Mapping[str, Any], pairs_data),
                    adv_map=adv_map,
                    calendar=cal,
                    cv=cv,
                    seed=seed,
                    component="theta_markov",
                    out_dir=out_dir,
                    params_for_log={
                        "min_revert_prob": p_min,
                        "horizon_days": horizon,
                    },
                    theta={
                        "theta_sig_hat": dict(theta_hat),
                        "theta_markov_hat": {
                            "min_revert_prob": p_min,
                            "horizon_days": horizon,
                        },
                    },
                    metric=resolved.realistic_metric,
                    initial_capital=init_cap,
                )
            else:
                sc = _score_fast_candidate(
                    per_pair_prices=per_pair_prices,
                    cal=cal,
                    init_cap=init_cap,
                    cv=cv,
                    seed=seed,
                    out_dir=out_dir,
                    z_default=z_default,
                    hmax0=hmax0,
                    cool0=cool0,
                    cfg=cfg_map,
                    spreads=spreads,
                    component="theta_markov",
                    params_for_log={
                        "min_revert_prob": p_min,
                        "horizon_days": horizon,
                    },
                    entry_z=theta_sig.entry_z,
                    exit_z=theta_sig.exit_z,
                    stop_z=theta_sig.stop_z,
                    markov_overrides={
                        "min_revert_prob": p_min,
                        "horizon_days": horizon,
                    },
                )
            cache_markov[key] = float(sc)
            return float(sc)

        pbounds_markov = {
            "min_revert_prob": theta_markov_space.min_revert_prob,
            "horizon_days": theta_markov_space.horizon_days,
        }

        if all(
            abs(pbounds_markov[key][0] - pbounds_markov[key][1]) < 1e-12
            for key in pbounds_markov
        ):
            theta_markov = MarkovParams(
                min_revert_prob=float(pbounds_markov["min_revert_prob"][0]),
                horizon_days=max(1, int(round(float(pbounds_markov["horizon_days"][0])))),
            )
            theta_markov_score = _obj_markov(
                theta_markov.min_revert_prob,
                float(theta_markov.horizon_days),
            )
        else:
            best_markov, theta_markov_score = _bayes_optimize(
                out_dir=out_dir,
                stage="theta_markov",
                pbounds=pbounds_markov,
                objective=_obj_markov,
                seed=seed,
                init_points=int(resolved.markov_budget.init_points),
                n_iter=int(resolved.markov_budget.n_iter),
                patience=int(resolved.markov_budget.patience),
            )
            theta_markov = MarkovParams(
                min_revert_prob=float(
                    np.clip(
                        float(
                            best_markov.get(
                                "min_revert_prob", markov_defaults.min_revert_prob
                            )
                        ),
                        0.0,
                        1.0,
                    )
                ),
                horizon_days=max(
                    1,
                    int(
                        round(
                            float(
                                best_markov.get(
                                    "horizon_days", markov_defaults.horizon_days
                                )
                            )
                        )
                    ),
                ),
            )
        theta_markov_hat = theta_markov.as_dict()
        theta_markov_score = float(theta_markov_score)

    final_component = "theta_markov" if theta_markov_hat else "theta_sig"
    final_score = float(
        theta_markov_score if theta_markov_score is not None else theta_sig_score
    )

    res = {
        "meta": {
            "ts_utc": utc_now().isoformat(timespec="seconds"),
            "seed": int(seed),
            "train": {
                "start": str(cal.min()),
                "end": str(cal.max()),
                "n_days": int(len(cal)),
            },
            "n_pairs": int(len(per_pair_prices)),
            "mode": str(mode),
            "selected_component": final_component,
            "cv": {
                "enabled": bool(cv.enabled),
                "scheme": cv.scheme,
                "n_blocks": int(cv.n_blocks),
                "k_test_blocks": int(cv.k_test_blocks),
                "purge": int(cv.purge),
                "embargo": float(cv.embargo),
                "max_folds": cv.max_folds,
                "aggregate": cv.aggregate,
                "trim_pct": float(cv.trim_pct),
                "shuffle": bool(cv.shuffle),
            },
        },
        "theta_sig_hat": dict(theta_hat),
        "theta_sig_score": float(theta_sig_score),
        "score": float(final_score),
    }
    if theta_markov_hat is not None:
        res["theta_markov_hat"] = dict(theta_markov_hat)
        res["theta_markov_score"] = float(
            theta_markov_score if theta_markov_score is not None else final_score
        )
    write_json(out_dir / "bo_best.json", res)
    return res
