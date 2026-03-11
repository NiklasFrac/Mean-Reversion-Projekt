from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.utils.alpha import evaluate_pair_cointegration, resolve_half_life_cfg
from backtest.utils.tz import (
    align_ts_to_index,
    ensure_dtindex_tz,
    to_naive_local,
    utc_now,
)

from .core import BAD_SCORE, _bayes_optimize, _safe_int
from .cv import (
    _fold_score_from_pnl,
    _fold_score_with_refit,
    _parse_bo_cv,
    _parse_bo_mode,
)
from .realistic import _fold_score_realistic, _parse_realistic_cfg
from .sim import _portfolio_pnl_equal_weight, _precompute_spreads, _simulate_stage_pnl

logger = logging.getLogger("backtest.paper_bo")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s")
    )
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _normalize_index_to_calendar(
    idx: pd.DatetimeIndex, cal: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
    if cal.tz is None:
        if idx.tz is not None:
            idx = to_naive_local(idx)
    else:
        idx = ensure_dtindex_tz(idx, str(cal.tz), ambiguous="NaT", nonexistent="NaT")
    return idx


def _slice_series(
    s: pd.Series, start: pd.Timestamp, end: pd.Timestamp, *, cal: pd.DatetimeIndex
) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    idx = _normalize_index_to_calendar(pd.DatetimeIndex(s.index), cal)
    out = s.copy()
    out.index = idx
    out = out.loc[~out.index.isna()]
    out = out.sort_index()
    out = out[(out.index >= start) & (out.index <= end)]
    out = out.loc[~out.index.duplicated(keep="last")]
    return out.dropna()


def _calendar_from_pairs_data(
    pairs_data: Mapping[str, Any], *, ex_tz: str | None = None
) -> pd.DatetimeIndex:
    idxs: list[pd.DatetimeIndex] = []
    ref_tz = ex_tz or None

    def _push_idx(obj: Any) -> None:
        nonlocal ref_tz
        if isinstance(obj, pd.Series):
            idx = pd.DatetimeIndex(pd.to_datetime(obj.index, errors="coerce"))
            if idx.empty:
                return
            if ref_tz is None and idx.tz is not None:
                ref_tz = idx.tz
                if idxs:
                    idxs[:] = [
                        _normalize_index_to_calendar(i, pd.DatetimeIndex([], tz=ref_tz))
                        for i in idxs
                    ]
            idx = _normalize_index_to_calendar(idx, pd.DatetimeIndex([], tz=ref_tz))
            idx = idx[~idx.isna()]
            if not idx.empty:
                idxs.append(idx)

    for _pair, meta in (pairs_data or {}).items():
        if not isinstance(meta, Mapping):
            continue
        df_prices = meta.get("prices")
        if isinstance(df_prices, pd.DataFrame):
            if "y" in df_prices.columns:
                _push_idx(df_prices["y"])
            if "x" in df_prices.columns:
                _push_idx(df_prices["x"])
        y = meta.get("t1_price")
        if y is None:
            y = meta.get("y")
        if y is None:
            y = meta.get("y_price")
        _push_idx(y)
        x = meta.get("t2_price")
        if x is None:
            x = meta.get("x")
        if x is None:
            x = meta.get("x_price")
        _push_idx(x)

    if not idxs:
        return pd.DatetimeIndex([])
    cal = idxs[0]
    for idx in idxs[1:]:
        cal = cal.union(idx)
    return cal.sort_values()


def _build_train_inputs_from_pairs_data(
    pairs_data: Mapping[str, Any],
    *,
    cal: pd.DatetimeIndex,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not pairs_data:
        return out

    cfg_map: Mapping[str, Any] = cfg or {}
    sig_cfg = (
        cfg_map.get("signal", {}) if isinstance(cfg_map.get("signal"), Mapping) else {}
    )
    spread_cfg = (
        cfg_map.get("spread_zscore", {})
        if isinstance(cfg_map.get("spread_zscore"), Mapping)
        else {}
    )
    z_default = max(1, int(round(float(spread_cfg.get("z_window", 30)))))
    hold_default = max(1, int(sig_cfg.get("max_hold_days", 10)))

    def _pair_prices(meta: Mapping[str, Any]) -> tuple[pd.Series, pd.Series] | None:
        df_prices = meta.get("prices")
        if isinstance(df_prices, pd.DataFrame) and set(df_prices.columns) >= {"y", "x"}:
            y = df_prices["y"]
            x = df_prices["x"]
            return y, x
        y_raw = meta.get("t1_price")
        if y_raw is None:
            y_raw = meta.get("y")
        if y_raw is None:
            y_raw = meta.get("y_price")
        x_raw = meta.get("t2_price")
        if x_raw is None:
            x_raw = meta.get("x")
        if x_raw is None:
            x_raw = meta.get("x_price")
        if isinstance(y_raw, pd.Series) and isinstance(x_raw, pd.Series):
            return y_raw, x_raw
        return None

    pairs_data_map = pairs_data or {}
    for pair in sorted(pairs_data_map.keys(), key=str):
        meta = pairs_data_map.get(pair)
        if not isinstance(meta, Mapping):
            continue
        px = _pair_prices(meta)
        if px is None:
            continue
        y_raw, x_raw = px
        y = _slice_series(y_raw, cal[0], cal[-1], cal=cal)
        x = _slice_series(x_raw, cal[0], cal[-1], cal=cal)
        idx = y.index.union(x.index).sort_values()
        if idx.empty:
            continue
        y = y.reindex(idx).ffill()
        x = x.reindex(idx).ffill()
        df_xy = pd.DataFrame({"y": y, "x": x}).dropna()
        if df_xy.empty:
            continue
        y = df_xy["y"]
        x = df_xy["x"]
        meta_map = meta.get("meta") if isinstance(meta.get("meta"), Mapping) else {}
        cointegration: Mapping[str, Any] = (
            cast(Mapping[str, Any], meta_map.get("cointegration"))
            if isinstance(meta_map, Mapping)
            and isinstance(meta_map.get("cointegration"), Mapping)
            else {}
        )
        try:
            pair_z_window = int(cointegration.get("z_window", z_default))
        except Exception:
            pair_z_window = int(z_default)
        try:
            pair_max_hold = int(cointegration.get("max_hold_days", hold_default))
        except Exception:
            pair_max_hold = int(hold_default)
        try:
            pair_half_life = float(cointegration.get("half_life", np.nan))
        except Exception:
            pair_half_life = float("nan")
        out[str(pair)] = {
            "y": y.astype(float),
            "x": x.astype(float),
            "z_window": max(1, int(pair_z_window)),
            "max_hold_days": max(1, int(pair_max_hold)),
            "half_life": float(pair_half_life),
        }

    return out


def _build_train_inputs(
    *,
    prices: pd.DataFrame,
    pairs: Mapping[str, Any],
    pairs_data: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], pd.DatetimeIndex]:
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("prices must have a non-empty DatetimeIndex")

    bt = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), Mapping) else {}
    splits_any = bt.get("splits") if isinstance(bt.get("splits"), Mapping) else None
    if not isinstance(splits_any, Mapping) or "train" not in splits_any:
        raise KeyError("backtest.splits.train missing (required for BO)")

    splits = cast(Mapping[str, Any], splits_any)
    tr_any = splits.get("train")
    tr = tr_any if isinstance(tr_any, Mapping) else {}
    tr0 = align_ts_to_index(tr.get("start"), prices.index)
    tr1 = align_ts_to_index(tr.get("end"), prices.index)
    cal = prices.index[(prices.index >= tr0) & (prices.index <= tr1)]
    if cal.empty:
        raise ValueError("Empty training calendar from backtest.splits.train")

    pp_cfg = (
        cfg.get("pair_prefilter", {})
        if isinstance(cfg.get("pair_prefilter"), Mapping)
        else {}
    )
    prefilter_active = bool(pp_cfg.get("prefilter_active", False))
    coint_alpha = float(pp_cfg.get("coint_alpha", 0.05))
    min_obs = max(2, int(pp_cfg.get("min_obs", 30)))
    half_life_cfg = (
        resolve_half_life_cfg(pp_cfg.get("half_life")) if prefilter_active else None
    )
    sig_cfg = cfg.get("signal", {}) if isinstance(cfg.get("signal"), Mapping) else {}
    spread_cfg = (
        cfg.get("spread_zscore", {})
        if isinstance(cfg.get("spread_zscore"), Mapping)
        else {}
    )
    z_default = max(1, int(round(float(spread_cfg.get("z_window", 30)))))
    hold_default = max(1, int(sig_cfg.get("max_hold_days", 10)))

    if pairs_data:
        out = _build_train_inputs_from_pairs_data(pairs_data, cal=cal, cfg=cfg)
    else:
        out = {}
        pairs_map = pairs or {}
        for pair in sorted(pairs_map.keys(), key=str):
            meta = pairs_map.get(pair)
            if not isinstance(meta, Mapping):
                continue
            t1 = meta.get("t1") or meta.get("y")
            t2 = meta.get("t2") or meta.get("x")
            if not t1 or not t2:
                continue
            s1 = prices.get(str(t1)) if str(t1) in prices.columns else None
            s2 = prices.get(str(t2)) if str(t2) in prices.columns else None
            if not isinstance(s1, pd.Series) or not isinstance(s2, pd.Series):
                continue
            y = _slice_series(s1, cal[0], cal[-1], cal=cal)
            x = _slice_series(s2, cal[0], cal[-1], cal=cal)
            idx = y.index.union(x.index).sort_values()
            if idx.empty:
                continue
            y = y.reindex(idx).ffill()
            x = x.reindex(idx).ffill()
            df_xy = pd.DataFrame({"y": y, "x": x}).dropna()
            if df_xy.empty:
                continue
            y = df_xy["y"]
            x = df_xy["x"]
            pair_runtime = {
                "z_window": int(z_default),
                "max_hold_days": int(hold_default),
                "half_life": float("nan"),
            }
            if prefilter_active:
                coint_diag = evaluate_pair_cointegration(
                    pd.DataFrame({"y": y, "x": x}),
                    coint_alpha=coint_alpha,
                    min_obs=min_obs,
                    half_life_cfg=half_life_cfg,
                )
                if not bool(coint_diag.get("passed", False)):
                    continue
                pair_runtime = {
                    "z_window": max(
                        1, int(coint_diag.get("z_window", z_default) or z_default)
                    ),
                    "max_hold_days": max(
                        1,
                        int(
                            coint_diag.get("max_hold_days", hold_default)
                            or hold_default
                        ),
                    ),
                    "half_life": float(coint_diag.get("half_life", np.nan)),
                }
            out[str(pair)] = {
                "y": y.astype(float),
                "x": x.astype(float),
                **pair_runtime,
            }

    if not out:
        raise ValueError(
            "No valid pairs could be built for BO (missing symbols in prices?)"
        )
    return out, cal


def _as_range(v: Any, cast_fn=float) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        xs = [cast_fn(x) for x in v]
        return float(min(xs)), float(max(xs))
    if isinstance(v, (list, tuple)) and len(v) == 1:
        x = cast_fn(v[0])
        return float(x), float(x)
    x = cast_fn(v)
    return float(x), float(x)


def _markov_bo_defaults(cfg: Mapping[str, Any]) -> tuple[bool, float, int]:
    raw = (
        cfg.get("markov_filter", {})
        if isinstance(cfg.get("markov_filter"), Mapping)
        else {}
    )
    enabled = bool(raw.get("enabled", False))
    min_revert_prob = raw.get("min_revert_prob", raw.get("threshold", 0.5))
    horizon_days = raw.get("horizon_days", raw.get("horizon", 10))
    try:
        p_min = float(min_revert_prob)
    except Exception:
        p_min = 0.5
    p_min = float(np.clip(p_min, 0.0, 1.0))
    try:
        horizon = int(round(float(horizon_days)))
    except Exception:
        horizon = 10
    return enabled, p_min, max(1, int(horizon))


def _single_stage_bo_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    bo_cfg = cfg.get("bo", {}) if isinstance(cfg.get("bo"), Mapping) else {}
    legacy = [
        key
        for key in ("stage1", "stage2", "z_window_range", "precompute_beta_z")
        if key in bo_cfg
    ]
    if legacy:
        joined = ", ".join(f"bo.{key}" for key in legacy)
        raise ValueError(
            f"Legacy BO config keys are no longer supported: {joined}. "
            "Use root-level bo search keys such as bo.entry_z_range and bo.init_points."
        )
    return bo_cfg


def run_paper_bo_conservative(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None = None,
    pairs: Mapping[str, Any],
    pairs_data: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any],
    adv_map: Mapping[str, float] | None = None,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run the BO pipeline on the conservative training window:
      - uses backtest.splits.train only
      - returns best parameters + metadata
      - evaluates exactly one BO objective mode: fast or realistic
      - if pairs_data is provided, BO uses the same filtered universe as the backtest
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = _safe_int(cfg.get("seed", 42), 42)
    mode = _parse_bo_mode(cfg)
    cv = _parse_bo_cv(cfg, mode=mode)

    bo_cfg = _single_stage_bo_cfg(cfg)
    realistic_cfg = _parse_realistic_cfg(cfg)
    markov_enabled, markov_p0, markov_h0 = _markov_bo_defaults(cfg)
    if markov_enabled and mode != "realistic":
        raise ValueError(
            "Markov BO requires bo.mode='realistic' when markov_filter.enabled=true."
        )

    per_pair_prices, cal = _build_train_inputs(
        prices=prices, pairs=pairs, pairs_data=pairs_data, cfg=cfg
    )

    if mode == "realistic":
        if prices_panel is None:
            raise ValueError("bo.mode='realistic' requires prices_panel.")
        if pairs_data is None:
            raise ValueError("bo.mode='realistic' requires pairs_data.")
        if not pairs_data:
            raise ValueError("bo.mode='realistic' requires non-empty pairs_data.")

    spread_cfg = (
        cfg.get("spread_zscore", {})
        if isinstance(cfg.get("spread_zscore"), Mapping)
        else {}
    )
    try:
        z_default = int(round(float(spread_cfg.get("z_window", 30))))
    except Exception:
        z_default = 30
    if z_default <= 1:
        raise ValueError("spread_zscore.z_window must be > 1 for BO.")

    spreads: dict[str, pd.Series] = {}
    if mode == "fast":
        spreads = _precompute_spreads(
            per_pair_prices, cfg=cfg, z_window_for_beta=int(z_default)
        )
        if not spreads:
            raise ValueError("BO failed: could not precompute spreads for any pair")

    sig_cfg = cfg.get("signal", {}) if isinstance(cfg.get("signal"), Mapping) else {}
    entry0 = float(sig_cfg.get("entry_z", 2.0))
    exit0 = float(sig_cfg.get("exit_z", 0.5))
    stop0 = float(sig_cfg.get("stop_z", 2.0))
    hmax0 = int(sig_cfg.get("max_hold_days", 10))
    cool0 = int(sig_cfg.get("cooldown_days", 0))
    init_cap = float((cfg.get("backtest", {}) or {}).get("initial_capital", 1.0))

    e_lo, e_hi = _as_range(bo_cfg.get("entry_z_range", [entry0]), float)
    x_lo, x_hi = _as_range(bo_cfg.get("exit_z_range", [exit0]), float)
    s_lo, s_hi = _as_range(bo_cfg.get("stop_z_range", [stop0]), float)

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
                cfg=cfg,
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
                metric=realistic_cfg.metric,
                initial_capital=init_cap,
            )
        elif cv.enabled:
            sc = _fold_score_with_refit(
                per_pair_prices=per_pair_prices,
                calendar=cal,
                initial_capital=init_cap,
                cv=cv,
                seed=seed,
                component="theta_sig",
                out_dir=out_dir,
                params_for_log={
                    "entry_z": ez,
                    "exit_z": xz,
                    "stop_z": sz,
                },
                z_window=int(z_default),
                entry_z=ez,
                exit_z=xz,
                stop_z=sz,
                max_hold_days=hmax0,
                cooldown_days=cool0,
                cfg=cfg,
            )
        else:
            pnl_by_pair = _simulate_stage_pnl(
                spreads=spreads,
                per_pair_prices=per_pair_prices,
                z_window=z_default,
                entry_z=ez,
                exit_z=xz,
                stop_z=sz,
                max_hold_days=hmax0,
                cooldown_days=cool0,
                cfg=cfg,
                calendar=cal,
            )
            pnl = _portfolio_pnl_equal_weight(pnl_by_pair, cal)
            sc = _fold_score_from_pnl(
                pnl,
                calendar=cal,
                initial_capital=init_cap,
                cv=cv,
                seed=seed,
                component="theta_sig",
                out_dir=out_dir,
                params_for_log={
                    "entry_z": ez,
                    "exit_z": xz,
                    "stop_z": sz,
                },
            )
        cache_theta[key] = float(sc)
        return float(sc)

    pbounds = {
        "entry_z": (float(min(e_lo, e_hi)), float(max(e_lo, e_hi))),
        "exit_z": (float(min(x_lo, x_hi)), float(max(x_lo, x_hi))),
        "stop_z": (float(min(s_lo, s_hi)), float(max(s_lo, s_hi))),
    }

    if all(abs(pbounds[key][0] - pbounds[key][1]) < 1e-12 for key in pbounds):
        theta_hat = {
            "entry_z": float(pbounds["entry_z"][0]),
            "exit_z": float(pbounds["exit_z"][0]),
            "stop_z": float(pbounds["stop_z"][0]),
        }
        theta_score = _obj_theta(
            theta_hat["entry_z"],
            theta_hat["exit_z"],
            theta_hat["stop_z"],
        )
    else:
        best_theta, theta_score = _bayes_optimize(
            out_dir=out_dir,
            stage="theta_sig",
            pbounds=pbounds,
            objective=_obj_theta,
            seed=seed,
            init_points=_safe_int(bo_cfg.get("init_points", 8), 8),
            n_iter=_safe_int(bo_cfg.get("n_iter", 24), 24),
            patience=_safe_int(bo_cfg.get("patience", 0), 0),
        )
        theta_hat = {
            "entry_z": float(best_theta.get("entry_z", entry0)),
            "exit_z": float(best_theta.get("exit_z", exit0)),
            "stop_z": float(best_theta.get("stop_z", stop0)),
        }

    theta_sig_score = float(theta_score)
    theta_markov_hat: dict[str, Any] | None = None
    theta_markov_score: float | None = None

    if markov_enabled:
        p_lo, p_hi = _as_range(bo_cfg.get("min_revert_prob_range", [markov_p0]), float)
        h_lo, h_hi = _as_range(bo_cfg.get("horizon_days_range", [markov_h0]), float)
        p_lo = float(np.clip(min(p_lo, p_hi), 0.0, 1.0))
        p_hi = float(np.clip(max(p_lo, p_hi), 0.0, 1.0))
        h_lo = float(max(1.0, min(h_lo, h_hi)))
        h_hi = float(max(h_lo, max(h_lo, h_hi)))

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

            sc = _fold_score_realistic(
                cfg=cfg,
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
                metric=realistic_cfg.metric,
                initial_capital=init_cap,
            )
            cache_markov[key] = float(sc)
            return float(sc)

        pbounds_markov = {
            "min_revert_prob": (float(p_lo), float(p_hi)),
            "horizon_days": (float(h_lo), float(h_hi)),
        }

        if all(
            abs(pbounds_markov[key][0] - pbounds_markov[key][1]) < 1e-12
            for key in pbounds_markov
        ):
            theta_markov_hat = {
                "min_revert_prob": float(pbounds_markov["min_revert_prob"][0]),
                "horizon_days": max(
                    1, int(round(float(pbounds_markov["horizon_days"][0])))
                ),
            }
            theta_markov_score = _obj_markov(
                theta_markov_hat["min_revert_prob"],
                float(theta_markov_hat["horizon_days"]),
            )
        else:
            best_markov, theta_markov_score = _bayes_optimize(
                out_dir=out_dir,
                stage="theta_markov",
                pbounds=pbounds_markov,
                objective=_obj_markov,
                seed=seed,
                init_points=_safe_int(
                    bo_cfg.get("markov_init_points", bo_cfg.get("init_points", 8)), 8
                ),
                n_iter=_safe_int(
                    bo_cfg.get("markov_n_iter", bo_cfg.get("n_iter", 24)), 24
                ),
                patience=_safe_int(
                    bo_cfg.get("markov_patience", bo_cfg.get("patience", 0)), 0
                ),
            )
            theta_markov_hat = {
                "min_revert_prob": float(
                    np.clip(
                        float(best_markov.get("min_revert_prob", markov_p0)), 0.0, 1.0
                    )
                ),
                "horizon_days": max(
                    1, int(round(float(best_markov.get("horizon_days", markov_h0))))
                ),
            }
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
    (out_dir / "bo_best.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return res
