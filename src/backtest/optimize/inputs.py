from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

import backtest.utils.pair_analysis as _pair_analysis
from backtest.config.cfg import AppConfig
from backtest.utils.tz import align_ts_to_index, assert_ny_series, same_tz_or_raise

evaluate_pair_cointegration = _pair_analysis.evaluate_pair_cointegration


@dataclass(frozen=True)
class TrainInputs:
    per_pair_prices: dict[str, dict[str, Any]]
    calendar: pd.DatetimeIndex


def _coerce_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_index_to_calendar(
    idx: pd.DatetimeIndex, cal: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce"))
    assert_ny_series(pd.Series(index=idx, dtype=float), context="optimize index")
    same_tz_or_raise(idx, cal, context="_normalize_index_to_calendar")
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


def calendar_from_pairs_data(pairs_data: Mapping[str, Any]) -> pd.DatetimeIndex:
    idxs: list[pd.DatetimeIndex] = []

    def _push_idx(obj: Any) -> None:
        if isinstance(obj, pd.Series):
            assert_ny_series(obj, context="calendar_from_pairs_data")
            idx = pd.DatetimeIndex(obj.index)
            if not idx.empty:
                idxs.append(idx)

    for meta in (pairs_data or {}).values():
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


def build_train_inputs_from_pairs_data(
    pairs_data: Mapping[str, Any],
    *,
    cal: pd.DatetimeIndex,
    cfg: AppConfig | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not pairs_data:
        return out

    z_default = max(1, int(cfg.spread_zscore.z_window)) if cfg is not None else 30
    hold_default = max(1, int(cfg.signal.max_hold_days)) if cfg is not None else 10

    def _pair_prices(meta: Mapping[str, Any]) -> tuple[pd.Series, pd.Series] | None:
        df_prices = meta.get("prices")
        if isinstance(df_prices, pd.DataFrame) and set(df_prices.columns) >= {"y", "x"}:
            return df_prices["y"], df_prices["x"]
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

    for pair in sorted((pairs_data or {}).keys(), key=str):
        meta = pairs_data.get(pair)
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
        beta_meta = _coerce_float_or_none(cointegration.get("beta"))
        beta_ok = bool(
            beta_meta is not None and np.isfinite(beta_meta) and beta_meta > 0.0
        )
        if not beta_ok:
            beta_hat, _ = _pair_analysis.estimate_beta_ols_with_intercept_details(y, x)
            if beta_hat is None:
                continue
        out[str(pair)] = {
            "y": y.astype(float),
            "x": x.astype(float),
            "z_window": max(1, int(pair_z_window)),
            "max_hold_days": max(1, int(pair_max_hold)),
            "half_life": float(pair_half_life),
        }

    return out


def build_train_inputs(
    *,
    prices: pd.DataFrame,
    pairs: Mapping[str, Any],
    pairs_data: Mapping[str, Any] | None = None,
    cfg: AppConfig,
) -> TrainInputs:
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("prices must have a non-empty DatetimeIndex")

    if "train" not in cfg.backtest.splits:
        raise KeyError("backtest.splits.train missing (required for BO)")
    train_split = cfg.backtest.splits["train"]
    tr0 = align_ts_to_index(train_split.start, prices.index)
    tr1 = align_ts_to_index(train_split.end, prices.index)
    cal = prices.index[(prices.index >= tr0) & (prices.index <= tr1)]
    if cal.empty:
        raise ValueError("Empty training calendar from backtest.splits.train")

    pp_cfg = cfg.pair_prefilter
    prefilter_active = bool(pp_cfg.prefilter_active)
    coint_alpha = float(pp_cfg.coint_alpha)
    min_obs = max(2, int(pp_cfg.min_obs))
    half_life_cfg = (
        _pair_analysis.resolve_half_life_cfg(
            {
                "min_days": pp_cfg.half_life.min_days,
                "max_days": pp_cfg.half_life.max_days,
                "max_hold_multiple": pp_cfg.half_life.max_hold_multiple,
                "min_derived_days": pp_cfg.half_life.min_derived_days,
            }
        )
        if prefilter_active
        else None
    )
    z_default = max(1, int(cfg.spread_zscore.z_window))
    hold_default = max(1, int(cfg.signal.max_hold_days))

    if pairs_data:
        per_pair_prices = build_train_inputs_from_pairs_data(pairs_data, cal=cal, cfg=cfg)
    else:
        per_pair_prices: dict[str, dict[str, Any]] = {}
        for pair in sorted((pairs or {}).keys(), key=str):
            meta = pairs.get(pair)
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
            else:
                beta_hat, _ = _pair_analysis.estimate_beta_ols_with_intercept_details(
                    y, x
                )
                if beta_hat is None:
                    continue
            per_pair_prices[str(pair)] = {
                "y": y.astype(float),
                "x": x.astype(float),
                **pair_runtime,
            }

    if not per_pair_prices:
        raise ValueError("No valid pairs could be built for BO (missing symbols in prices?)")
    return TrainInputs(per_pair_prices=per_pair_prices, calendar=cal)


def prices_frame_from_pairs_data(
    pairs_data: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    cal = calendar_from_pairs_data(pairs_data)
    if cal.empty:
        raise ValueError("pairs_data does not contain usable price series.")

    per_pair_prices = build_train_inputs_from_pairs_data(pairs_data, cal=cal)
    if not per_pair_prices:
        raise ValueError("pairs_data does not contain usable price series.")

    series_by_sym: dict[str, pd.Series] = {}
    pairs_map: dict[str, dict[str, str]] = {}

    def _sym(v: Any) -> str | None:
        s = str(v or "").strip().upper()
        return s or None

    def _merge_series(existing: pd.Series | None, new: pd.Series) -> pd.Series:
        if existing is None:
            return new
        merged = existing.combine_first(new)
        return merged if merged.count() >= max(existing.count(), new.count()) else (
            existing if existing.count() >= new.count() else new
        )

    for pair, meta in (pairs_data or {}).items():
        if not isinstance(meta, Mapping):
            continue
        raw_meta = meta.get("meta") if isinstance(meta.get("meta"), Mapping) else meta
        t1 = _sym(raw_meta.get("t1") if isinstance(raw_meta, Mapping) else None)
        t2 = _sym(raw_meta.get("t2") if isinstance(raw_meta, Mapping) else None)
        if t1 is None or t2 is None:
            p = str(pair)
            for sep in ("-", "/", "_", "|", ":"):
                if sep in p:
                    a, b = p.split(sep, 1)
                    t1 = _sym(a)
                    t2 = _sym(b)
                    break
        if t1 is None or t2 is None:
            continue
        pair_key = str(pair)
        if pair_key not in per_pair_prices:
            continue
        yz = per_pair_prices[pair_key]
        y = yz.get("y")
        x = yz.get("x")
        if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
            continue
        series_by_sym[t1] = _merge_series(series_by_sym.get(t1), y)
        series_by_sym[t2] = _merge_series(series_by_sym.get(t2), x)
        pairs_map[pair_key] = {"t1": t1, "t2": t2}

    if not series_by_sym or not pairs_map:
        raise ValueError("pairs_data does not contain usable price series.")

    prices_df = pd.concat(
        {k: pd.to_numeric(v, errors="coerce") for k, v in series_by_sym.items()},
        axis=1,
    )
    return prices_df, pairs_map
