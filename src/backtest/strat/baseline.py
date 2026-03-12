from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from backtest.strat.markov_filter import build_markov_entry_filter
from backtest.utils import strategy as _helpers
from backtest.run.trade_builder import TradeBuilder, _trades_to_orders

logger = logging.getLogger("backtest.strat.baseline")


def _get_tickers_from_meta(data: dict[str, Any]) -> tuple[str, str] | None:
    return _helpers.get_tickers_from_meta(data)


def _estimate_beta_ols_with_intercept(y: pd.Series, x: pd.Series) -> float:
    return _helpers.estimate_beta_ols_with_intercept(
        y,
        x,
        ols_cls=OLS,
        add_constant_fn=add_constant,
    )


def _estimate_positive_beta_ols_with_intercept(
    y: pd.Series, x: pd.Series
) -> tuple[float | None, str | None]:
    return _helpers.estimate_beta_ols_with_intercept_details(
        y,
        x,
        ols_cls=OLS,
        add_constant_fn=add_constant,
    )


def _rolling_zscore_past_only(
    spread: pd.Series, *, window: int, min_periods: int
) -> pd.Series:
    return _helpers.rolling_zscore_past_only(
        spread,
        window=window,
        min_periods=min_periods,
    )


def _prior_train_history(
    train_index: pd.DatetimeIndex, *, eval_index: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    return _helpers.prior_train_history(train_index, eval_index=eval_index)


def _rolling_zscore_on_allowed_dates(
    spread: pd.Series,
    *,
    allowed_index: pd.DatetimeIndex,
    window: int,
    min_periods: int,
    full_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    return _helpers.rolling_zscore_on_allowed_dates(
        spread,
        allowed_index=allowed_index,
        window=window,
        min_periods=min_periods,
        full_index=full_index,
    )


def _rolling_zscore_stats_on_allowed_dates(
    spread: pd.Series,
    *,
    allowed_index: pd.DatetimeIndex,
    window: int,
    min_periods: int,
    full_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    return _helpers.rolling_zscore_stats_on_allowed_dates(
        spread,
        allowed_index=allowed_index,
        window=window,
        min_periods=min_periods,
        full_index=full_index,
    )


def _frozen_zscore(
    spread: pd.Series, *, train_index: pd.DatetimeIndex
) -> tuple[pd.Series, bool]:
    return _helpers.frozen_zscore(
        spread,
        train_index=train_index,
    )


def _frozen_zscore_stats(
    spread: pd.Series,
    *,
    train_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    return _helpers.frozen_zscore_stats(
        spread,
        train_index=train_index,
    )


def _positions_from_z(
    z: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_days: int,
    cooldown_days: int,
    test_start: pd.Timestamp,
    entry_end: pd.Timestamp,
    allow_exit_after_end: bool,
    entry_gate: pd.Series | None = None,
) -> pd.Series:
    idx = z.index
    out = pd.Series(0, index=idx, dtype=int)
    pos = 0
    held = 0
    cool_left = 0
    e = float(abs(entry_z))
    x = float(abs(exit_z))
    s = float(abs(stop_z))
    mh = int(max_hold_days)
    cool = int(max(0, cooldown_days))
    prev = float("nan")

    for t in idx:
        ts = pd.Timestamp(t)
        if ts < test_start:
            pos = 0
            held = 0
            cool_left = 0
            out.at[t] = 0
            prev = float("nan")
            continue
        if not allow_exit_after_end and ts > entry_end:
            pos = 0
            held = 0
            cool_left = 0
            out.at[t] = 0
            prev = float("nan")
            continue
        if pos == 0 and ts > entry_end:
            out.at[t] = 0
            prev = float("nan") if not np.isfinite(float(z.at[t])) else float(z.at[t])
            continue
        if cool_left > 0:
            out.at[t] = 0
            cool_left -= 1
            prev = float("nan") if not np.isfinite(float(z.at[t])) else float(z.at[t])
            continue
        zt = z.at[t]
        try:
            zval = float(zt)
        except Exception:
            zval = float("nan")

        if not np.isfinite(zval):
            if pos != 0:
                pos = 0
                held = 0
                cool_left = cool
            out.at[t] = 0
            prev = float("nan")
            continue

        if pos == 0:
            held = 0
            gate_ok = True
            if entry_gate is not None:
                try:
                    gate_raw = entry_gate.at[t]
                    gate_ok = True if pd.isna(gate_raw) else bool(gate_raw)
                except Exception:
                    gate_ok = True
            long_entry = (
                gate_ok
                and zval <= -e
                and zval > -s
                and (not np.isfinite(prev) or prev > -e)
            )
            short_entry = (
                gate_ok
                and zval >= e
                and zval < s
                and (not np.isfinite(prev) or prev < e)
            )
            if long_entry:
                pos = 1
            elif short_entry:
                pos = -1
            out.at[t] = pos
            prev = zval
            continue

        held += 1
        if abs(zval) <= x or abs(zval) >= s or (mh > 0 and held >= mh):
            pos = 0
            held = 0
            cool_left = cool
            out.at[t] = 0
            prev = zval
            continue

        if (pos > 0 and zval >= e) or (pos < 0 and zval <= -e):
            pos = 0
            held = 0
            cool_left = cool
            out.at[t] = 0
            prev = zval
            continue

        out.at[t] = pos
        prev = zval

    return out


def _entry_intents_from_z(
    z: pd.Series,
    *,
    entry_z: float,
    stop_z: float,
    test_start: pd.Timestamp,
    entry_end: pd.Timestamp,
    entry_gate: pd.Series | None = None,
) -> pd.DataFrame:
    idx = pd.DatetimeIndex(z.index)
    rows: list[dict[str, Any]] = []
    e = float(abs(entry_z))
    s = float(abs(stop_z))
    prev = float("nan")

    for t in idx:
        ts = pd.Timestamp(t)
        if ts < test_start or ts > entry_end:
            prev = float("nan")
            continue

        z_raw = z.get(t)
        try:
            zval = float(z_raw)
        except Exception:
            zval = float("nan")
        if not np.isfinite(zval):
            prev = float("nan")
            continue

        gate_ok = True
        if entry_gate is not None:
            try:
                gate_raw = entry_gate.at[t]
                gate_ok = True if pd.isna(gate_raw) else bool(gate_raw)
            except Exception:
                gate_ok = True

        long_entry = (
            gate_ok
            and zval <= -e
            and zval > -s
            and (not np.isfinite(prev) or prev > -e)
        )
        short_entry = (
            gate_ok
            and zval >= e
            and zval < s
            and (not np.isfinite(prev) or prev < e)
        )
        if long_entry or short_entry:
            rows.append(
                {
                    "signal_date": pd.Timestamp(ts),
                    "signal": int(1 if long_entry else -1),
                    "z_signal": float(zval),
                }
            )
        prev = zval

    return pd.DataFrame(rows)


def _coerce_ts_like_index(ts: Any, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return _helpers.coerce_ts_like_index(ts, idx)


def _resolve_train_index(
    cfg: dict[str, Any],
    *,
    idx: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> pd.DatetimeIndex:
    return _helpers.resolve_train_index(
        cfg, idx=idx, train_start=train_start, train_end=train_end
    )


class BaselineZScoreStrategy:
    """
    Baseline strategy as described in `administration/baseline_strategy.txt`.
    """

    def __init__(self, cfg: dict[str, Any], *, borrow_ctx: Any | None = None):
        self.cfg = cfg
        self.borrow_ctx = borrow_ctx
        self.initial_capital = float(
            (cfg.get("backtest") or {}).get("initial_capital", 1_000_000.0)
        )

    def __call__(
        self, pairs_data: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        bt = (
            self.cfg.get("backtest", {})
            if isinstance(self.cfg.get("backtest"), dict)
            else {}
        )
        splits = bt.get("splits") if isinstance(bt.get("splits"), dict) else None
        if (
            not isinstance(splits, dict)
            or "train" not in splits
            or "test" not in splits
        ):
            raise KeyError(
                "BaselineZScoreStrategy requires backtest.splits.{train,test}"
            )
        strat_cfg = (
            self.cfg.get("strategy", {})
            if isinstance(self.cfg.get("strategy"), dict)
            else {}
        )
        pair_z_window_as_volatility_window = bool(
            strat_cfg.get("pair_z_window_as_volatility_window", False)
        )

        train = splits.get("train") or {}
        test = splits.get("test") or {}
        train_start_raw = pd.to_datetime(cast(Any, train.get("start")))
        train_end_raw = pd.to_datetime(cast(Any, train.get("end")))
        test_start_raw = pd.to_datetime(cast(Any, test.get("start")))
        entry_end_raw = pd.to_datetime(
            cast(Any, test.get("entry_end", test.get("end")))
        )
        exit_end_raw = pd.to_datetime(cast(Any, test.get("exit_end", test.get("end"))))

        sig_cfg = (
            self.cfg.get("signal", {})
            if isinstance(self.cfg.get("signal"), dict)
            else {}
        )
        entry_z = float(sig_cfg.get("entry_z", sig_cfg.get("min_zscore", 2.0)))
        exit_z = float(sig_cfg.get("exit_z", sig_cfg.get("exit_threshold", 0.5)))
        stop_z = float(sig_cfg.get("stop_z", sig_cfg.get("stop_threshold", 2.0)))
        max_hold_days_default = int(sig_cfg.get("max_hold_days", 10))
        cooldown_days = int(sig_cfg.get("cooldown_days", 0))

        sz_cfg = (
            self.cfg.get("spread_zscore", {})
            if isinstance(self.cfg.get("spread_zscore"), dict)
            else {}
        )
        w_sig_default = max(1, int(sz_cfg.get("z_window", 60)))
        z_min_periods_cfg = sz_cfg.get("z_min_periods")
        freeze_stats = bool(sz_cfg.get("freeze_stats", False))

        results: dict[str, dict[str, Any]] = {}
        for pair, data in (pairs_data or {}).items():
            if not isinstance(data, dict):
                continue

            meta: Mapping[str, Any] = (
                cast(Mapping[str, Any], data.get("meta"))
                if isinstance(data.get("meta"), dict)
                else {}
            )
            adv_t1 = meta.get("adv_t1")
            adv_t2 = meta.get("adv_t2")
            coint_meta: Mapping[str, Any] = (
                cast(Mapping[str, Any], meta.get("cointegration"))
                if isinstance(meta.get("cointegration"), dict)
                else {}
            )

            try:
                pair_w_sig = int(coint_meta.get("z_window", w_sig_default))
            except Exception:
                pair_w_sig = int(w_sig_default)
            pair_w_sig = max(1, int(pair_w_sig))

            if z_min_periods_cfg is not None:
                try:
                    pair_w_min = int(z_min_periods_cfg)
                except Exception:
                    pair_w_min = int(math.ceil(0.5 * float(pair_w_sig)))
                pair_w_min = max(1, min(pair_w_sig, int(pair_w_min)))
            else:
                pair_w_min = max(1, int(math.ceil(0.5 * float(pair_w_sig))))

            try:
                pair_max_hold_days = int(
                    coint_meta.get("max_hold_days", max_hold_days_default)
                )
            except Exception:
                pair_max_hold_days = int(max_hold_days_default)
            pair_max_hold_days = max(1, int(pair_max_hold_days))

            df_prices = data.get("prices")
            if isinstance(df_prices, pd.DataFrame) and set(df_prices.columns) >= {
                "y",
                "x",
            }:
                df = df_prices.loc[:, ["y", "x"]].copy()
            else:
                y = data.get("t1_price")
                x = data.get("t2_price")
                if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
                    continue
                df = pd.DataFrame({"y": y, "x": x})

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[~df.index.isna()].sort_index()

            # Coerce split boundaries to the pair's index tz (tz-aware vs tz-naive).
            idx = pd.DatetimeIndex(df.index)
            train_start = _coerce_ts_like_index(train_start_raw, idx)
            train_end = _coerce_ts_like_index(train_end_raw, idx)
            test_start = _coerce_ts_like_index(test_start_raw, idx)
            entry_end = _coerce_ts_like_index(entry_end_raw, idx)
            exit_end = _coerce_ts_like_index(exit_end_raw, idx)
            if exit_end < entry_end:
                exit_end = entry_end
            train_index = _resolve_train_index(
                self.cfg, idx=idx, train_start=train_start, train_end=train_end
            )

            lower = min(train_start, test_start)
            upper = (
                max(exit_end, train_index.max()) if not train_index.empty else exit_end
            )
            df = df[(df.index >= lower) & (df.index <= upper)]
            if df.empty:
                continue
            df = df.dropna(subset=["y", "x"])
            if df.empty:
                continue

            df_train = df.reindex(train_index).dropna(subset=["y", "x"])
            if df_train.empty:
                continue
            if len(df_train) < max(10, pair_w_min):
                continue
            if (
                float(df_train["y"].std(ddof=0) or 0.0) <= 0.0
                or float(df_train["x"].std(ddof=0) or 0.0) <= 0.0
            ):
                continue

            beta_hat, beta_reason = _estimate_positive_beta_ols_with_intercept(
                df_train["y"], df_train["x"]
            )
            if beta_hat is None:
                logger.debug(
                    "baseline: skipped %s due to invalid train beta (%s)",
                    pair,
                    beta_reason or "beta_estimation_failed",
                )
                continue
            beta_series = pd.Series(float(beta_hat), index=df.index, name="beta")
            spread = (df["y"] - beta_series * df["x"]).rename("spread")
            eval_index = df.index[(df.index >= test_start) & (df.index <= exit_end)]
            if eval_index.empty:
                continue
            if freeze_stats:
                z, _, sigma_t, ok = _frozen_zscore_stats(
                    spread, train_index=pd.DatetimeIndex(df_train.index)
                )
                if not ok:
                    allowed = (
                        _prior_train_history(train_index, eval_index=eval_index)
                        .union(eval_index)
                        .sort_values()
                    )
                    z, _, sigma_t = _rolling_zscore_stats_on_allowed_dates(
                        spread,
                        allowed_index=allowed,
                        window=pair_w_sig,
                        min_periods=pair_w_min,
                        full_index=df.index,
                    )
            else:
                allowed = (
                    _prior_train_history(train_index, eval_index=eval_index)
                    .union(eval_index)
                    .sort_values()
                )
                z, _, sigma_t = _rolling_zscore_stats_on_allowed_dates(
                    spread,
                    allowed_index=allowed,
                    window=pair_w_sig,
                    min_periods=pair_w_min,
                    full_index=df.index,
                )
            markov_filter = build_markov_entry_filter(
                self.cfg,
                z=z,
                train_index=_prior_train_history(train_index, eval_index=eval_index),
                eval_index=eval_index,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            tickers = _get_tickers_from_meta(data)
            if not tickers:
                continue
            t1_sym, t2_sym = tickers[0], tickers[1]
            pair_key = f"{t1_sym}-{t2_sym}"

            cfg_pair = deepcopy(self.cfg)
            cfg_pair["_z_cache"] = z
            cfg_pair["_sigma_cache"] = sigma_t
            cfg_pair["_pair_key"] = pair_key
            cfg_pair["_t1_symbol"] = t1_sym
            cfg_pair["_t2_symbol"] = t2_sym
            cfg_pair.setdefault("signal", {})
            cfg_pair.setdefault("spread_zscore", {})
            signal_cfg = cast(dict[str, Any], cfg_pair["signal"])
            spread_cfg = cast(dict[str, Any], cfg_pair["spread_zscore"])
            spread_cfg["z_window"] = int(pair_w_sig)
            spread_cfg["z_min_periods"] = int(pair_w_min)
            signal_cfg["exit_z"] = float(exit_z)
            signal_cfg["stop_z"] = float(stop_z)
            signal_cfg["max_hold_days"] = int(pair_max_hold_days)
            signal_cfg["cooldown_days"] = int(cooldown_days)
            if pair_z_window_as_volatility_window:
                signal_cfg["volatility_window"] = int(pair_w_sig)

            intents_df = _entry_intents_from_z(
                z,
                entry_z=entry_z,
                stop_z=stop_z,
                test_start=test_start,
                entry_end=entry_end,
                entry_gate=markov_filter.entry_gate,
            )
            if intents_df.empty:
                continue
            intents_df = intents_df.copy()
            intents_df.insert(0, "pair", pair_key)
            intents_df["entry_end"] = pd.Timestamp(entry_end)
            intents_df["exit_end"] = pd.Timestamp(exit_end)

            results[pair] = {
                "intents": intents_df,
                "state": {
                    "pair_key": pair_key,
                    "y_symbol": t1_sym,
                    "x_symbol": t2_sym,
                    "prices": df.copy(),
                    "beta": beta_series.copy(),
                    "z": z.copy(),
                    "sigma": sigma_t.copy(),
                    "entry_z": float(entry_z),
                    "exit_z": float(exit_z),
                    "stop_z": float(stop_z),
                    "volatility_window": int(
                        pair_w_sig
                        if pair_z_window_as_volatility_window
                        else int(signal_cfg.get("volatility_window", pair_w_sig))
                    ),
                    "max_hold_days": int(pair_max_hold_days),
                    "cooldown_days": int(cooldown_days),
                    "test_start": pd.Timestamp(test_start),
                    "entry_end": pd.Timestamp(entry_end),
                    "exit_end": pd.Timestamp(exit_end),
                    "adv_t1": float(adv_t1) if adv_t1 is not None else None,
                    "adv_t2": float(adv_t2) if adv_t2 is not None else None,
                },
                "markov_filter": markov_filter.diagnostics,
            }

        return results
