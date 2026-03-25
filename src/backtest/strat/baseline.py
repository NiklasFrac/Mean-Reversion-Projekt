from __future__ import annotations

import logging
import math
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.config.types import SignalConfig, SpreadZscoreConfig
from backtest.strat.markov_filter import build_markov_entry_filter
from backtest.utils.pairs import get_tickers_from_meta
from backtest.utils.pair_analysis import (
    coerce_ts_like_index,
    estimate_beta_ols_with_intercept_details,
    frozen_zscore_stats,
    prior_train_history,
    resolve_train_index,
    rolling_zscore_stats_on_allowed_dates,
)

logger = logging.getLogger("backtest.strat.baseline")

_SIGNAL_DEFAULTS = SignalConfig()
_SPREAD_ZSCORE_DEFAULTS = SpreadZscoreConfig()


def _finite_float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _entry_gate_allows(entry_gate: pd.Series | None, ts: Any) -> bool:
    if entry_gate is None:
        return True
    try:
        gate_raw = entry_gate.at[ts]
    except Exception:
        return True
    return True if pd.isna(gate_raw) else bool(gate_raw)


def _entry_signal_from_z(
    zval: float,
    *,
    prev_z: float,
    entry_z: float,
    stop_z: float,
    gate_ok: bool,
) -> int:
    if not gate_ok or not np.isfinite(zval):
        return 0
    entry_abs = float(abs(entry_z))
    stop_abs = float(abs(stop_z))
    is_fresh_long = not np.isfinite(prev_z) or prev_z > -entry_abs
    is_fresh_short = not np.isfinite(prev_z) or prev_z < entry_abs
    if zval <= -entry_abs and zval > -stop_abs and is_fresh_long:
        return 1
    if zval >= entry_abs and zval < stop_abs and is_fresh_short:
        return -1
    return 0


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
    prev = float("nan")

    for t in idx:
        ts = pd.Timestamp(t)
        if ts < test_start or ts > entry_end:
            prev = float("nan")
            continue

        zval = _finite_float_or_nan(z.get(t))
        if not np.isfinite(zval):
            prev = float("nan")
            continue

        signal = _entry_signal_from_z(
            zval,
            prev_z=prev,
            entry_z=entry_z,
            stop_z=stop_z,
            gate_ok=_entry_gate_allows(entry_gate, t),
        )
        if signal != 0:
            rows.append(
                {
                    "signal_date": pd.Timestamp(ts),
                    "signal": int(signal),
                    "z_signal": float(zval),
                }
            )
        prev = zval

    return pd.DataFrame(rows)


class BaselineZScoreStrategy:
 

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

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
        entry_z = float(sig_cfg.get("entry_z", _SIGNAL_DEFAULTS.entry_z))
        exit_z = float(sig_cfg.get("exit_z", _SIGNAL_DEFAULTS.exit_z))
        stop_z = float(sig_cfg.get("stop_z", _SIGNAL_DEFAULTS.stop_z))
        max_hold_days_default = int(
            sig_cfg.get("max_hold_days", _SIGNAL_DEFAULTS.max_hold_days)
        )
        cooldown_days = int(sig_cfg.get("cooldown_days", _SIGNAL_DEFAULTS.cooldown_days))

        sz_cfg = (
            self.cfg.get("spread_zscore", {})
            if isinstance(self.cfg.get("spread_zscore"), dict)
            else {}
        )
        w_sig_default = max(1, int(sz_cfg.get("z_window", _SPREAD_ZSCORE_DEFAULTS.z_window)))
        z_min_periods_cfg = sz_cfg.get("z_min_periods")
        freeze_stats = bool(
            sz_cfg.get("freeze_stats", _SPREAD_ZSCORE_DEFAULTS.freeze_stats)
        )

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
            if not (
                isinstance(df_prices, pd.DataFrame)
                and set(df_prices.columns) >= {"y", "x"}
            ):
                continue
            df = df_prices.loc[:, ["y", "x"]].copy()

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[~df.index.isna()].sort_index()

            # Coerce split boundaries to the pair's index tz (tz-aware vs tz-naive).
            idx = pd.DatetimeIndex(df.index)
            train_start = coerce_ts_like_index(train_start_raw, idx)
            train_end = coerce_ts_like_index(train_end_raw, idx)
            test_start = coerce_ts_like_index(test_start_raw, idx)
            entry_end = coerce_ts_like_index(entry_end_raw, idx)
            exit_end = coerce_ts_like_index(exit_end_raw, idx)
            if exit_end < entry_end:
                exit_end = entry_end
            train_index = resolve_train_index(
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

            beta_hat, beta_reason = estimate_beta_ols_with_intercept_details(
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
            prior_train_index = prior_train_history(train_index, eval_index=eval_index)
            allowed_index = prior_train_index.union(eval_index).sort_values()
            if freeze_stats:
                z, _, sigma_t, ok = frozen_zscore_stats(
                    spread, train_index=pd.DatetimeIndex(df_train.index)
                )
                if not ok:
                    z, _, sigma_t = rolling_zscore_stats_on_allowed_dates(
                        spread,
                        allowed_index=allowed_index,
                        window=pair_w_sig,
                        min_periods=pair_w_min,
                        full_index=df.index,
                    )
            else:
                z, _, sigma_t = rolling_zscore_stats_on_allowed_dates(
                    spread,
                    allowed_index=allowed_index,
                    window=pair_w_sig,
                    min_periods=pair_w_min,
                    full_index=df.index,
                )
            markov_filter = build_markov_entry_filter(
                self.cfg,
                z=z,
                train_index=prior_train_index,
                eval_index=eval_index,
                entry_z=entry_z,
                exit_z=exit_z,
            )
            tickers = get_tickers_from_meta(data)
            if not tickers:
                continue
            t1_sym, t2_sym = tickers[0], tickers[1]
            pair_key = f"{t1_sym}-{t2_sym}"
            volatility_window = (
                int(pair_w_sig)
                if pair_z_window_as_volatility_window
                else int(
                    sig_cfg.get(
                        "volatility_window", _SIGNAL_DEFAULTS.volatility_window
                    )
                )
            )

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
                    "volatility_window": int(volatility_window),
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
