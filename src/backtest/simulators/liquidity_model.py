"""
Symbol-/date-aware liquidity heuristics for LOB execution (free-data compatible).

Inputs supported (free data):
- Daily OHLCV panel (MultiIndex columns: (symbol, field))
- Optional ADV map in USD (symbol -> dollar ADV)

Outputs:
- Per-symbol, per-date book parameters that scale spread/depth/dynamics with liquidity & volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.simulators.common import price_at_or_prior
from backtest.utils.tz import NY_TZ, align_ts_to_index, coerce_ts_to_tz

__all__ = ["LiquidityModel", "LiquidityModelCfg"]


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _infer_panel_field_level(cols: pd.MultiIndex) -> int:
    field_tokens = {
        "close",
        "adj_close",
        "open",
        "high",
        "low",
        "volume",
        "vwap",
        "price",
    }
    best_level = 0
    best_hits = -1
    for lvl in range(cols.nlevels):
        try:
            vals = {str(v).lower() for v in cols.get_level_values(lvl).unique()}
        except Exception:
            continue
        hits = len(vals.intersection(field_tokens))
        if hits > best_hits:
            best_hits = hits
            best_level = lvl
    return int(best_level)


def _to_ny_ts(ts: pd.Timestamp, *, tz: str = NY_TZ) -> pd.Timestamp:
    return coerce_ts_to_tz(ts, tz)


@dataclass(frozen=True)
class LiquidityModelCfg:
    enabled: bool = True
    # Shift series lookups by N bars to keep signals past-only (daily bars: 1 = prior session).
    asof_shift: int = 1

    # Rolling windows (daily bars)
    vol_window: int = 30
    adv_window: int = 60
    min_periods_frac: float = 0.5

    # Spread model (bps)
    spread_floor_bps: float = 0.5
    spread_sigma_mult: float = 0.005  # daily sigma (log-return std) -> bps
    spread_adv_mult: float = 15.0  # bps * 1/sqrt($ADV in $1m)
    adv_ref_usd: float = 1e6

    # Depth model (shares at top level)
    depth_frac_of_adv_shares: float = 0.001  # 0.1% of ADV-shares as top depth
    depth_gamma: float = 1.0  # decay across levels: size_i ~ size_1 / i^gamma
    min_depth_shares: int = 25
    max_depth_shares: int = 250_000
    min_level_shares: int = 1

    # Dynamics scaling (book evolution between entry/exit)
    lam_adv_power: float = 0.25
    lam_min: float = 0.25
    lam_max: float = 25.0

    cancel_base: float = 0.15
    cancel_sigma_mult: float = 1.5
    cancel_min: float = 0.01
    cancel_max: float = 0.90

    max_add_frac_of_top: float = 0.50
    max_cancel_frac_of_top: float = 0.25
    max_add_min: int = 50
    max_cancel_min: int = 25
    max_add_max: int = 50_000
    max_cancel_max: int = 50_000

    # Tick-size heuristic for sub-dollar names (optional)
    tick_subpenny: float = 0.0001
    tick_penny: float = 0.01
    tick_switch_price: float = 1.0

    # Stress regime model
    stress_enabled: bool = True
    stress_intensity: float = 1.0
    stress_max_entry_delay_days: int = 1
    stress_max_exit_grace_days: int = 2
    stress_panic_cross_bps: float = 50.0

    @staticmethod
    def from_exec_lob(exec_lob: Mapping[str, Any] | None) -> "LiquidityModelCfg":
        d = dict(exec_lob or {})
        lm = d.get("liq_model") if isinstance(d.get("liq_model"), Mapping) else {}
        lm = dict(lm or {})
        sm = d.get("stress_model") if isinstance(d.get("stress_model"), Mapping) else {}
        sm = dict(sm or {})
        return LiquidityModelCfg(
            enabled=bool(lm.get("enabled", True)),
            asof_shift=_safe_int(lm.get("asof_shift", 1), 1),
            vol_window=_safe_int(lm.get("vol_window", 30), 30),
            adv_window=_safe_int(lm.get("adv_window", 60), 60),
            min_periods_frac=_safe_float(lm.get("min_periods_frac", 0.5), 0.5),
            spread_floor_bps=_safe_float(lm.get("spread_floor_bps", 0.5), 0.5),
            spread_sigma_mult=_safe_float(lm.get("spread_sigma_mult", 0.005), 0.005),
            spread_adv_mult=_safe_float(lm.get("spread_adv_mult", 15.0), 15.0),
            adv_ref_usd=_safe_float(lm.get("adv_ref_usd", 1e6), 1e6),
            depth_frac_of_adv_shares=_safe_float(
                lm.get("depth_frac_of_adv_shares", 0.001), 0.001
            ),
            depth_gamma=_safe_float(lm.get("depth_gamma", 1.0), 1.0),
            min_depth_shares=_safe_int(lm.get("min_depth_shares", 25), 25),
            max_depth_shares=_safe_int(lm.get("max_depth_shares", 250_000), 250_000),
            min_level_shares=_safe_int(lm.get("min_level_shares", 1), 1),
            lam_adv_power=_safe_float(lm.get("lam_adv_power", 0.25), 0.25),
            lam_min=_safe_float(lm.get("lam_min", 0.25), 0.25),
            lam_max=_safe_float(lm.get("lam_max", 25.0), 25.0),
            cancel_base=_safe_float(lm.get("cancel_base", 0.15), 0.15),
            cancel_sigma_mult=_safe_float(lm.get("cancel_sigma_mult", 1.5), 1.5),
            cancel_min=_safe_float(lm.get("cancel_min", 0.01), 0.01),
            cancel_max=_safe_float(lm.get("cancel_max", 0.90), 0.90),
            max_add_frac_of_top=_safe_float(lm.get("max_add_frac_of_top", 0.50), 0.50),
            max_cancel_frac_of_top=_safe_float(
                lm.get("max_cancel_frac_of_top", 0.25), 0.25
            ),
            max_add_min=_safe_int(lm.get("max_add_min", 50), 50),
            max_cancel_min=_safe_int(lm.get("max_cancel_min", 25), 25),
            max_add_max=_safe_int(lm.get("max_add_max", 50_000), 50_000),
            max_cancel_max=_safe_int(lm.get("max_cancel_max", 50_000), 50_000),
            tick_subpenny=_safe_float(lm.get("tick_subpenny", 0.0001), 0.0001),
            tick_penny=_safe_float(lm.get("tick_penny", 0.01), 0.01),
            tick_switch_price=_safe_float(lm.get("tick_switch_price", 1.0), 1.0),
            stress_enabled=bool(sm.get("enabled", True)),
            stress_intensity=_safe_float(sm.get("intensity", 1.0), 1.0),
            stress_max_entry_delay_days=max(
                0, _safe_int(sm.get("max_entry_delay_days", 1), 1)
            ),
            stress_max_exit_grace_days=max(
                0, _safe_int(sm.get("max_exit_grace_days", 2), 2)
            ),
            stress_panic_cross_bps=max(
                0.0, _safe_float(sm.get("panic_cross_bps", 50.0), 50.0)
            ),
        )


class LiquidityModel:
    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        cfg: LiquidityModelCfg,
        adv_map_usd: Mapping[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.panel = panel
        self.adv_map_usd = dict(adv_map_usd or {})

        self._field_level: int | None = None
        if isinstance(panel.columns, pd.MultiIndex) and panel.columns.nlevels >= 2:
            self._field_level = _infer_panel_field_level(panel.columns)

        self._cache_close: dict[str, pd.Series] = {}
        self._cache_open: dict[str, pd.Series] = {}
        self._cache_high: dict[str, pd.Series] = {}
        self._cache_low: dict[str, pd.Series] = {}
        self._cache_vol: dict[str, pd.Series] = {}
        self._cache_sigma: dict[str, pd.Series] = {}
        self._cache_sigma_med: dict[str, pd.Series] = {}
        self._cache_vol_med: dict[str, pd.Series] = {}
        self._cache_adv_usd: dict[str, pd.Series] = {}

    def _series(self, symbol: str, field: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        fld = str(field).strip().lower()
        if self.panel is None or self.panel.empty:
            return None
        if not isinstance(self.panel.index, pd.DatetimeIndex):
            return None

        if (
            isinstance(self.panel.columns, pd.MultiIndex)
            and self._field_level is not None
        ):
            try:
                s = self.panel.xs(
                    fld, axis=1, level=self._field_level, drop_level=True
                )[sym]
            except Exception:
                return None
            out = pd.to_numeric(s, errors="coerce").astype(float)
            out = out.loc[~out.index.duplicated()].sort_index()
            return out

        # Flat panel not supported for OHLCV inference.
        return None

    def close(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_close:
            return self._cache_close[sym]
        s = self._series(sym, "close")
        if s is not None:
            self._cache_close[sym] = s
        return s

    def open(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_open:
            return self._cache_open[sym]
        s = self._series(sym, "open")
        if s is not None:
            self._cache_open[sym] = s
        return s

    def high(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_high:
            return self._cache_high[sym]
        s = self._series(sym, "high")
        if s is not None:
            self._cache_high[sym] = s
        return s

    def low(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_low:
            return self._cache_low[sym]
        s = self._series(sym, "low")
        if s is not None:
            self._cache_low[sym] = s
        return s

    def volume(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_vol:
            return self._cache_vol[sym]
        s = self._series(sym, "volume")
        if s is not None:
            self._cache_vol[sym] = s
        return s

    def sigma(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_sigma:
            return self._cache_sigma[sym]
        c = self.close(sym)
        if c is None or c.empty:
            return None
        r = np.log(c).diff()
        w = max(2, int(self.cfg.vol_window))
        minp = max(2, int(np.ceil(float(w) * float(self.cfg.min_periods_frac))))
        s = r.rolling(w, min_periods=minp).std(ddof=0)
        s = s.replace([np.inf, -np.inf], np.nan).ffill()
        self._cache_sigma[sym] = s
        return s

    def sigma_median(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_sigma_med:
            return self._cache_sigma_med[sym]
        s = self.sigma(sym)
        if s is None or s.empty:
            return None
        w = max(5, int(self.cfg.vol_window))
        minp = max(5, int(np.ceil(float(w) * float(self.cfg.min_periods_frac))))
        med = s.rolling(w, min_periods=minp).median()
        med = med.replace([np.inf, -np.inf], np.nan).ffill()
        self._cache_sigma_med[sym] = med
        return med

    def adv_usd(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_adv_usd:
            return self._cache_adv_usd[sym]
        c = self.close(sym)
        v = self.volume(sym)
        if c is None or v is None or c.empty or v.empty:
            return None
        dv = (c * v).replace([np.inf, -np.inf], np.nan)
        w = max(5, int(self.cfg.adv_window))
        minp = max(5, int(np.ceil(float(w) * float(self.cfg.min_periods_frac))))
        a = dv.rolling(w, min_periods=minp).median()
        a = a.replace([np.inf, -np.inf], np.nan).ffill()
        self._cache_adv_usd[sym] = a
        return a

    def volume_median(self, symbol: str) -> pd.Series | None:
        sym = str(symbol).strip().upper()
        if sym in self._cache_vol_med:
            return self._cache_vol_med[sym]
        v = self.volume(sym)
        if v is None or v.empty:
            return None
        w = max(5, int(self.cfg.adv_window))
        minp = max(5, int(np.ceil(float(w) * float(self.cfg.min_periods_frac))))
        med = v.rolling(w, min_periods=minp).median()
        med = med.replace([np.inf, -np.inf], np.nan).ffill()
        self._cache_vol_med[sym] = med
        return med

    def _tick_for_price(self, px: float, base_tick: float) -> float:
        if not np.isfinite(px) or px <= 0:
            return float(base_tick)
        if float(px) < float(self.cfg.tick_switch_price):
            return float(self.cfg.tick_subpenny)
        return (
            float(self.cfg.tick_penny)
            if np.isfinite(self.cfg.tick_penny)
            else float(base_tick)
        )

    @staticmethod
    def _asof_shift_value(
        series: pd.Series | None, ts: pd.Timestamp, *, shift: int
    ) -> float | None:
        if series is None or series.empty:
            return None
        if not isinstance(series.index, pd.DatetimeIndex):
            return None
        s = series
        idx = cast(pd.DatetimeIndex, s.index)
        t = pd.Timestamp(ts)
        try:
            t = align_ts_to_index(t, idx)
        except Exception:
            t = pd.Timestamp(t)
        pos = int(idx.searchsorted(t, side="right") - 1 - int(max(0, shift)))
        if pos < 0:
            return None
        try:
            v = float(s.iloc[pos])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _spread_ticks(
        self, *, px: float, sigma: float, adv_usd: float, tick: float
    ) -> int:
        adv_scale = max(1e-12, float(adv_usd) / float(self.cfg.adv_ref_usd))
        spread_bps = (
            float(self.cfg.spread_floor_bps)
            + float(self.cfg.spread_sigma_mult) * float(sigma) * 1e4
            + float(self.cfg.spread_adv_mult) * (1.0 / np.sqrt(adv_scale))
        )
        spread_px = float(spread_bps) * float(px) / 1e4
        ticks = int(max(1, int(round(spread_px / float(tick)))))
        return ticks

    def _level_sizes(self, *, depth_top: int, levels: int) -> list[int]:
        g = float(max(0.0, self.cfg.depth_gamma))
        out: list[int] = []
        for i in range(1, int(levels) + 1):
            denom = float(i) ** g if g > 0 else 1.0
            sz = int(round(float(depth_top) / denom))
            out.append(int(max(self.cfg.min_level_shares, sz)))
        return out

    @staticmethod
    def _stress_regime(score: float) -> str:
        if score < 1.0:
            return "normal"
        if score < 1.75:
            return "elevated"
        if score < 2.5:
            return "stress"
        return "panic"

    @staticmethod
    def _stress_modifiers(regime: str) -> dict[str, float]:
        table = {
            "normal": {
                "spread_mult": 1.0,
                "depth_mult": 1.0,
                "cancel_add": 0.0,
                "maker_touch_mult": 1.0,
                "fill_mean_mult": 1.0,
                "aggr_prob": 0.00,
                "aggr_max_frac": 0.00,
            },
            "elevated": {
                "spread_mult": 1.5,
                "depth_mult": 0.75,
                "cancel_add": 0.10,
                "maker_touch_mult": 0.8,
                "fill_mean_mult": 0.9,
                "aggr_prob": 0.10,
                "aggr_max_frac": 0.25,
            },
            "stress": {
                "spread_mult": 2.5,
                "depth_mult": 0.50,
                "cancel_add": 0.20,
                "maker_touch_mult": 0.5,
                "fill_mean_mult": 0.7,
                "aggr_prob": 0.25,
                "aggr_max_frac": 0.50,
            },
            "panic": {
                "spread_mult": 4.0,
                "depth_mult": 0.25,
                "cancel_add": 0.35,
                "maker_touch_mult": 0.2,
                "fill_mean_mult": 0.4,
                "aggr_prob": 0.45,
                "aggr_max_frac": 1.00,
            },
        }
        return dict(table.get(str(regime).lower(), table["normal"]))

    def stress_state(self, symbol: str, ts: pd.Timestamp) -> dict[str, Any]:
        sym = str(symbol).strip().upper()
        t = _to_ny_ts(pd.Timestamp(ts))
        if not self.cfg.stress_enabled:
            mods = self._stress_modifiers("normal")
            return {
                "stress_score": 0.0,
                "stress_regime": "normal",
                "gap_bps": 0.0,
                "range_bps": 0.0,
                "volume_rel": 1.0,
                "sigma_rel": 1.0,
                **mods,
            }

        open_s = self.open(sym)
        high_s = self.high(sym)
        low_s = self.low(sym)
        close_s = self.close(sym)
        vol_s = self.volume(sym)
        sig_s = self.sigma(sym)
        sig_med_s = self.sigma_median(sym)
        vol_med_s = self.volume_median(sym)

        o = price_at_or_prior(open_s, t)
        h = price_at_or_prior(high_s, t)
        low_px = price_at_or_prior(low_s, t)
        c = price_at_or_prior(close_s, t)
        v = price_at_or_prior(vol_s, t, allow_zero=True)
        prev_c = self._asof_shift_value(close_s, t, shift=1)

        gap = 0.0
        if prev_c is not None and o is not None and np.isfinite(prev_c) and prev_c > 0:
            gap = abs(float(o) / float(prev_c) - 1.0)

        range_rel = 0.0
        if (
            h is not None
            and low_px is not None
            and c is not None
            and np.isfinite(c)
            and float(c) > 0.0
        ):
            range_rel = abs(float(h) - float(low_px)) / float(c)

        shift = int(max(0, self.cfg.asof_shift))
        sigma_val = self._asof_shift_value(sig_s, t, shift=shift)
        sigma_med_val = self._asof_shift_value(sig_med_s, t, shift=shift)
        sigma_rel = 1.0
        if (
            sigma_val is not None
            and sigma_med_val is not None
            and np.isfinite(sigma_val)
            and np.isfinite(sigma_med_val)
            and float(sigma_med_val) > 0.0
        ):
            sigma_rel = float(sigma_val) / float(sigma_med_val)

        vol_med_val = self._asof_shift_value(vol_med_s, t, shift=1)
        volume_rel = 1.0
        if (
            v is not None
            and vol_med_val is not None
            and np.isfinite(v)
            and np.isfinite(vol_med_val)
            and float(vol_med_val) > 0.0
        ):
            volume_rel = float(v) / float(vol_med_val)

        gap_term = _clip(float(gap) / 0.02, 0.0, 3.0)
        range_term = _clip(float(range_rel) / 0.03, 0.0, 3.0)
        sigma_term = _clip(float(sigma_rel) / 2.0, 0.0, 3.0)
        volume_term = _clip(float(volume_rel) / 3.0, 0.0, 3.0)
        score = float(self.cfg.stress_intensity) * (
            0.35 * gap_term + 0.35 * range_term + 0.20 * sigma_term + 0.10 * volume_term
        )
        regime = self._stress_regime(score)
        mods = self._stress_modifiers(regime)
        return {
            "stress_score": float(score),
            "stress_regime": str(regime),
            "gap_bps": float(gap) * 1e4,
            "range_bps": float(range_rel) * 1e4,
            "volume_rel": float(volume_rel),
            "sigma_rel": float(sigma_rel),
            "gap_term": float(gap_term),
            "range_term": float(range_term),
            "sigma_term": float(sigma_term),
            "volume_term": float(volume_term),
            **mods,
        }

    def book_params(
        self,
        symbol: str,
        ts: pd.Timestamp,
        *,
        base: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Return a per-(symbol,date) book parameter dict compatible with `OrderBook` and step().

        Uses only information at-or-prior to `ts` (rolling windows are past-only).
        """
        sym = str(symbol).strip().upper()
        t = _to_ny_ts(pd.Timestamp(ts))

        levels = _safe_int(base.get("levels", 5), 5)
        steps_per_day = _safe_int(base.get("steps_per_day", 4), 4)
        base_tick = _safe_float(base.get("tick", 0.01), 0.01)

        close_s = self.close(sym)
        shift = int(max(0, self.cfg.asof_shift))
        px_val = (
            self._asof_shift_value(close_s, t, shift=shift)
            if shift
            else price_at_or_prior(close_s, t)
        )
        px = _safe_float(px_val, default=float("nan"))
        if not (np.isfinite(px) and px > 0):
            px = _safe_float(base.get("mid_price", 100.0), 100.0)

        tick = self._tick_for_price(px, base_tick)

        sig_s = self.sigma(sym)
        sigma_val = (
            self._asof_shift_value(sig_s, t, shift=shift)
            if shift
            else price_at_or_prior(sig_s, t)
        )
        sigma = _safe_float(sigma_val, default=float("nan"))
        if not (np.isfinite(sigma) and sigma > 0):
            sigma = 0.02

        adv_s = self.adv_usd(sym)
        adv_val = (
            self._asof_shift_value(adv_s, t, shift=shift)
            if shift
            else price_at_or_prior(adv_s, t)
        )
        adv_usd = _safe_float(adv_val, default=float("nan"))
        if not (np.isfinite(adv_usd) and adv_usd > 0):
            adv_usd = _safe_float(self.adv_map_usd.get(sym), default=float("nan"))
        if not (np.isfinite(adv_usd) and adv_usd > 0):
            # last fallback: estimate from price * current volume
            vol_s = self.volume(sym)
            vol_val = (
                self._asof_shift_value(vol_s, t, shift=shift)
                if shift
                else price_at_or_prior(vol_s, t)
            )
            vv = _safe_float(vol_val, default=float("nan"))
            adv_usd = (
                float(px * vv)
                if (np.isfinite(vv) and vv > 0)
                else float(self.cfg.adv_ref_usd)
            )

        adv_shares = float(adv_usd) / float(px) if (np.isfinite(px) and px > 0) else 0.0
        adv_shares = float(max(0.0, adv_shares))

        spread_ticks = self._spread_ticks(
            px=px, sigma=sigma, adv_usd=adv_usd, tick=tick
        )

        depth_top = int(
            round(float(self.cfg.depth_frac_of_adv_shares) * float(adv_shares))
        )
        depth_top = int(
            max(self.cfg.min_depth_shares, min(self.cfg.max_depth_shares, depth_top))
        )
        stress = self.stress_state(sym, t)
        spread_ticks = int(
            max(
                1,
                round(float(spread_ticks) * float(stress.get("spread_mult", 1.0))),
            )
        )
        depth_top = int(
            max(
                self.cfg.min_depth_shares,
                min(
                    self.cfg.max_depth_shares,
                    round(float(depth_top) * float(stress.get("depth_mult", 1.0))),
                ),
            )
        )
        level_sizes = self._level_sizes(depth_top=depth_top, levels=levels)

        # Dynamics
        base_lam = _safe_float(base.get("lam", 2.0), 2.0)
        adv_scale = max(1e-12, float(adv_usd) / float(self.cfg.adv_ref_usd))
        lam = float(base_lam) * float(adv_scale ** float(self.cfg.lam_adv_power))
        lam = _clip(lam, self.cfg.lam_min, self.cfg.lam_max)

        cancel_base = _safe_float(
            base.get("cancel_prob", self.cfg.cancel_base), self.cfg.cancel_base
        )
        cancel = float(cancel_base) + float(self.cfg.cancel_sigma_mult) * float(sigma)
        cancel += float(stress.get("cancel_add", 0.0))
        cancel = _clip(cancel, self.cfg.cancel_min, self.cfg.cancel_max)

        # max_add/max_cancel: scale with top depth but keep sane defaults
        max_add = int(round(float(self.cfg.max_add_frac_of_top) * float(depth_top)))
        max_cancel = int(
            round(float(self.cfg.max_cancel_frac_of_top) * float(depth_top))
        )
        max_add = int(max(self.cfg.max_add_min, min(self.cfg.max_add_max, max_add)))
        max_cancel = int(
            max(self.cfg.max_cancel_min, min(self.cfg.max_cancel_max, max_cancel))
        )

        out = dict(base)
        out.update(
            {
                "tick": float(tick),
                "levels": int(levels),
                "min_spread_ticks": int(spread_ticks),
                "steps_per_day": int(max(1, steps_per_day)),
                "lam": float(lam),
                "cancel_prob": float(cancel),
                "max_add": int(max_add),
                "max_cancel": int(max_cancel),
                "level_sizes": list(level_sizes),
                # diagnostics
                "_liq_px": float(px),
                "_liq_sigma": float(sigma),
                "_liq_adv_usd": float(adv_usd),
                "_liq_depth_top": int(depth_top),
                "_liq_spread_ticks": int(spread_ticks),
                "_stress_score": float(stress.get("stress_score", 0.0)),
                "_stress_regime": str(stress.get("stress_regime", "normal")),
                "_stress_gap_bps": float(stress.get("gap_bps", 0.0)),
                "_stress_range_bps": float(stress.get("range_bps", 0.0)),
                "_stress_volume_rel": float(stress.get("volume_rel", 1.0)),
                "_stress_sigma_rel": float(stress.get("sigma_rel", 1.0)),
                "_stress_maker_touch_mult": float(stress.get("maker_touch_mult", 1.0)),
                "_stress_fill_mean_mult": float(stress.get("fill_mean_mult", 1.0)),
                "_stress_aggr_prob": float(stress.get("aggr_prob", 0.0)),
                "_stress_aggr_max_frac": float(stress.get("aggr_max_frac", 0.0)),
            }
        )
        return out
