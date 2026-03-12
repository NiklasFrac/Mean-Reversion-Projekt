"""
LOB execution annotator.

The LOB mode is execution-authoritative:
- gross_pnl always follows executed prices and executed dates
- slippage/impact stay as diagnostics
- liquidity stress is modeled via delays / blocked entries / forced exits
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.loader import select_field_from_panel
from backtest.simulators.common import price_at_or_prior
from backtest.simulators.costs import compute_post_lob_costs
from backtest.simulators.fill_model import FillModelCfg, sample_package_fill_fraction
from backtest.simulators.liquidity_model import LiquidityModel, LiquidityModelCfg
from backtest.simulators.orderbook_sim import OrderBook

logger = logging.getLogger("backtest.simulators.lob")
logger.addHandler(logging.NullHandler())

__all__ = ["annotate_with_lob"]


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


def _get(mapping: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = mapping or {}
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_exec_lob_cfg(cfg_obj: Any) -> dict[str, Any]:
    direct: Any = None
    try:
        direct = getattr(cfg_obj, "exec_lob", None)
        if isinstance(direct, Mapping) and direct:
            return dict(direct)
    except Exception:
        pass
    raw_yaml = getattr(cfg_obj, "raw_yaml", {}) or {}
    if isinstance(raw_yaml, Mapping):
        ex = raw_yaml.get("execution", None)
        if isinstance(ex, Mapping) and isinstance(ex.get("lob"), Mapping):
            return dict(cast(Mapping[str, Any], ex.get("lob")))
    if isinstance(direct, Mapping):
        return dict(direct)
    return {}


def _normalize_symbol(x: Any) -> str:
    return str(x or "").strip().upper()


def _trade_units_from_row(row: pd.Series) -> tuple[int, int]:
    sig = _safe_int(row.get("signal", row.get("position", 0)), 0)
    size = _safe_int(row.get("size", row.get("units", 0)), 0)
    if sig == 0 or size <= 0:
        return 0, 0
    beta = _safe_float(row.get("beta_entry", row.get("beta", 1.0)), 1.0)
    beta_eff = float(beta) if np.isfinite(beta) and float(beta) != 0.0 else 1.0
    beta_abs = float(abs(beta_eff))
    x_units_abs = max(1, int(round(float(size) * beta_abs)))
    y_units = int(size) * (1 if sig > 0 else -1)
    x_sign = int(np.sign(-float(sig) * beta_eff))
    x_units = int(x_sign if x_sign != 0 else -int(sig)) * int(x_units_abs)
    return y_units, x_units


def _mk_book_params(cfg_obj: Any) -> dict[str, Any]:
    lob_cfg = _resolve_exec_lob_cfg(cfg_obj)
    raw = getattr(cfg_obj, "raw_yaml", {}) or {}
    seed = _safe_int(_get(raw, "seed", default=None), default=-1)
    seed_val = None if seed < 0 else int(seed)
    return {
        "tick": _safe_float(lob_cfg.get("tick", 0.01), 0.01),
        "levels": _safe_int(lob_cfg.get("levels", 5), 5),
        "size_per_level": _safe_int(lob_cfg.get("size_per_level", 1_000), 1_000),
        "min_spread_ticks": _safe_int(lob_cfg.get("min_spread_ticks", 1), 1),
        "steps_per_day": _safe_int(lob_cfg.get("steps_per_day", 4), 4),
        "lam": _safe_float(lob_cfg.get("lam", 2.0), 2.0),
        "max_add": _safe_int(lob_cfg.get("max_add", 500), 500),
        "bias_top": _safe_float(lob_cfg.get("bias_top", 0.7), 0.7),
        "cancel_prob": _safe_float(lob_cfg.get("cancel_prob", 0.15), 0.15),
        "max_cancel": _safe_int(lob_cfg.get("max_cancel", 200), 200),
        "seed": seed_val,
    }


def _mid_at(price_series: pd.Series | None, ts: pd.Timestamp) -> float | None:
    px = price_at_or_prior(price_series, ts)
    if px is None:
        return None
    return float(px) if np.isfinite(px) and px > 0 else None


def _seeded_rng(seed: int | None, *parts: int) -> np.random.Generator:
    if seed is None or seed < 0:
        return np.random.default_rng()
    seq = [int(seed), 7919]
    seq.extend(int(p) for p in parts)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seq)))


def _coerce_ts_to_index(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    tz = getattr(idx, "tz", None)
    if tz is not None:
        if out.tzinfo is None:
            out = out.tz_localize(tz)
        else:
            out = out.tz_convert(tz)
    elif out.tzinfo is not None:
        out = out.tz_localize(None)
    return out


def _build_session_index(
    *,
    calendar: pd.DatetimeIndex | None,
    y_series: pd.Series | None,
    x_series: pd.Series | None,
) -> pd.DatetimeIndex:
    if (
        calendar is not None
        and isinstance(calendar, pd.DatetimeIndex)
        and len(calendar)
    ):
        idx = pd.DatetimeIndex(calendar).sort_values()
        return idx[~idx.duplicated()]

    parts: list[pd.DatetimeIndex] = []
    for series in (y_series, x_series):
        if isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(series.index).sort_values()
            parts.append(idx[~idx.duplicated()])
    if not parts:
        return pd.DatetimeIndex([])
    out = parts[0]
    for part in parts[1:]:
        out = out.union(part)
    out = out.sort_values()
    return out[~out.duplicated()]


def _candidate_sessions(
    idx: pd.DatetimeIndex,
    *,
    start_ts: pd.Timestamp,
    max_delay_days: int,
    cap_ts: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    if idx is None or len(idx) == 0:
        return []
    norm_idx = idx.normalize()
    start = _coerce_ts_to_index(pd.Timestamp(start_ts), idx).normalize()
    pos0 = int(norm_idx.searchsorted(start, side="left"))
    if pos0 >= len(idx):
        return []
    pos1 = min(len(idx), pos0 + int(max(0, max_delay_days)) + 1)
    out = list(idx[pos0:pos1])
    if cap_ts is None:
        return out
    cap = _coerce_ts_to_index(pd.Timestamp(cap_ts), idx).normalize()
    return [ts for ts in out if pd.Timestamp(ts).normalize() <= cap]


def _make_book(mid_price: float, params: Mapping[str, Any], shard_id: int) -> OrderBook:
    return OrderBook(
        mid_price=float(mid_price),
        levels=int(params.get("levels", 5)),
        size_per_level=int(params.get("size_per_level", 1_000)),
        tick=float(params.get("tick", 0.01)),
        seed=params.get("seed"),
        min_spread_ticks=int(params.get("min_spread_ticks", 1)),
        level_sizes=list(params.get("level_sizes") or []) or None,
        shard_id=int(shard_id),
    )


def _apply_book_state(
    ob: OrderBook,
    *,
    mid_price: float,
    params: Mapping[str, Any],
    base_params: Mapping[str, Any],
    step_day: bool,
) -> None:
    try:
        ob.set_public_level_sizes(list(params.get("level_sizes") or []))
    except Exception:
        pass
    try:
        ob.set_min_spread_ticks(
            int(params.get("min_spread_ticks", base_params.get("min_spread_ticks", 1)))
        )
    except Exception:
        pass
    ob.recenter(float(mid_price), preserve_spread=True)
    if not step_day:
        return
    n_steps = int(
        max(1, params.get("steps_per_day", base_params.get("steps_per_day", 1)))
    )
    for _ in range(n_steps):
        top_bid = int(ob.bids[0].total_size) if ob.bids else 0
        top_ask = int(ob.asks[0].total_size) if ob.asks else 0
        top_now = max(
            1,
            top_bid,
            top_ask,
            int(params.get("_liq_depth_top", 1) or 1),
        )
        aggr_prob = float(params.get("_stress_aggr_prob", 0.0) or 0.0)
        aggr_frac = float(params.get("_stress_aggr_max_frac", 0.0) or 0.0)
        aggr_max = 0
        if aggr_prob > 0.0 and aggr_frac > 0.0:
            aggr_max = max(1, int(round(float(top_now) * float(aggr_frac))))
        ob.step(
            lam=float(params.get("lam", base_params.get("lam", 2.0))),
            max_add=int(params.get("max_add", base_params.get("max_add", 500))),
            bias_top=float(params.get("bias_top", base_params.get("bias_top", 0.7))),
            cancel_prob=float(
                params.get("cancel_prob", base_params.get("cancel_prob", 0.15))
            ),
            max_cancel=int(
                params.get("max_cancel", base_params.get("max_cancel", 200))
            ),
            aggr_prob=aggr_prob,
            aggr_max=int(aggr_max),
        )


def _daily_high_low(
    liq_model: LiquidityModel | None, symbol: str, ts: pd.Timestamp
) -> tuple[float | None, float | None]:
    if liq_model is None:
        return None, None
    hi = price_at_or_prior(liq_model.high(symbol), ts)
    lo = price_at_or_prior(liq_model.low(symbol), ts)
    hi_f = float(hi) if hi is not None and np.isfinite(hi) else None
    lo_f = float(lo) if lo is not None and np.isfinite(lo) else None
    return hi_f, lo_f


def _day_bar(
    liq_model: LiquidityModel | None,
    *,
    symbol: str,
    ts: pd.Timestamp,
    price_series: pd.Series | None,
) -> dict[str, float | None]:
    close_px = _mid_at(price_series, ts)
    open_px = None
    high_px = None
    low_px = None
    volume = None
    if liq_model is not None:
        op = price_at_or_prior(liq_model.open(symbol), ts)
        hi = price_at_or_prior(liq_model.high(symbol), ts)
        lo = price_at_or_prior(liq_model.low(symbol), ts)
        vol = price_at_or_prior(liq_model.volume(symbol), ts, allow_zero=True)
        open_px = float(op) if op is not None and np.isfinite(op) else None
        high_px = float(hi) if hi is not None and np.isfinite(hi) else None
        low_px = float(lo) if lo is not None and np.isfinite(lo) else None
        volume = float(vol) if vol is not None and np.isfinite(vol) else None
    return {
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
    }


def _flow_mode(flow_cfg: Mapping[str, Any] | None) -> str:
    mode = str((flow_cfg or {}).get("mode", "taker")).strip().lower()
    return mode if mode in {"taker", "maker", "mixed"} else "taker"


def _maker_price(
    ob: OrderBook, flow_cfg: Mapping[str, Any] | None, *, side: str, mid: float | None
) -> float | None:
    pref = str((flow_cfg or {}).get("maker_price", "best")).strip().lower()
    if pref == "mid":
        px = float(mid) if mid is not None and np.isfinite(mid) and mid > 0 else None
    else:
        px = ob.best_bid() if side == "buy" else ob.best_ask()
    if px is None and mid is not None and np.isfinite(mid) and mid > 0:
        px = float(mid)
    return float(px) if px is not None and np.isfinite(px) and px > 0 else None


def _same_side_top_depth(ob: OrderBook, side: str) -> int:
    try:
        if str(side).lower() == "buy" and ob.bids:
            return int(max(0, int(ob.bids[0].total_size)))
        if str(side).lower() == "sell" and ob.asks:
            return int(max(0, int(ob.asks[0].total_size)))
    except Exception:
        pass
    return 0


def _normalize_fills(fills: Any) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    try:
        items = list(fills or [])
    except Exception:
        items = []
    for item in items:
        try:
            px_raw, qty_raw = item
        except Exception:
            continue
        px = _safe_float(px_raw, default=float("nan"))
        qty = _safe_int(qty_raw, default=0)
        if np.isfinite(px) and px > 0.0 and qty > 0:
            out.append((float(px), int(qty)))
    return out


def _fills_vwap(fills: Any) -> tuple[int, float | None]:
    norm = _normalize_fills(fills)
    if not norm:
        return 0, None
    filled = int(sum(int(qty) for _, qty in norm))
    if filled <= 0:
        return 0, None
    notional = float(sum(float(px) * float(qty) for px, qty in norm))
    return int(filled), float(notional / float(filled))


def _maker_turnover_target(
    ob: OrderBook,
    *,
    qty: int,
    top_depth: int,
    maker_max_top_frac: float,
    steps_per_day: int,
    aggr_prob: float,
    aggr_max_frac: float,
) -> int:
    req = int(max(0, qty))
    if req <= 0:
        return 0
    top_now = int(max(1, top_depth))
    maker_base = float(top_now) * float(max(0.05, maker_max_top_frac))
    maker_stress = (
        float(top_now)
        * float(max(0.0, aggr_prob))
        * float(max(0.0, aggr_max_frac))
        * float(max(1, steps_per_day))
    )
    turnover_cap = float(max(1.0, maker_base + maker_stress))
    coverage = float(min(1.0, turnover_cap / float(max(1, req))))
    fill_mean = float(max(0.05, min(0.90, 0.25 + 0.55 * coverage)))
    fill_frac = fill_mean
    if getattr(ob, "_global_seed", None) is not None:
        kappa = float(8.0 + 10.0 * coverage)
        mu_c = float(max(1e-6, min(1.0 - 1e-6, fill_mean)))
        try:
            fill_frac = float(ob.rng.beta(mu_c * kappa, (1.0 - mu_c) * kappa))
        except Exception:
            fill_frac = fill_mean
    target = int(round(float(turnover_cap) * float(max(0.0, min(1.0, fill_frac)))))
    return int(max(0, min(req, target)))


def _pick_vwap(rep: Mapping[str, Any]) -> float | None:
    for key in ("vwap", "avg_price"):
        val = rep.get(key, None)
        if val is None:
            continue
        out = _safe_float(val, default=float("nan"))
        if np.isfinite(out) and out > 0:
            return float(out)
    return None


def _exec_order(
    ob: OrderBook,
    *,
    symbol: str,
    side: str,
    qty: int,
    mid: float | None,
    ts: pd.Timestamp,
    liq_model: LiquidityModel | None,
    flow_cfg: Mapping[str, Any] | None,
    is_short: bool,
    maker_touch_mult: float = 1.0,
    maker_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if qty <= 0:
        return {
            "filled_size": 0,
            "avg_price": None,
            "vwap": None,
            "signed_slippage": 0.0,
            "slippage_ticks": 0.0,
            "unfilled_size": 0,
            "fills": [],
            "pre_best_bid": None,
            "pre_best_ask": None,
            "post_best_bid": None,
            "post_best_ask": None,
            "role": "taker",
        }

    cfg = dict(flow_cfg or {})
    mode = _flow_mode(cfg)
    fallback_to_taker = bool(cfg.get("fallback_to_taker", True))
    maker_prob = float(cfg.get("maker_prob", 0.3))
    maker_max_top_frac = float(cfg.get("maker_max_top_frac", 0.25))
    maker_touch_prob = float(cfg.get("maker_touch_prob", 1.0))
    maker_touch_prob = float(max(0.0, min(1.0, maker_touch_prob * maker_touch_mult)))

    use_maker = False
    top_depth = 0
    if mode == "maker":
        use_maker = True
    elif mode == "mixed":
        try:
            use_maker = bool(ob.rng.random() < maker_prob)
        except Exception:
            use_maker = False

    limit_px = _maker_price(ob, cfg, side=side, mid=mid) if use_maker else None
    if use_maker and limit_px is not None:
        try:
            top_depth = _same_side_top_depth(ob, side)
            if maker_max_top_frac > 0:
                top = int(max(0, top_depth))
                if top > 0 and qty > int(
                    max(1, round(float(top) * maker_max_top_frac))
                ):
                    use_maker = False
        except Exception:
            pass

    if use_maker and limit_px is not None:
        hi, lo = _daily_high_low(liq_model, symbol, ts)
        touched = True
        if side == "buy" and lo is not None:
            touched = float(limit_px) >= float(lo)
        if side == "sell" and hi is not None:
            touched = float(limit_px) <= float(hi)
        if not touched:
            use_maker = False
        elif maker_touch_prob < 1.0:
            try:
                use_maker = bool(ob.rng.random() < maker_touch_prob)
            except Exception:
                use_maker = False

    if use_maker and limit_px is not None:
        pre_bb = ob.best_bid()
        pre_ba = ob.best_ask()
        state = dict(maker_state or {})
        steps_per_day = max(1, _safe_int(state.get("steps_per_day", 1), 1))
        oid = 0
        try:
            post_rep = ob.process_limit_order(
                side=side,
                price=float(limit_px),
                size=int(qty),
                tif="gtc",
                post_only=True,
                po_action="slide",
                owner="lob-maker",
                symbol=symbol,
                is_short=bool(is_short),
            )
            oid = _safe_int(post_rep.get("resting_oid"), 0)
        except Exception:
            oid = 0

        if oid > 0:
            for _ in range(max(0, steps_per_day - 1)):
                ob.step(
                    lam=float(state.get("lam", 0.0) or 0.0),
                    max_add=int(max(2, _safe_int(state.get("max_add", 0), 0))),
                    bias_top=float(state.get("bias_top", 0.7) or 0.7),
                    cancel_prob=float(state.get("cancel_prob", 0.0) or 0.0),
                    max_cancel=int(max(1, _safe_int(state.get("max_cancel", 1), 1))),
                    aggr_prob=0.0,
                    aggr_max=0,
                )

            pos = ob.order_position(oid) or {}
            qty_ahead = max(0, _safe_int(pos.get("qty_ahead", 0), 0))
            top_now = int(
                max(
                    1,
                    top_depth,
                    _safe_int(state.get("_liq_depth_top", 1), 1),
                )
            )
            maker_target = _maker_turnover_target(
                ob,
                qty=int(qty),
                top_depth=int(top_now),
                maker_max_top_frac=float(maker_max_top_frac),
                steps_per_day=int(steps_per_day),
                aggr_prob=float(
                    max(
                        0.0,
                        min(1.0, _safe_float(state.get("_stress_aggr_prob", 0.0), 0.0)),
                    )
                ),
                aggr_max_frac=float(
                    max(
                        0.0,
                        _safe_float(state.get("_stress_aggr_max_frac", 0.0), 0.0),
                    )
                ),
            )
            if maker_target > 0:
                contra_side = "sell" if side == "buy" else "buy"
                try:
                    ob.process_market_order(
                        side=contra_side,
                        size=int(max(1, qty_ahead + maker_target)),
                        tif="ioc",
                        symbol=symbol,
                        is_short=False,
                    )
                except Exception:
                    pass

        maker_fills = _normalize_fills(ob.collect_fills_for_oid(oid) if oid > 0 else [])
        maker_filled, maker_avg = _fills_vwap(maker_fills)
        _ = ob.order_position(oid) if oid > 0 else None
        residual = int(max(0, qty - maker_filled))
        if oid > 0 and residual > 0:
            try:
                ob.cancel_order(oid)
            except Exception:
                pass

        if residual <= 0 and maker_filled > 0 and maker_avg is not None:
            post_bb = ob.best_bid()
            post_ba = ob.best_ask()
            return {
                "filled_size": int(maker_filled),
                "avg_price": float(maker_avg),
                "vwap": float(maker_avg),
                "signed_slippage": 0.0,
                "slippage_ticks": 0.0,
                "unfilled_size": 0,
                "fills": maker_fills,
                "pre_best_bid": pre_bb,
                "pre_best_ask": pre_ba,
                "post_best_bid": post_bb,
                "post_best_ask": post_ba,
                "role": "maker",
                "maker_filled_size": int(maker_filled),
                "taker_filled_size": 0,
                "spread_cost_cash": 0.0,
                "impact_cost_cash": 0.0,
            }

        if residual > 0 and not fallback_to_taker:
            post_bb = ob.best_bid()
            post_ba = ob.best_ask()
            return {
                "filled_size": int(maker_filled),
                "avg_price": float(maker_avg) if maker_avg is not None else None,
                "vwap": float(maker_avg) if maker_avg is not None else None,
                "signed_slippage": 0.0,
                "slippage_ticks": 0.0,
                "unfilled_size": int(residual),
                "fills": maker_fills,
                "pre_best_bid": pre_bb,
                "pre_best_ask": pre_ba,
                "post_best_bid": post_bb,
                "post_best_ask": post_ba,
                "role": "maker" if maker_filled > 0 else "blocked",
                "maker_filled_size": int(maker_filled),
                "taker_filled_size": 0,
                "spread_cost_cash": 0.0,
                "impact_cost_cash": 0.0,
            }

        if residual > 0:
            taker_rep = ob.process_market_order(
                side=side,
                size=int(residual),
                tif="ioc",
                symbol=symbol,
                is_short=bool(is_short),
            )
            taker_fills = _normalize_fills(taker_rep.get("fills", []))
            taker_filled, _ = _fills_vwap(taker_fills)
            filled_total = int(maker_filled + taker_filled)
            fills_total = maker_fills + taker_fills
            _, avg_total = _fills_vwap(fills_total)
            taker_slip, taker_imp = _slip_decomp(taker_rep, side, ref_mid=mid)
            signed_slippage = float(
                _safe_float(taker_rep.get("signed_slippage", 0.0), 0.0)
            )
            signed_slippage = (
                float(signed_slippage) * float(taker_filled) / float(filled_total)
                if filled_total > 0 and taker_filled > 0
                else 0.0
            )
            return {
                "filled_size": int(filled_total),
                "avg_price": float(avg_total) if avg_total is not None else None,
                "vwap": float(avg_total) if avg_total is not None else None,
                "signed_slippage": float(signed_slippage),
                "slippage_ticks": float(signed_slippage / ob.tick)
                if ob.tick > 0
                else 0.0,
                "unfilled_size": int(max(0, qty - filled_total)),
                "fills": fills_total,
                "pre_best_bid": pre_bb,
                "pre_best_ask": pre_ba,
                "post_best_bid": taker_rep.get("post_best_bid", ob.best_bid()),
                "post_best_ask": taker_rep.get("post_best_ask", ob.best_ask()),
                "role": "taker",
                "role_detail": "maker_partial_taker"
                if maker_filled > 0
                else "maker_fallback_taker",
                "maker_filled_size": int(maker_filled),
                "taker_filled_size": int(taker_filled),
                "spread_cost_cash": float(taker_slip),
                "impact_cost_cash": float(taker_imp),
            }

    if mode == "maker" and not fallback_to_taker:
        return {
            "filled_size": 0,
            "avg_price": None,
            "vwap": None,
            "signed_slippage": 0.0,
            "slippage_ticks": 0.0,
            "unfilled_size": int(qty),
            "fills": [],
            "pre_best_bid": ob.best_bid(),
            "pre_best_ask": ob.best_ask(),
            "post_best_bid": ob.best_bid(),
            "post_best_ask": ob.best_ask(),
            "role": "blocked",
        }

    return ob.process_market_order(
        side=side,
        size=int(qty),
        tif="ioc",
        symbol=symbol,
        is_short=bool(is_short),
    )


def _slip_decomp(
    rep: Mapping[str, Any], side: str, *, ref_mid: float | None
) -> tuple[float, float]:
    spread_cash = _safe_float(rep.get("spread_cost_cash"), default=float("nan"))
    impact_cash = _safe_float(rep.get("impact_cost_cash"), default=float("nan"))
    if np.isfinite(spread_cash) and np.isfinite(impact_cash):
        return float(spread_cash), float(impact_cash)
    try:
        filled = int(rep.get("filled_size") or 0)
    except Exception:
        filled = 0
    if filled <= 0:
        return 0.0, 0.0
    try:
        role = str(rep.get("role", "") or "").lower()
        if "maker" in role:
            return 0.0, 0.0
    except Exception:
        pass
    avg = _safe_float(rep.get("avg_price", rep.get("vwap")), default=float("nan"))
    bid = _safe_float(rep.get("pre_best_bid"), default=float("nan"))
    ask = _safe_float(rep.get("pre_best_ask"), default=float("nan"))
    if not (np.isfinite(avg) and np.isfinite(bid) and np.isfinite(ask)):
        return 0.0, 0.0
    mid = (
        float(ref_mid)
        if ref_mid is not None and np.isfinite(ref_mid) and ref_mid > 0
        else 0.5 * (bid + ask)
    )
    if not (np.isfinite(mid) and mid > 0):
        return 0.0, 0.0
    if str(side).lower() == "buy":
        spread_px = max(0.0, float(ask - mid))
        impact_px = max(0.0, float(avg - ask))
    else:
        spread_px = max(0.0, float(mid - bid))
        impact_px = max(0.0, float(bid - avg))
    return -spread_px * float(filled), -impact_px * float(filled)


def _state_for_symbol(
    *,
    liq_model: LiquidityModel | None,
    symbol: str,
    ts: pd.Timestamp,
    base_book_params: Mapping[str, Any],
    mid_hint: float | None = None,
) -> dict[str, Any]:
    if liq_model is None:
        out = dict(base_book_params)
        if mid_hint is not None and np.isfinite(mid_hint) and mid_hint > 0:
            out["_liq_px"] = float(mid_hint)
        out.setdefault("_stress_regime", "normal")
        out.setdefault("_stress_score", 0.0)
        out.setdefault("_stress_gap_bps", 0.0)
        out.setdefault("_stress_range_bps", 0.0)
        out.setdefault("_stress_volume_rel", 1.0)
        out.setdefault("_stress_maker_touch_mult", 1.0)
        out.setdefault("_stress_fill_mean_mult", 1.0)
        out.setdefault("_stress_aggr_prob", 0.0)
        out.setdefault("_stress_aggr_max_frac", 0.0)
        out.setdefault("_liq_spread_ticks", base_book_params.get("min_spread_ticks", 1))
        out.setdefault("_liq_depth_top", base_book_params.get("size_per_level", 1_000))
        return out
    return liq_model.book_params(symbol, pd.Timestamp(ts), base=base_book_params)


def _pair_fill_fraction(
    *,
    y_sym: str,
    x_sym: str,
    ts: pd.Timestamp,
    y_qty_abs: int,
    x_qty_abs: int,
    y_ref_px: float,
    x_ref_px: float,
    liq_model: LiquidityModel | None,
    base_book_params: Mapping[str, Any],
    fill_cfg: FillModelCfg,
    seed: int | None,
    shard_id: int,
) -> tuple[float, float, dict[str, Any], dict[str, Any], str]:
    y_state = _state_for_symbol(
        liq_model=liq_model,
        symbol=y_sym,
        ts=ts,
        base_book_params=base_book_params,
        mid_hint=y_ref_px,
    )
    x_state = _state_for_symbol(
        liq_model=liq_model,
        symbol=x_sym,
        ts=ts,
        base_book_params=base_book_params,
        mid_hint=x_ref_px,
    )
    pair_state = (
        y_state
        if float(y_state.get("_stress_score", 0.0))
        >= float(x_state.get("_stress_score", 0.0))
        else x_state
    )
    pair_regime = str(pair_state.get("_stress_regime", "normal"))
    if not fill_cfg.enabled or liq_model is None:
        return 1.0, 1.0, y_state, x_state, pair_regime

    y_lv = list(y_state.get("level_sizes") or [])
    x_lv = list(x_state.get("level_sizes") or [])
    depth_y = float(sum(y_lv)) if y_lv else 0.0
    depth_x = float(sum(x_lv)) if x_lv else 0.0
    depth_pair = float(max(1.0, min(depth_y or 1.0, depth_x or 1.0)))
    qty_pair = float(max(abs(int(y_qty_abs)), abs(int(x_qty_abs))))

    adv_y = float(y_state.get("_liq_adv_usd", np.nan))
    adv_x = float(x_state.get("_liq_adv_usd", np.nan))
    adv_pair = (
        float(min(adv_y, adv_x))
        if np.isfinite(adv_y) and np.isfinite(adv_x)
        else float("nan")
    )
    if not (np.isfinite(adv_pair) and adv_pair > 0):
        adv_pair = float(
            max(
                adv_y if np.isfinite(adv_y) else 0.0,
                adv_x if np.isfinite(adv_x) else 0.0,
                1.0,
            )
        )

    gross_notional = float(
        abs(y_qty_abs) * max(y_ref_px, 0.0) + abs(x_qty_abs) * max(x_ref_px, 0.0)
    )
    participation_usd = float(gross_notional / adv_pair) if adv_pair > 0 else 0.0
    sigma_pair = float(
        max(
            float(y_state.get("_liq_sigma", 0.0) or 0.0),
            float(x_state.get("_liq_sigma", 0.0) or 0.0),
            0.0,
        )
    )
    sampled_fill, diag = sample_package_fill_fraction(
        cfg=fill_cfg,
        seed=seed,
        shard_id=int(shard_id),
        depth_total_shares_pair=depth_pair,
        qty_pair_shares=qty_pair,
        adv_usd_pair=adv_pair,
        adv_ref_usd=float(getattr(liq_model.cfg, "adv_ref_usd", 1e6)),
        participation_usd=participation_usd,
        sigma_pair=sigma_pair,
    )
    base_fill = float(sampled_fill)
    fill_mult = float(
        min(
            y_state.get("_stress_fill_mean_mult", 1.0),
            x_state.get("_stress_fill_mean_mult", 1.0),
        )
    )
    eff_fill = float(max(0.0, min(1.0, float(base_fill) * fill_mult)))
    eff_expected = float(
        max(0.0, min(1.0, float(diag.get("expected", 1.0)) * fill_mult))
    )
    return eff_fill, eff_expected, y_state, x_state, pair_regime


def _regime_rank(regime: str) -> int:
    table = {"normal": 0, "elevated": 1, "stress": 2, "panic": 3}
    return int(table.get(str(regime or "").strip().lower(), 0))


def _worse_regime(*regimes: str) -> str:
    best = "normal"
    best_rank = -1
    for regime in regimes:
        rank = _regime_rank(str(regime))
        if rank > best_rank:
            best = str(regime or "normal").strip().lower() or "normal"
            best_rank = rank
    return best


def _effective_exit_grace_days(max_days: int, regime: str) -> int:
    max_days_i = max(0, int(max_days))
    shrink = {
        "normal": 0,
        "elevated": 0,
        "stress": 1,
        "panic": max_days_i,
    }
    return int(max(0, max_days_i - int(shrink.get(str(regime).lower(), 0))))


def _panic_cross_bps_for_regime(base_bps: float, regime: str) -> float:
    mult = {
        "normal": 1.0,
        "elevated": 1.5,
        "stress": 2.5,
        "panic": 5.0,
    }
    return float(max(0.0, float(base_bps)) * float(mult.get(str(regime).lower(), 1.0)))


def _leg_package_capacity_shares(
    *,
    qty_abs: int,
    state: Mapping[str, Any],
    flow_cfg: Mapping[str, Any] | None,
    fill_frac: float,
    steps_per_day: int,
    exit_capacity_mult: float,
) -> float:
    if int(qty_abs) <= 0:
        return 0.0

    level_sizes_raw = list(state.get("level_sizes") or [])
    level_sizes: list[float] = []
    for val in level_sizes_raw:
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval) and fval > 0.0:
            level_sizes.append(float(fval))

    top_depth = (
        float(level_sizes[0])
        if level_sizes
        else max(1.0, float(state.get("_liq_depth_top", 1.0) or 1.0))
    )
    visible_depth = (
        float(sum(level_sizes))
        if level_sizes
        else max(top_depth, top_depth * float(max(1, steps_per_day)))
    )
    flow = dict(flow_cfg or {})
    mode = _flow_mode(flow)
    fallback_to_taker = bool(flow.get("fallback_to_taker", True))
    fill_cap = float(max(0.0, min(1.0, float(fill_frac))))
    exit_mult = float(max(0.0, float(exit_capacity_mult)))
    stress_cap_mult = {
        "normal": 1.0,
        "elevated": 0.9,
        "stress": 0.65,
        "panic": 0.35,
    }.get(str(state.get("_stress_regime", "normal")).lower(), 1.0)
    cap_mult = float(exit_mult) * float(stress_cap_mult)

    taker_cap = float(visible_depth) * (0.10 + 0.90 * fill_cap) * cap_mult
    taker_cap = max(float(top_depth) * 0.25 * max(1.0, cap_mult), taker_cap)
    adv_usd = _safe_float(state.get("_liq_adv_usd", np.nan), float("nan"))
    ref_px = _safe_float(state.get("_liq_px", np.nan), float("nan"))
    if np.isfinite(adv_usd) and adv_usd > 0.0 and np.isfinite(ref_px) and ref_px > 0.0:
        adv_shares = float(adv_usd) / float(ref_px)
        adv_participation = 0.01 + 0.015 * fill_cap
        adv_cap = float(adv_shares) * float(adv_participation) * cap_mult
        taker_cap = max(taker_cap, adv_cap)

    maker_touch = float(
        max(
            0.0,
            min(
                1.0,
                _safe_float(flow.get("maker_touch_prob", 1.0), 1.0)
                * _safe_float(state.get("_stress_maker_touch_mult", 1.0), 1.0),
            ),
        )
    )
    maker_top_frac = float(
        max(0.0, _safe_float(flow.get("maker_max_top_frac", 0.25), 0.25))
    )
    aggr_prob = float(
        max(0.0, min(1.0, _safe_float(state.get("_stress_aggr_prob", 0.0), 0.0)))
    )
    aggr_max_frac = float(
        max(0.0, _safe_float(state.get("_stress_aggr_max_frac", 0.0), 0.0))
    )
    maker_turnover = float(top_depth) * maker_top_frac + float(
        top_depth
    ) * aggr_prob * aggr_max_frac * float(max(1, steps_per_day))
    maker_cap = maker_touch * maker_turnover * max(0.25, fill_cap)

    if mode == "maker":
        cap = maker_cap if not fallback_to_taker else max(maker_cap, taker_cap)
    elif mode == "mixed":
        cap = max(maker_cap, taker_cap)
    else:
        cap = taker_cap
    return float(max(0.0, cap))


def _pair_package_capacity(
    *,
    y_qty_abs: int,
    x_qty_abs: int,
    y_state: Mapping[str, Any],
    x_state: Mapping[str, Any],
    flow_cfg: Mapping[str, Any] | None,
    fill_frac: float,
    base_book_params: Mapping[str, Any],
    exit_capacity_mult: float = 1.0,
) -> tuple[float, float, float]:
    steps_per_day = max(1, _safe_int(base_book_params.get("steps_per_day", 4), 4))
    cap_y = _leg_package_capacity_shares(
        qty_abs=int(y_qty_abs),
        state=y_state,
        flow_cfg=flow_cfg,
        fill_frac=fill_frac,
        steps_per_day=steps_per_day,
        exit_capacity_mult=exit_capacity_mult,
    )
    cap_x = _leg_package_capacity_shares(
        qty_abs=int(x_qty_abs),
        state=x_state,
        flow_cfg=flow_cfg,
        fill_frac=fill_frac,
        steps_per_day=steps_per_day,
        exit_capacity_mult=exit_capacity_mult,
    )
    frac_y = (
        float("inf")
        if int(y_qty_abs) <= 0
        else float(cap_y) / float(max(1, int(y_qty_abs)))
    )
    frac_x = (
        float("inf")
        if int(x_qty_abs) <= 0
        else float(cap_x) / float(max(1, int(x_qty_abs)))
    )
    pair_frac = float(min(frac_y, frac_x))
    if not np.isfinite(pair_frac):
        pair_frac = 1.0
    return float(max(0.0, min(1.0, pair_frac))), float(cap_y), float(cap_x)


def _emergency_penalty_cost(
    *,
    residual_y: int,
    residual_x: int,
    y_force_px: float,
    x_force_px: float,
    planned_y_units: int,
    planned_x_units: int,
    panic_cross_bps: float,
) -> float:
    qty_y = max(0, int(residual_y))
    qty_x = max(0, int(residual_x))
    if qty_y <= 0 and qty_x <= 0:
        return 0.0
    planned_y = max(1, abs(int(planned_y_units)))
    planned_x = max(1, abs(int(planned_x_units)))
    residual_frac = float(
        max(float(qty_y) / float(planned_y), float(qty_x) / float(planned_x))
    )
    force_notional = float(
        abs(float(y_force_px)) * float(qty_y) + abs(float(x_force_px)) * float(qty_x)
    )
    extra_bps = float(max(0.0, float(panic_cross_bps)) * residual_frac * 0.10)
    return float(-force_notional * extra_bps / 10_000.0)


def _current_volume_at(
    symbol: str,
    ts: pd.Timestamp,
    *,
    volume_panel: pd.DataFrame | None,
    liq_model: LiquidityModel | None,
) -> float | None:
    if volume_panel is not None and symbol in volume_panel.columns:
        v = price_at_or_prior(volume_panel[symbol], ts, allow_zero=True)
        if v is not None and np.isfinite(v):
            return float(v)
    if liq_model is not None:
        v = price_at_or_prior(liq_model.volume(symbol), ts, allow_zero=True)
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def _full_fill(rep: Mapping[str, Any], req: int) -> bool:
    try:
        filled = int(rep.get("filled_size") or 0)
    except Exception:
        filled = 0
    return req <= 0 or (filled >= req and _pick_vwap(rep) is not None)


def _emergency_cross_price(
    *,
    symbol: str,
    units: int,
    ts: pd.Timestamp,
    liq_model: LiquidityModel | None,
    price_series: pd.Series | None,
    panic_cross_bps: float,
) -> float | None:
    bar = _day_bar(liq_model, symbol=symbol, ts=ts, price_series=price_series)
    close_px = bar["close"]
    high_px = bar["high"]
    low_px = bar["low"]
    if close_px is None:
        return None
    if units > 0:
        ref = min(
            float(close_px),
            float(low_px) if low_px is not None else float(close_px),
        )
        return float(ref) * (1.0 - float(panic_cross_bps) / 10_000.0)
    ref = max(
        float(close_px),
        float(high_px) if high_px is not None else float(close_px),
    )
    return float(ref) * (1.0 + float(panic_cross_bps) / 10_000.0)


def _session_gap(
    idx: pd.DatetimeIndex, planned_ts: pd.Timestamp, actual_ts: pd.Timestamp | None
) -> int:
    if actual_ts is None or pd.isna(actual_ts) or idx is None or len(idx) == 0:
        return 0
    norm_idx = idx.normalize()
    planned = _coerce_ts_to_index(pd.Timestamp(planned_ts), idx).normalize()
    actual = _coerce_ts_to_index(pd.Timestamp(actual_ts), idx).normalize()
    p0 = int(norm_idx.searchsorted(planned, side="left"))
    p1 = int(norm_idx.searchsorted(actual, side="left"))
    return int(max(0, p1 - p0))


def _exec_one_leg(
    *,
    symbol: str,
    units: int,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_mid: float | None,
    exit_mid: float | None,
    price_series: pd.Series | None,
    book_params: dict[str, Any],
    liq_model: LiquidityModel | None,
    shard_id: int,
    order_flow_entry: Mapping[str, Any] | None = None,
    order_flow_exit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if units == 0:
        return {
            "entry_vwap": None,
            "exit_vwap": None,
            "entry_ticks": 0.0,
            "exit_ticks": 0.0,
            "entry_liquidity": "taker",
            "exit_liquidity": "taker",
            "lob_spread_ticks": None,
            "lob_depth_top": None,
            "lob_adv_usd": None,
            "lob_sigma": None,
            "slippage_cost_entry": 0.0,
            "slippage_cost_exit": 0.0,
            "impact_cost_entry": 0.0,
            "impact_cost_exit": 0.0,
            "entry_filled_size": 0,
            "entry_unfilled_size": 0,
            "entry_requested_size": 0,
            "exit_filled_size": 0,
            "exit_unfilled_size": 0,
            "exit_requested_size": 0,
            "slippage_cost": 0.0,
            "impact_cost": 0.0,
        }

    req = int(abs(units))
    m0 = (
        float(entry_mid)
        if entry_mid is not None and np.isfinite(entry_mid) and entry_mid > 0
        else _mid_at(price_series, entry_ts)
    )
    m1 = (
        float(exit_mid)
        if exit_mid is not None and np.isfinite(exit_mid) and exit_mid > 0
        else _mid_at(price_series, exit_ts)
    )
    if m0 is None or m1 is None:
        return {
            "entry_vwap": None,
            "exit_vwap": None,
            "entry_ticks": 0.0,
            "exit_ticks": 0.0,
            "entry_liquidity": "blocked",
            "exit_liquidity": "blocked",
            "lob_spread_ticks": None,
            "lob_depth_top": None,
            "lob_adv_usd": None,
            "lob_sigma": None,
            "slippage_cost_entry": 0.0,
            "slippage_cost_exit": 0.0,
            "impact_cost_entry": 0.0,
            "impact_cost_exit": 0.0,
            "entry_filled_size": 0,
            "entry_unfilled_size": int(req),
            "entry_requested_size": int(req),
            "exit_filled_size": 0,
            "exit_unfilled_size": int(req),
            "exit_requested_size": int(req),
            "slippage_cost": 0.0,
            "impact_cost": 0.0,
        }

    params_entry = (
        liq_model.book_params(symbol, pd.Timestamp(entry_ts), base=book_params)
        if liq_model is not None
        else dict(book_params)
    )
    ob = _make_book(float(m0), params_entry, shard_id)
    rep0 = _exec_order(
        ob,
        symbol=symbol,
        side="buy" if units > 0 else "sell",
        qty=req,
        mid=float(m0),
        ts=pd.Timestamp(entry_ts),
        liq_model=liq_model,
        flow_cfg=order_flow_entry,
        is_short=bool(units < 0),
        maker_touch_mult=float(params_entry.get("_stress_maker_touch_mult", 1.0)),
        maker_state=params_entry,
    )
    fill0 = int(rep0.get("filled_size") or 0)
    unfill0 = int(rep0.get("unfilled_size") or max(0, req - fill0))

    if price_series is not None and isinstance(price_series.index, pd.DatetimeIndex):
        idx = _build_session_index(
            calendar=cast(pd.DatetimeIndex, price_series.index),
            y_series=None,
            x_series=None,
        )
        days = [
            ts
            for ts in idx
            if pd.Timestamp(ts).normalize() > pd.Timestamp(entry_ts).normalize()
            and pd.Timestamp(ts).normalize() <= pd.Timestamp(exit_ts).normalize()
        ]
        for day in days:
            mid_day = _mid_at(price_series, pd.Timestamp(day))
            if mid_day is None:
                continue
            params_day = (
                liq_model.book_params(symbol, pd.Timestamp(day), base=book_params)
                if liq_model is not None
                else dict(book_params)
            )
            _apply_book_state(
                ob,
                mid_price=float(mid_day),
                params=params_day,
                base_params=book_params,
                step_day=True,
            )

    params_exit = (
        liq_model.book_params(symbol, pd.Timestamp(exit_ts), base=book_params)
        if liq_model is not None
        else dict(book_params)
    )
    _apply_book_state(
        ob,
        mid_price=float(m1),
        params=params_exit,
        base_params=book_params,
        step_day=True,
    )
    rep1 = _exec_order(
        ob,
        symbol=symbol,
        side="sell" if units > 0 else "buy",
        qty=req,
        mid=float(m1),
        ts=pd.Timestamp(exit_ts),
        liq_model=liq_model,
        flow_cfg=order_flow_exit,
        is_short=False,
        maker_touch_mult=float(params_exit.get("_stress_maker_touch_mult", 1.0)),
        maker_state=params_exit,
    )
    fill1 = int(rep1.get("filled_size") or 0)
    unfill1 = int(rep1.get("unfilled_size") or max(0, req - fill1))
    slip0, imp0 = _slip_decomp(
        rep0,
        "buy" if units > 0 else "sell",
        ref_mid=float(m0),
    )
    slip1, imp1 = _slip_decomp(
        rep1,
        "sell" if units > 0 else "buy",
        ref_mid=float(m1),
    )
    return {
        "entry_vwap": _pick_vwap(rep0),
        "exit_vwap": _pick_vwap(rep1),
        "entry_ticks": float(_safe_float(rep0.get("slippage_ticks", 0.0), 0.0)),
        "exit_ticks": float(_safe_float(rep1.get("slippage_ticks", 0.0), 0.0)),
        "entry_liquidity": str(rep0.get("role", "taker") or "taker"),
        "exit_liquidity": str(rep1.get("role", "taker") or "taker"),
        "entry_filled_size": int(max(0, fill0)),
        "entry_unfilled_size": int(max(0, unfill0)),
        "entry_requested_size": int(max(0, req)),
        "exit_filled_size": int(max(0, fill1)),
        "exit_unfilled_size": int(max(0, unfill1)),
        "exit_requested_size": int(max(0, req)),
        "slippage_cost": float(slip0 + slip1),
        "impact_cost": float(imp0 + imp1),
        "slippage_cost_entry": float(slip0),
        "slippage_cost_exit": float(slip1),
        "impact_cost_entry": float(imp0),
        "impact_cost_exit": float(imp1),
        "lob_spread_ticks": params_entry.get("_liq_spread_ticks"),
        "lob_depth_top": params_entry.get("_liq_depth_top"),
        "lob_adv_usd": params_entry.get("_liq_adv_usd"),
        "lob_sigma": params_entry.get("_liq_sigma"),
    }


def _simulate_trade_row(
    *,
    row: pd.Series,
    row_ord: int,
    book_params: dict[str, Any],
    liq_model: LiquidityModel | None,
    fill_cfg: FillModelCfg,
    order_flow_entry: Mapping[str, Any],
    order_flow_exit: Mapping[str, Any],
    session_idx: pd.DatetimeIndex,
    y_series: pd.Series | None,
    x_series: pd.Series | None,
    volume_panel: pd.DataFrame | None,
    seed: int | None,
    stress_model: Mapping[str, Any],
) -> dict[str, Any]:
    planned_entry = pd.to_datetime(row.get("entry_date"), errors="coerce")
    planned_exit = pd.to_datetime(row.get("exit_date"), errors="coerce")
    y_sym = _normalize_symbol(
        row.get("y_symbol") or row.get("t1") or row.get("t1_symbol")
    )
    x_sym = _normalize_symbol(
        row.get("x_symbol") or row.get("t2") or row.get("t2_symbol")
    )
    y_units = int(row.get("y_units") or 0)
    x_units = int(row.get("x_units") or 0)

    default_out: dict[str, Any] = {
        "planned_entry_date": planned_entry,
        "planned_exit_date": planned_exit,
        "entry_date": planned_entry,
        "exit_date": planned_exit,
        "entry_price_y": np.nan,
        "entry_price_x": np.nan,
        "exit_price_y": np.nan,
        "exit_price_x": np.nan,
        "exec_actual_entry_date": pd.NaT,
        "exec_actual_exit_date": pd.NaT,
        "exec_entry_delay_days": 0,
        "exec_exit_delay_days": 0,
        "exec_entry_status": "blocked",
        "exec_exit_status": "",
        "exec_forced_exit": False,
        "exec_emergency_penalty_cost": 0.0,
        "exec_stress_regime_entry": "",
        "exec_stress_regime_exit": "",
        "exec_residual_units_y": 0,
        "exec_residual_units_x": 0,
        "exec_entry_vwap_y": np.nan,
        "exec_entry_vwap_x": np.nan,
        "exec_exit_vwap_y": np.nan,
        "exec_exit_vwap_x": np.nan,
        "exec_entry_ticks_y": 0.0,
        "exec_entry_ticks_x": 0.0,
        "exec_exit_ticks_y": 0.0,
        "exec_exit_ticks_x": 0.0,
        "liquidity_entry_y": "blocked",
        "liquidity_entry_x": "blocked",
        "liquidity_exit_y": "blocked",
        "liquidity_exit_x": "blocked",
        "liquidity_y": "blocked",
        "liquidity_x": "blocked",
        "lob_spread_ticks_y": np.nan,
        "lob_spread_ticks_x": np.nan,
        "lob_depth_top_y": np.nan,
        "lob_depth_top_x": np.nan,
        "lob_adv_usd_y": np.nan,
        "lob_adv_usd_x": np.nan,
        "lob_sigma_y": np.nan,
        "lob_sigma_x": np.nan,
        "lob_gap_bps_y": np.nan,
        "lob_gap_bps_x": np.nan,
        "lob_range_bps_y": np.nan,
        "lob_range_bps_x": np.nan,
        "lob_volume_rel_y": np.nan,
        "lob_volume_rel_x": np.nan,
        "lob_regime_entry": "",
        "lob_regime_exit": "",
        "exec_fill_frac": 0.0,
        "exec_fill_expected": 0.0,
        "slippage_cost": 0.0,
        "impact_cost": 0.0,
        "slippage_cost_entry": 0.0,
        "slippage_cost_exit": 0.0,
        "impact_cost_entry": 0.0,
        "impact_cost_exit": 0.0,
        "lob_gross_pnl": 0.0,
        "lob_net_pnl": 0.0,
        "gross_pnl": 0.0,
        "notional_y": 0.0,
        "notional_x": 0.0,
        "gross_notional": 0.0,
        "fees": 0.0,
        "fees_entry": 0.0,
        "fees_exit": 0.0,
        "exec_rejected": False,
        "exec_reject_reason": "",
        "exec_diag_costs_only": True,
        "y_units": int(y_units),
        "x_units": int(x_units),
        "units_y": int(y_units),
        "units_x": int(x_units),
    }

    if pd.isna(planned_entry) or pd.isna(planned_exit):
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_dates"
        return default_out
    if planned_exit < planned_entry:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "bad_dates"
        return default_out
    if not y_sym or not x_sym:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_symbol"
        return default_out
    if y_series is None or x_series is None:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_price_series"
        return default_out
    if len(session_idx) == 0:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_calendar"
        return default_out

    max_entry_delay_days = int(
        _safe_int(stress_model.get("max_entry_delay_days", 1), 1)
    )
    max_exit_grace_days = int(_safe_int(stress_model.get("max_exit_grace_days", 2), 2))
    panic_cross_bps = float(
        _safe_float(stress_model.get("panic_cross_bps", 50.0), 50.0)
    )

    entry_candidates = _candidate_sessions(
        session_idx,
        start_ts=planned_entry,
        max_delay_days=max_entry_delay_days,
        cap_ts=planned_exit,
    )
    if not entry_candidates:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_entry_session"
        return default_out

    actual_entry: pd.Timestamp | None = None
    entry_rep_y: dict[str, Any] | None = None
    entry_rep_x: dict[str, Any] | None = None
    entry_state_y: dict[str, Any] | None = None
    entry_state_x: dict[str, Any] | None = None
    ob_y: OrderBook | None = None
    ob_x: OrderBook | None = None

    for attempt_idx, entry_day in enumerate(entry_candidates):
        py_mid = _mid_at(y_series, entry_day)
        px_mid = _mid_at(x_series, entry_day)
        if py_mid is None or px_mid is None:
            continue

        fill_frac, fill_expected, y_state, x_state, pair_regime = _pair_fill_fraction(
            y_sym=y_sym,
            x_sym=x_sym,
            ts=entry_day,
            y_qty_abs=abs(y_units),
            x_qty_abs=abs(x_units),
            y_ref_px=float(py_mid),
            x_ref_px=float(px_mid),
            liq_model=liq_model,
            base_book_params=book_params,
            fill_cfg=fill_cfg,
            seed=seed,
            shard_id=int(row_ord * 16 + attempt_idx * 2 + 1),
        )
        default_out["exec_fill_frac"] = float(fill_frac)
        default_out["exec_fill_expected"] = float(fill_expected)
        default_out["lob_regime_entry"] = str(pair_regime)

        vol_y = _current_volume_at(
            y_sym, entry_day, volume_panel=volume_panel, liq_model=liq_model
        )
        vol_x = _current_volume_at(
            x_sym, entry_day, volume_panel=volume_panel, liq_model=liq_model
        )
        if (vol_y is not None and vol_y <= 0.0) or (vol_x is not None and vol_x <= 0.0):
            continue

        cap_pair_frac, _, _ = _pair_package_capacity(
            y_qty_abs=abs(y_units),
            x_qty_abs=abs(x_units),
            y_state=y_state,
            x_state=x_state,
            flow_cfg=order_flow_entry,
            fill_frac=fill_frac,
            base_book_params=book_params,
            exit_capacity_mult=1.0,
        )
        if cap_pair_frac < 1.0:
            continue

        ob_y_try = _make_book(float(py_mid), y_state, row_ord * 2 + 0)
        ob_x_try = _make_book(float(px_mid), x_state, row_ord * 2 + 1)

        rep_y = _exec_order(
            ob_y_try,
            symbol=y_sym,
            side="buy" if y_units > 0 else "sell",
            qty=abs(y_units),
            mid=float(py_mid),
            ts=entry_day,
            liq_model=liq_model,
            flow_cfg=order_flow_entry,
            is_short=bool(y_units < 0),
            maker_touch_mult=float(y_state.get("_stress_maker_touch_mult", 1.0)),
            maker_state=y_state,
        )
        rep_x = _exec_order(
            ob_x_try,
            symbol=x_sym,
            side="buy" if x_units > 0 else "sell",
            qty=abs(x_units),
            mid=float(px_mid),
            ts=entry_day,
            liq_model=liq_model,
            flow_cfg=order_flow_entry,
            is_short=bool(x_units < 0),
            maker_touch_mult=float(x_state.get("_stress_maker_touch_mult", 1.0)),
            maker_state=x_state,
        )
        if not (_full_fill(rep_y, abs(y_units)) and _full_fill(rep_x, abs(x_units))):
            continue

        actual_entry = pd.Timestamp(entry_day)
        entry_rep_y = rep_y
        entry_rep_x = rep_x
        entry_state_y = y_state
        entry_state_x = x_state
        ob_y = ob_y_try
        ob_x = ob_x_try
        break

    if (
        actual_entry is None
        or entry_rep_y is None
        or entry_rep_x is None
        or ob_y is None
        or ob_x is None
    ):
        default_out["entry_date"] = pd.NaT
        default_out["exit_date"] = pd.NaT
        default_out["planned_entry_date"] = planned_entry
        default_out["planned_exit_date"] = planned_exit
        default_out["exec_actual_entry_date"] = pd.NaT
        default_out["exec_actual_exit_date"] = pd.NaT
        default_out["y_units"] = 0
        default_out["x_units"] = 0
        default_out["units_y"] = 0
        default_out["units_x"] = 0
        return default_out

    entry_vwap_y = float(cast(float, _pick_vwap(entry_rep_y)))
    entry_vwap_x = float(cast(float, _pick_vwap(entry_rep_x)))
    default_out["exec_actual_entry_date"] = actual_entry
    default_out["entry_date"] = actual_entry
    default_out["exec_entry_status"] = (
        "filled"
        if pd.Timestamp(actual_entry).normalize()
        == pd.Timestamp(planned_entry).normalize()
        else "delayed"
    )
    default_out["entry_price_y"] = entry_vwap_y
    default_out["entry_price_x"] = entry_vwap_x
    default_out["exec_entry_vwap_y"] = entry_vwap_y
    default_out["exec_entry_vwap_x"] = entry_vwap_x
    default_out["exec_entry_ticks_y"] = float(
        _safe_float(entry_rep_y.get("slippage_ticks", 0.0), 0.0)
    )
    default_out["exec_entry_ticks_x"] = float(
        _safe_float(entry_rep_x.get("slippage_ticks", 0.0), 0.0)
    )
    default_out["liquidity_entry_y"] = str(
        entry_rep_y.get("role", "taker") or "taker"
    ).lower()
    default_out["liquidity_entry_x"] = str(
        entry_rep_x.get("role", "taker") or "taker"
    ).lower()
    default_out["liquidity_y"] = default_out["liquidity_entry_y"]
    default_out["liquidity_x"] = default_out["liquidity_entry_x"]
    default_out["lob_spread_ticks_y"] = (
        entry_state_y.get("_liq_spread_ticks") if entry_state_y is not None else np.nan
    )
    default_out["lob_spread_ticks_x"] = (
        entry_state_x.get("_liq_spread_ticks") if entry_state_x is not None else np.nan
    )
    default_out["lob_depth_top_y"] = (
        entry_state_y.get("_liq_depth_top") if entry_state_y is not None else np.nan
    )
    default_out["lob_depth_top_x"] = (
        entry_state_x.get("_liq_depth_top") if entry_state_x is not None else np.nan
    )
    default_out["lob_adv_usd_y"] = (
        entry_state_y.get("_liq_adv_usd") if entry_state_y is not None else np.nan
    )
    default_out["lob_adv_usd_x"] = (
        entry_state_x.get("_liq_adv_usd") if entry_state_x is not None else np.nan
    )
    default_out["lob_sigma_y"] = (
        entry_state_y.get("_liq_sigma") if entry_state_y is not None else np.nan
    )
    default_out["lob_sigma_x"] = (
        entry_state_x.get("_liq_sigma") if entry_state_x is not None else np.nan
    )
    default_out["lob_gap_bps_y"] = (
        entry_state_y.get("_stress_gap_bps") if entry_state_y is not None else np.nan
    )
    default_out["lob_gap_bps_x"] = (
        entry_state_x.get("_stress_gap_bps") if entry_state_x is not None else np.nan
    )
    default_out["lob_range_bps_y"] = (
        entry_state_y.get("_stress_range_bps") if entry_state_y is not None else np.nan
    )
    default_out["lob_range_bps_x"] = (
        entry_state_x.get("_stress_range_bps") if entry_state_x is not None else np.nan
    )
    default_out["lob_volume_rel_y"] = (
        entry_state_y.get("_stress_volume_rel") if entry_state_y is not None else np.nan
    )
    default_out["lob_volume_rel_x"] = (
        entry_state_x.get("_stress_volume_rel") if entry_state_x is not None else np.nan
    )
    default_out["exec_stress_regime_entry"] = str(
        default_out.get("lob_regime_entry", "")
    )

    slip0_y, imp0_y = _slip_decomp(
        entry_rep_y,
        "buy" if y_units > 0 else "sell",
        ref_mid=_mid_at(y_series, actual_entry),
    )
    slip0_x, imp0_x = _slip_decomp(
        entry_rep_x,
        "buy" if x_units > 0 else "sell",
        ref_mid=_mid_at(x_series, actual_entry),
    )
    default_out["slippage_cost_entry"] = float(slip0_y + slip0_x)
    default_out["impact_cost_entry"] = float(imp0_y + imp0_x)

    exit_start = max(pd.Timestamp(planned_exit), pd.Timestamp(actual_entry))
    exit_candidates = _candidate_sessions(
        session_idx,
        start_ts=exit_start,
        max_delay_days=max_exit_grace_days,
        cap_ts=None,
    )
    if not exit_candidates:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_exit_session"
        return default_out

    exit_regime_seed = "normal"
    for exit_day in exit_candidates:
        py_mid = _mid_at(y_series, pd.Timestamp(exit_day))
        px_mid = _mid_at(x_series, pd.Timestamp(exit_day))
        if py_mid is None or px_mid is None:
            continue
        exit_state_y0 = _state_for_symbol(
            liq_model=liq_model,
            symbol=y_sym,
            ts=pd.Timestamp(exit_day),
            base_book_params=book_params,
            mid_hint=float(py_mid),
        )
        exit_state_x0 = _state_for_symbol(
            liq_model=liq_model,
            symbol=x_sym,
            ts=pd.Timestamp(exit_day),
            base_book_params=book_params,
            mid_hint=float(px_mid),
        )
        exit_regime_seed = _worse_regime(
            str(exit_state_y0.get("_stress_regime", "normal")),
            str(exit_state_x0.get("_stress_regime", "normal")),
        )
        break
    exit_grace_days_eff = _effective_exit_grace_days(
        max_exit_grace_days, exit_regime_seed
    )
    exit_candidates = exit_candidates[: int(exit_grace_days_eff) + 1]
    if not exit_candidates:
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_exit_session"
        return default_out

    residual_y = abs(int(y_units))
    residual_x = abs(int(x_units))
    current_day = pd.Timestamp(actual_entry)
    exit_cash_y = 0.0
    exit_cash_x = 0.0
    exit_filled_y = 0
    exit_filled_x = 0
    exit_slip = 0.0
    exit_imp = 0.0
    actual_exit: pd.Timestamp | None = None
    exit_role_y = "blocked"
    exit_role_x = "blocked"
    forced_exit = False
    exit_regime_worst = str(exit_regime_seed)

    for attempt_idx, exit_day in enumerate(exit_candidates):
        days_to_step = [
            ts
            for ts in session_idx
            if pd.Timestamp(ts).normalize() > pd.Timestamp(current_day).normalize()
            and pd.Timestamp(ts).normalize() <= pd.Timestamp(exit_day).normalize()
        ]
        for step_day in days_to_step:
            mid_y = _mid_at(y_series, pd.Timestamp(step_day))
            mid_x = _mid_at(x_series, pd.Timestamp(step_day))
            if mid_y is not None:
                state_y = _state_for_symbol(
                    liq_model=liq_model,
                    symbol=y_sym,
                    ts=pd.Timestamp(step_day),
                    base_book_params=book_params,
                    mid_hint=mid_y,
                )
                _apply_book_state(
                    ob_y,
                    mid_price=float(mid_y),
                    params=state_y,
                    base_params=book_params,
                    step_day=True,
                )
            if mid_x is not None:
                state_x = _state_for_symbol(
                    liq_model=liq_model,
                    symbol=x_sym,
                    ts=pd.Timestamp(step_day),
                    base_book_params=book_params,
                    mid_hint=mid_x,
                )
                _apply_book_state(
                    ob_x,
                    mid_price=float(mid_x),
                    params=state_x,
                    base_params=book_params,
                    step_day=True,
                )
        current_day = pd.Timestamp(exit_day)
        py_mid = _mid_at(y_series, current_day)
        px_mid = _mid_at(x_series, current_day)
        if py_mid is None or px_mid is None:
            continue

        fill_frac, _, state_y, state_x, pair_regime = _pair_fill_fraction(
            y_sym=y_sym,
            x_sym=x_sym,
            ts=current_day,
            y_qty_abs=residual_y,
            x_qty_abs=residual_x,
            y_ref_px=float(py_mid),
            x_ref_px=float(px_mid),
            liq_model=liq_model,
            base_book_params=book_params,
            fill_cfg=fill_cfg,
            seed=seed,
            shard_id=int(row_ord * 64 + attempt_idx * 4 + 2),
        )
        exit_regime_worst = _worse_regime(exit_regime_worst, str(pair_regime))
        default_out["lob_regime_exit"] = str(exit_regime_worst)

        vol_y = _current_volume_at(
            y_sym, current_day, volume_panel=volume_panel, liq_model=liq_model
        )
        vol_x = _current_volume_at(
            x_sym, current_day, volume_panel=volume_panel, liq_model=liq_model
        )
        if (vol_y is not None and vol_y <= 0.0) or (vol_x is not None and vol_x <= 0.0):
            continue

        is_grace = (
            attempt_idx > 0
            or pd.Timestamp(current_day).normalize()
            > pd.Timestamp(planned_exit).normalize()
        )
        flow_exit_use = (
            {"mode": "taker", "fallback_to_taker": True}
            if is_grace
            else dict(order_flow_exit)
        )
        exit_capacity_mult = {
            "normal": 1.0,
            "elevated": 0.8,
            "stress": 0.55,
            "panic": 0.25,
        }.get(str(pair_regime).lower(), 1.0)
        attempt_frac, _, _ = _pair_package_capacity(
            y_qty_abs=residual_y,
            x_qty_abs=residual_x,
            y_state=state_y,
            x_state=state_x,
            flow_cfg=flow_exit_use,
            fill_frac=fill_frac,
            base_book_params=book_params,
            exit_capacity_mult=float(exit_capacity_mult),
        )
        qty_try_y = residual_y
        qty_try_x = residual_x
        if attempt_frac <= 0.0:
            qty_try_y = 0
            qty_try_x = 0
        else:
            qty_try_y = (
                max(1, int(np.floor(float(residual_y) * attempt_frac + 1e-12)))
                if residual_y > 0
                else 0
            )
            qty_try_x = (
                max(1, int(np.floor(float(residual_x) * attempt_frac + 1e-12)))
                if residual_x > 0
                else 0
            )

        rep_y = _exec_order(
            ob_y,
            symbol=y_sym,
            side="sell" if y_units > 0 else "buy",
            qty=qty_try_y,
            mid=float(py_mid),
            ts=current_day,
            liq_model=liq_model,
            flow_cfg=flow_exit_use,
            is_short=False,
            maker_touch_mult=float(state_y.get("_stress_maker_touch_mult", 1.0)),
            maker_state=state_y,
        )
        rep_x = _exec_order(
            ob_x,
            symbol=x_sym,
            side="sell" if x_units > 0 else "buy",
            qty=qty_try_x,
            mid=float(px_mid),
            ts=current_day,
            liq_model=liq_model,
            flow_cfg=flow_exit_use,
            is_short=False,
            maker_touch_mult=float(state_x.get("_stress_maker_touch_mult", 1.0)),
            maker_state=state_x,
        )
        fill_y = int(rep_y.get("filled_size") or 0)
        fill_x = int(rep_x.get("filled_size") or 0)
        if fill_y > 0:
            exit_role_y = str(rep_y.get("role", exit_role_y) or exit_role_y).lower()
            exit_px_y = _pick_vwap(rep_y)
            if exit_px_y is not None:
                exit_cash_y += float(exit_px_y) * float(fill_y)
                exit_filled_y += int(fill_y)
        if fill_x > 0:
            exit_role_x = str(rep_x.get("role", exit_role_x) or exit_role_x).lower()
            exit_px_x = _pick_vwap(rep_x)
            if exit_px_x is not None:
                exit_cash_x += float(exit_px_x) * float(fill_x)
                exit_filled_x += int(fill_x)

        slip_y, imp_y = _slip_decomp(
            rep_y,
            "sell" if y_units > 0 else "buy",
            ref_mid=float(py_mid),
        )
        slip_x, imp_x = _slip_decomp(
            rep_x,
            "sell" if x_units > 0 else "buy",
            ref_mid=float(px_mid),
        )
        exit_slip += float(slip_y + slip_x)
        exit_imp += float(imp_y + imp_x)

        residual_y = max(0, residual_y - fill_y)
        residual_x = max(0, residual_x - fill_x)
        if residual_y == 0 and residual_x == 0:
            actual_exit = pd.Timestamp(current_day)
            break

    if residual_y > 0 or residual_x > 0:
        forced_day = pd.Timestamp(exit_candidates[-1])
        force_regime = _worse_regime(
            exit_regime_worst, str(default_out.get("lob_regime_exit", "normal"))
        )
        force_bps = _panic_cross_bps_for_regime(panic_cross_bps, force_regime)
        residual_before_force_y = int(residual_y)
        residual_before_force_x = int(residual_x)
        py_force = _emergency_cross_price(
            symbol=y_sym,
            units=y_units,
            ts=forced_day,
            liq_model=liq_model,
            price_series=y_series,
            panic_cross_bps=force_bps,
        )
        px_force = _emergency_cross_price(
            symbol=x_sym,
            units=x_units,
            ts=forced_day,
            liq_model=liq_model,
            price_series=x_series,
            panic_cross_bps=force_bps,
        )
        if py_force is None or px_force is None:
            default_out["exec_rejected"] = True
            default_out["exec_reject_reason"] = "missing_force_price"
            return default_out
        if residual_y > 0:
            exit_cash_y += float(py_force) * float(residual_y)
            exit_filled_y += int(residual_y)
            exit_role_y = "forced"
            residual_y = 0
        if residual_x > 0:
            exit_cash_x += float(px_force) * float(residual_x)
            exit_filled_x += int(residual_x)
            exit_role_x = "forced"
            residual_x = 0
        forced_exit = True
        actual_exit = forced_day
        default_out["exec_emergency_penalty_cost"] = _emergency_penalty_cost(
            residual_y=residual_before_force_y,
            residual_x=residual_before_force_x,
            y_force_px=float(py_force),
            x_force_px=float(px_force),
            planned_y_units=int(y_units),
            planned_x_units=int(x_units),
            panic_cross_bps=float(force_bps),
        )
        default_out["exec_residual_units_y"] = int(residual_before_force_y)
        default_out["exec_residual_units_x"] = int(residual_before_force_x)

    exit_vwap_y = float(exit_cash_y / exit_filled_y) if exit_filled_y > 0 else np.nan
    exit_vwap_x = float(exit_cash_x / exit_filled_x) if exit_filled_x > 0 else np.nan
    if (
        actual_exit is None
        or not np.isfinite(exit_vwap_y)
        or not np.isfinite(exit_vwap_x)
    ):
        default_out["exec_rejected"] = True
        default_out["exec_reject_reason"] = "missing_exit_execution"
        return default_out

    default_out["exec_actual_exit_date"] = actual_exit
    default_out["exit_date"] = actual_exit
    default_out["exit_price_y"] = float(exit_vwap_y)
    default_out["exit_price_x"] = float(exit_vwap_x)
    default_out["exec_exit_vwap_y"] = float(exit_vwap_y)
    default_out["exec_exit_vwap_x"] = float(exit_vwap_x)
    default_out["liquidity_exit_y"] = str(exit_role_y).lower()
    default_out["liquidity_exit_x"] = str(exit_role_x).lower()
    default_out["exec_exit_status"] = (
        "forced"
        if forced_exit
        else (
            "filled"
            if pd.Timestamp(actual_exit).normalize()
            == pd.Timestamp(planned_exit).normalize()
            else "delayed"
        )
    )
    default_out["exec_forced_exit"] = bool(forced_exit)
    default_out["exec_stress_regime_exit"] = str(default_out.get("lob_regime_exit", ""))
    if not forced_exit:
        default_out["exec_residual_units_y"] = int(residual_y)
        default_out["exec_residual_units_x"] = int(residual_x)
    default_out["slippage_cost_exit"] = float(exit_slip)
    default_out["impact_cost_exit"] = float(exit_imp)
    default_out["slippage_cost"] = float(default_out["slippage_cost_entry"]) + float(
        exit_slip
    )
    default_out["impact_cost"] = float(default_out["impact_cost_entry"]) + float(
        exit_imp
    )

    gross = float(y_units) * (float(exit_vwap_y) - float(entry_vwap_y)) + float(
        x_units
    ) * (float(exit_vwap_x) - float(entry_vwap_x))
    default_out["lob_gross_pnl"] = float(gross)
    default_out["lob_net_pnl"] = float(gross)
    default_out["gross_pnl"] = float(gross)
    default_out["notional_y"] = float(y_units) * float(entry_vwap_y)
    default_out["notional_x"] = float(x_units) * float(entry_vwap_x)
    default_out["gross_notional"] = float(
        abs(default_out["notional_y"]) + abs(default_out["notional_x"])
    )
    default_out["exec_entry_delay_days"] = _session_gap(
        session_idx, planned_entry, actual_entry
    )
    default_out["exec_exit_delay_days"] = _session_gap(
        session_idx, planned_exit, actual_exit
    )
    return default_out


def annotate_with_lob(
    trades_df: pd.DataFrame,
    price_data: Mapping[str, pd.Series],
    cfg_obj: Any,
    *,
    market_data_panel: pd.DataFrame | None = None,
    adv_map: Mapping[str, float] | None = None,
    calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return trades_df

    df = trades_df.copy()
    df.attrs = dict(getattr(trades_df, "attrs", {}) or {})
    lob_cfg = _resolve_exec_lob_cfg(cfg_obj)
    book_params = _mk_book_params(cfg_obj)
    df.attrs["tick"] = float(book_params["tick"])

    liq_cfg = LiquidityModelCfg.from_exec_lob(lob_cfg)
    liq_model: LiquidityModel | None = None
    try:
        if (
            liq_cfg.enabled
            and market_data_panel is not None
            and isinstance(market_data_panel, pd.DataFrame)
            and not market_data_panel.empty
            and isinstance(market_data_panel.columns, pd.MultiIndex)
            and market_data_panel.columns.nlevels >= 2
        ):
            liq_model = LiquidityModel(
                market_data_panel, cfg=liq_cfg, adv_map_usd=adv_map
            )
            df.attrs["lob_liq_model"] = "enabled"
        else:
            df.attrs["lob_liq_model"] = "disabled"
    except Exception:
        liq_model = None
        df.attrs["lob_liq_model"] = "failed"

    fill_cfg = FillModelCfg.from_exec_lob(lob_cfg)
    df.attrs["lob_fill_model"] = "enabled" if fill_cfg.enabled else "disabled"

    volume_panel: pd.DataFrame | None = None
    if (
        market_data_panel is not None
        and isinstance(market_data_panel, pd.DataFrame)
        and isinstance(market_data_panel.columns, pd.MultiIndex)
        and market_data_panel.columns.nlevels >= 2
    ):
        try:
            volume_panel = select_field_from_panel(market_data_panel, field="volume")
        except Exception:
            volume_panel = None

    df["entry_date"] = pd.to_datetime(
        cast(Any, df.get("entry_date", pd.NaT)), errors="coerce"
    )
    df["exit_date"] = pd.to_datetime(
        cast(Any, df.get("exit_date", pd.NaT)), errors="coerce"
    )

    intended_y: list[int] = []
    intended_x: list[int] = []
    explicit_units_input: list[bool] = []
    for _, row in df.iterrows():
        if (
            "y_units" in row.index
            and "x_units" in row.index
            and pd.notna(row.get("y_units"))
            and pd.notna(row.get("x_units"))
        ):
            explicit_units = True
            try:
                y_units = int(round(float(cast(Any, row.get("y_units")))))
                x_units = int(round(float(cast(Any, row.get("x_units")))))
            except Exception:
                y_units, x_units = _trade_units_from_row(row)
                explicit_units = False
        else:
            explicit_units = False
            y_units, x_units = _trade_units_from_row(row)
        intended_y.append(int(y_units))
        intended_x.append(int(x_units))
        explicit_units_input.append(bool(explicit_units))

    df["y_units"] = intended_y
    df["x_units"] = intended_x
    df["intended_y_units"] = intended_y
    df["intended_x_units"] = intended_x
    df["units_y"] = intended_y
    df["units_x"] = intended_x
    df["_lob_explicit_units_input"] = pd.Series(
        explicit_units_input, index=df.index, dtype=bool
    )

    order_flow_entry: dict[str, Any] = {}
    order_flow_exit: dict[str, Any] = {}
    try:
        order_flow_raw = dict(lob_cfg.get("order_flow") or {})
        base = {k: v for k, v in order_flow_raw.items() if k not in {"entry", "exit"}}
        order_flow_entry = {**base, **dict(order_flow_raw.get("entry") or {})}
        order_flow_exit = {**base, **dict(order_flow_raw.get("exit") or {})}
    except Exception:
        order_flow_entry = {}
        order_flow_exit = {}

    stress_model = dict(lob_cfg.get("stress_model") or {})
    seed = cast(int | None, book_params.get("seed", None))
    results: list[dict[str, Any]] = []
    for row_ord, (_, row) in enumerate(df.iterrows()):
        y_sym = _normalize_symbol(
            row.get("y_symbol") or row.get("t1") or row.get("t1_symbol")
        )
        x_sym = _normalize_symbol(
            row.get("x_symbol") or row.get("t2") or row.get("t2_symbol")
        )
        y_series = price_data.get(y_sym)
        x_series = price_data.get(x_sym)
        session_idx = _build_session_index(
            calendar=calendar, y_series=y_series, x_series=x_series
        )
        try:
            sim = _simulate_trade_row(
                row=row,
                row_ord=row_ord,
                book_params=book_params,
                liq_model=liq_model,
                fill_cfg=fill_cfg,
                order_flow_entry=order_flow_entry,
                order_flow_exit=order_flow_exit,
                session_idx=session_idx,
                y_series=y_series,
                x_series=x_series,
                volume_panel=volume_panel,
                seed=seed,
                stress_model=stress_model,
            )
        except Exception as exc:
            logger.warning("LOB simulation failed for row %s: %s", row_ord, exc)
            sim = {
                "exec_rejected": True,
                "exec_reject_reason": "lob_runtime_error",
            }
        results.append(sim)

    sim_df = pd.DataFrame(results, index=df.index)
    for col in sim_df.columns:
        df[col] = sim_df[col]

    try:
        post_costs = compute_post_lob_costs(df, lob_cfg)
        for col in ("fees", "fees_entry", "fees_exit"):
            if col in post_costs.columns:
                df[col] = (
                    pd.to_numeric(post_costs[col], errors="coerce")
                    .fillna(0.0)
                    .astype(float)
                )
    except Exception as exc:
        logger.warning("post_lob_costs skipped: %s", exc)
        for col in ("fees", "fees_entry", "fees_exit"):
            if col not in df.columns:
                df[col] = 0.0

    buyin_penalty = pd.to_numeric(
        pd.Series(df.get("buyin_penalty_cost", 0.0), index=df.index),
        errors="coerce",
    ).fillna(0.0)
    emergency_penalty = pd.to_numeric(
        pd.Series(df.get("exec_emergency_penalty_cost", 0.0), index=df.index),
        errors="coerce",
    ).fillna(0.0)
    df["lob_net_pnl"] = (
        pd.to_numeric(df["lob_gross_pnl"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)
        + buyin_penalty
        + emergency_penalty
    )
    df["gross_pnl"] = pd.to_numeric(df["lob_gross_pnl"], errors="coerce").fillna(0.0)
    df["exec_diag_costs_only"] = True

    blocked = (
        df.get("exec_entry_status", pd.Series("", index=df.index))
        .astype(str)
        .eq("blocked")
    )
    delayed_entry = (
        df.get("exec_entry_status", pd.Series("", index=df.index))
        .astype(str)
        .eq("delayed")
    )
    delayed_exit = (
        df.get("exec_exit_status", pd.Series("", index=df.index))
        .astype(str)
        .eq("delayed")
    )
    forced_exit = (
        pd.Series(df.get("exec_forced_exit", False), index=df.index)
        .fillna(False)
        .astype(bool)
    )
    entry_hist = (
        df.get("lob_regime_entry", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .value_counts()
    )
    exit_hist = (
        df.get("lob_regime_exit", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .value_counts()
    )
    df.attrs["exec_entry_blocked_count"] = int(blocked.sum())
    df.attrs["exec_delayed_entry_count"] = int(delayed_entry.sum())
    df.attrs["exec_delayed_exit_count"] = int(delayed_exit.sum())
    df.attrs["exec_forced_exit_count"] = int(forced_exit.sum())
    df.attrs["exec_regime_histogram"] = {
        "entry": {str(k): int(v) for k, v in entry_hist.to_dict().items() if str(k)},
        "exit": {str(k): int(v) for k, v in exit_hist.to_dict().items() if str(k)},
    }

    if "_lob_explicit_units_input" in df.columns:
        df = df.drop(columns=["_lob_explicit_units_input"])
    return df
