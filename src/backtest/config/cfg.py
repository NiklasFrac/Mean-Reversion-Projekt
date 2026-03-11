"""
BacktestConfig + YAML loader (conservative windowing only).

Notes:
- Supported execution modes are lob and light.
- The only supported run mode is conservative via `backtest.splits`.
- Raw YAML is preserved in BacktestConfig.raw_yaml for optional modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, cast

import pandas as pd

from backtest.windowing.splits import require_backtest_splits

from .yaml_cfg_extractors import (
    _extract_calendar_cfg_for_yaml,
    _extract_execution_cfg_for_yaml,
)

__all__ = ["BacktestConfig", "make_config_from_yaml"]


_LOB_BOOK_DEFAULTS: dict[str, tuple[type[int] | type[float], int | float]] = {
    "tick": (float, 0.01),
    "levels": (int, 10),
    "size_per_level": (int, 800),
    "min_spread_ticks": (int, 1),
    "steps_per_day": (int, 390),
    "lam": (float, 12.0),
    "max_add": (int, 500),
    "bias_top": (float, 0.7),
    "cancel_prob": (float, 0.15),
    "max_cancel": (int, 200),
}
_LOB_LEGACY_KEYS = (
    "book",
    "step",
    "orders",
    "policy",
    "fees",
    "impact",
    "diagnostics",
    "pricing",
)
_LOB_DEAD_POST_COST_KEYS = (
    "borrow_bps",
    "borrow_day_basis",
    "borrow_short_only",
    "report_slippage_impact",
)
_LOB_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "liq_model": (
        "enabled",
        "asof_shift",
        "vol_window",
        "adv_window",
        "min_periods_frac",
        "spread_floor_bps",
        "spread_sigma_mult",
        "spread_adv_mult",
        "adv_ref_usd",
        "depth_frac_of_adv_shares",
        "depth_gamma",
        "min_depth_shares",
        "max_depth_shares",
        "min_level_shares",
        "lam_adv_power",
        "lam_min",
        "lam_max",
        "cancel_base",
        "cancel_sigma_mult",
        "cancel_min",
        "cancel_max",
        "max_add_frac_of_top",
        "max_cancel_frac_of_top",
        "max_add_min",
        "max_cancel_min",
        "max_add_max",
        "max_cancel_max",
        "tick_subpenny",
        "tick_penny",
        "tick_switch_price",
    ),
    "fill_model": (
        "enabled",
        "base_fill",
        "safe_depth_share",
        "depth_share_50",
        "depth_shape",
        "safe_participation",
        "participation_50",
        "participation_shape",
        "sigma_mult",
        "beta_kappa_base",
        "beta_kappa_adv_power",
        "beta_kappa_min",
        "beta_kappa_max",
        "allow_reject",
        "reject_below",
        "min_fill_if_filled",
    ),
    "post_costs": (
        "per_trade",
        "maker_bps",
        "taker_bps",
        "maker_per_share",
        "taker_per_share",
        "min_fee",
        "max_fee",
    ),
    "order_flow": (
        "mode",
        "maker_price",
        "maker_prob",
        "maker_max_top_frac",
        "maker_touch_prob",
        "fallback_to_taker",
        "entry",
        "exit",
    ),
    "stress_model": (
        "enabled",
        "intensity",
        "max_entry_delay_days",
        "max_exit_grace_days",
        "panic_cross_bps",
    ),
}
_LOB_ORDER_FLOW_CHILD_KEYS = (
    "mode",
    "maker_price",
    "maker_prob",
    "maker_max_top_frac",
    "maker_touch_prob",
    "fallback_to_taker",
)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    splits: dict[str, dict[str, Any]] | None = None
    calendar_mapping: str = "prior"
    annualization_factor: int | None = None
    settlement_lag_bars: int = 0
    risk_enabled: bool = False
    risk_cfg: dict[str, Any] | None = None
    exec_mode: str = "lob"
    exec_lob: dict[str, Any] | None = None
    exec_light: dict[str, Any] | None = None
    raw_yaml: dict[str, Any] = field(default_factory=dict)


def _clamp_exec_mode(s: str) -> str:
    s = (s or "lob").lower()
    return s if s in {"lob", "light"} else "lob"


def _clamp_calendar_mapping(s: str) -> str:
    s = (s or "prior").lower()
    return s if s in {"prior", "strict", "next"} else "prior"


def _as_dict(x: Any) -> dict[str, Any] | None:
    return dict(x) if isinstance(x, Mapping) else None


def _nonneg_float(val: Any, default: float = 0.0) -> float:
    try:
        out = float(val)
    except Exception:
        out = default
    if not pd.notna(out):
        return float(default)
    return max(0.0, float(out))


def _safe_int_like(val: Any, default: int = 0) -> int:
    try:
        out = int(round(float(val)))
    except Exception:
        out = int(default)
    return int(out)


def _copy_allowed(
    d: Mapping[str, Any] | None, allowed: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(d, Mapping):
        return {}
    return {key: d[key] for key in allowed if key in d}


def _reject_legacy_keys(cfg: Mapping[str, Any], bt: Mapping[str, Any]) -> None:
    walkforward = bt.get("walkforward")
    wf = walkforward if isinstance(walkforward, Mapping) else {}
    ex = cfg.get("execution") if isinstance(cfg.get("execution"), Mapping) else {}
    lob = (
        ex.get("lob")
        if isinstance(ex, Mapping) and isinstance(ex.get("lob"), Mapping)
        else {}
    )
    post_costs = (
        lob.get("post_costs")
        if isinstance(lob, Mapping) and isinstance(lob.get("post_costs"), Mapping)
        else {}
    )

    legacy_keys: list[str] = []
    if "costs" in cfg:
        legacy_keys.append("costs")
    if "window_end_policy" in bt:
        legacy_keys.append("backtest.window_end_policy")
    if "carry_open_trades" in wf:
        legacy_keys.append("backtest.walkforward.carry_open_trades")
    if "stateful_sizing" in wf:
        legacy_keys.append("backtest.walkforward.stateful_sizing")
    if isinstance(ex, Mapping) and "exec_lob" in ex:
        legacy_keys.append("execution.exec_lob")
    if isinstance(ex, Mapping) and "override_pnl" in ex:
        legacy_keys.append("execution.override_pnl")
    if isinstance(lob, Mapping):
        for key in _LOB_LEGACY_KEYS:
            if key in lob:
                legacy_keys.append(f"execution.lob.{key}")
    if isinstance(post_costs, Mapping):
        for key in _LOB_DEAD_POST_COST_KEYS:
            if key in post_costs:
                legacy_keys.append(f"execution.lob.post_costs.{key}")

    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(
            f"Legacy config keys are no longer supported: {joined}. "
            "Use execution.light.fees for non-LOB fees and execution.lob.post_costs for LOB fees. "
            "LOB now always uses execution-authoritative PnL, so execution.override_pnl was removed. "
            "Trades now carry across intermediate walk-forward windows automatically and "
            "are hard-exited only on the final global OOS day."
        )


def _coerce_lob_cfg_from_dict(d: Mapping[str, Any] | None) -> dict[str, Any]:
    lob = dict(d or {})
    out: dict[str, Any] = {"enabled": bool(lob.get("enabled", True))}
    for key, (tp, default) in _LOB_BOOK_DEFAULTS.items():
        if tp is int:
            try:
                out[key] = int(round(float(lob.get(key, default))))
            except Exception:
                out[key] = int(default)
        else:
            try:
                val = float(lob.get(key, default))
            except Exception:
                val = float(default)
            out[key] = val if pd.notna(val) else float(default)

    liq_model = _copy_allowed(
        _as_dict(lob.get("liq_model")), _LOB_SECTION_KEYS["liq_model"]
    )
    if liq_model:
        out["liq_model"] = liq_model

    fill_model = _copy_allowed(
        _as_dict(lob.get("fill_model")), _LOB_SECTION_KEYS["fill_model"]
    )
    if fill_model:
        out["fill_model"] = fill_model

    post_costs = _copy_allowed(
        _as_dict(lob.get("post_costs")), _LOB_SECTION_KEYS["post_costs"]
    )
    if post_costs:
        out["post_costs"] = post_costs

    order_flow = _copy_allowed(
        _as_dict(lob.get("order_flow")), _LOB_SECTION_KEYS["order_flow"]
    )
    if order_flow:
        if "entry" in order_flow:
            order_flow["entry"] = _copy_allowed(
                _as_dict(order_flow.get("entry")), _LOB_ORDER_FLOW_CHILD_KEYS
            )
        if "exit" in order_flow:
            order_flow["exit"] = _copy_allowed(
                _as_dict(order_flow.get("exit")), _LOB_ORDER_FLOW_CHILD_KEYS
            )
        out["order_flow"] = order_flow

    stress_model = _copy_allowed(
        _as_dict(lob.get("stress_model")), _LOB_SECTION_KEYS["stress_model"]
    )
    out["stress_model"] = {
        "enabled": bool(stress_model.get("enabled", True)),
        "intensity": _nonneg_float(stress_model.get("intensity", 1.0), 1.0),
        "max_entry_delay_days": max(
            0, _safe_int_like(stress_model.get("max_entry_delay_days", 1), 1)
        ),
        "max_exit_grace_days": max(
            0, _safe_int_like(stress_model.get("max_exit_grace_days", 2), 2)
        ),
        "panic_cross_bps": _nonneg_float(
            stress_model.get("panic_cross_bps", 50.0), 50.0
        ),
    }

    return out


def _coerce_light_cfg_from_dict(d: Mapping[str, Any] | None) -> dict[str, Any]:
    d = dict(d or {})
    fees: Mapping[str, Any] = (
        cast(Mapping[str, Any], d.get("fees"))
        if isinstance(d.get("fees"), Mapping)
        else {}
    )
    return {
        "enabled": bool(d.get("enabled", True)),
        "reject_on_missing_price": bool(d.get("reject_on_missing_price", True)),
        "fees": {
            "per_trade": _nonneg_float(fees.get("per_trade", 0.0)),
            "bps": _nonneg_float(fees.get("bps", 0.0)),
            "per_share": _nonneg_float(fees.get("per_share", 0.0)),
            "min_fee": _nonneg_float(fees.get("min_fee", 0.0)),
            "max_fee": _nonneg_float(fees.get("max_fee", 0.0)),
        },
    }


def make_config_from_yaml(yaml_cfg: dict[str, Any]) -> BacktestConfig:
    cfg = dict(yaml_cfg or {})

    bt = (cfg.get("backtest") or {}) if isinstance(cfg.get("backtest"), dict) else {}
    _reject_legacy_keys(cfg, bt)
    risk = (cfg.get("risk") or {}) if isinstance(cfg.get("risk"), dict) else {}
    exec_norm = _extract_execution_cfg_for_yaml(cfg)
    cal_norm = _extract_calendar_cfg_for_yaml(bt, cfg)

    lob_norm_in = exec_norm.get("lob", {}) if exec_norm.get("mode") == "lob" else {}
    lob_cfg = (
        _coerce_lob_cfg_from_dict(lob_norm_in)
        if isinstance(lob_norm_in, Mapping)
        else None
    )
    light_norm_in = exec_norm.get("light", {})
    light_cfg = (
        _coerce_light_cfg_from_dict(light_norm_in)
        if isinstance(light_norm_in, Mapping)
        else None
    )

    exec_mode = _clamp_exec_mode(str(exec_norm.get("mode", "lob")))
    cal_map = _clamp_calendar_mapping(
        str(bt.get("calendar_mapping", cal_norm.get("calendar_mapping", "prior")))
    )

    splits = require_backtest_splits(
        cfg,
        keys=("train", "test"),
        err_cls=KeyError,
        err_msg="backtest.splits.{train,test} missing (conservative mode is required)",
    )
    splits_typed = cast(dict[str, dict[str, Any]], splits)

    try:
        tr0 = pd.to_datetime(cast(Any, (splits.get("train") or {}).get("start")))
        tr1 = pd.to_datetime(cast(Any, (splits.get("train") or {}).get("end")))
        te0 = pd.to_datetime(cast(Any, (splits.get("test") or {}).get("start")))
        te1 = pd.to_datetime(cast(Any, (splits.get("test") or {}).get("end")))
        if not (tr0 <= tr1 < te0 <= te1):
            raise ValueError(
                "backtest.splits must be disjoint and ordered: train < test"
            )
        if "analysis" in splits:
            an0 = pd.to_datetime(cast(Any, (splits.get("analysis") or {}).get("start")))
            an1 = pd.to_datetime(cast(Any, (splits.get("analysis") or {}).get("end")))
            if not (an0 <= an1 < tr0):
                raise ValueError(
                    "backtest.splits must be disjoint and ordered: analysis < train < test"
                )
    except Exception as e:
        raise ValueError(f"Invalid backtest.splits: {e}") from e

    out = BacktestConfig(
        initial_capital=max(0.0, float(bt.get("initial_capital", 1_000_000.0))),
        splits=splits_typed,
        calendar_mapping=cal_map,
        annualization_factor=bt.get("annualization_factor"),
        settlement_lag_bars=max(
            0,
            int(bt.get("settlement_lag_bars", cal_norm.get("settlement_lag_bars", 0))),
        ),
        risk_enabled=bool(risk.get("enabled", False)),
        risk_cfg=risk if isinstance(risk, dict) else {},
        exec_mode=exec_mode,
        exec_lob=lob_cfg,
        exec_light=light_cfg,
        raw_yaml=dict(cfg),
    )

    if out.exec_mode == "lob" and out.exec_lob is None:
        out = replace(out, exec_lob=_coerce_lob_cfg_from_dict({}))
    if out.exec_mode == "light" and out.exec_light is None:
        out = replace(out, exec_light=_coerce_light_cfg_from_dict({}))

    return out
