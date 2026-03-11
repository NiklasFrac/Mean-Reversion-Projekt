from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ExecutionCostSpec:
    mode: Literal["lob", "light"]
    enabled: bool
    per_trade: float
    fee_bps: float
    per_share_fee: float
    min_fee: float
    max_fee: float
    source: str


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _to_nonneg_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = default
    if out < 0.0:
        return 0.0
    return float(out)


def resolve_execution_cost_spec(cfg: Mapping[str, Any]) -> ExecutionCostSpec:
    ex = _as_dict(cfg.get("execution"))
    mode_raw = str(ex.get("mode", "lob")).strip().lower()
    mode: Literal["lob", "light"] = "light" if mode_raw == "light" else "lob"

    if mode == "light":
        light = _as_dict(ex.get("light"))
        fees = _as_dict(light.get("fees"))
        return ExecutionCostSpec(
            mode="light",
            enabled=bool(light.get("enabled", True)),
            per_trade=_to_nonneg_float(fees.get("per_trade", 0.0)),
            fee_bps=_to_nonneg_float(fees.get("bps", 0.0)),
            per_share_fee=_to_nonneg_float(fees.get("per_share", 0.0)),
            min_fee=_to_nonneg_float(fees.get("min_fee", 0.0)),
            max_fee=_to_nonneg_float(fees.get("max_fee", 0.0)),
            source="execution.light.fees",
        )

    lob = _as_dict(ex.get("lob"))
    post_costs = _as_dict(lob.get("post_costs"))
    taker_bps = _to_nonneg_float(post_costs.get("taker_bps", 0.0))
    taker_ps = _to_nonneg_float(post_costs.get("taker_per_share", 0.0))
    return ExecutionCostSpec(
        mode="lob",
        enabled=bool(lob.get("enabled", True)),
        per_trade=_to_nonneg_float(post_costs.get("per_trade", 0.0)),
        fee_bps=taker_bps,
        per_share_fee=taker_ps,
        min_fee=_to_nonneg_float(post_costs.get("min_fee", 0.0)),
        max_fee=_to_nonneg_float(post_costs.get("max_fee", 0.0)),
        source="execution.lob.post_costs",
    )


def zero_fee_patch_for_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = resolve_execution_cost_spec(cfg)
    if spec.mode == "light":
        return {
            "execution": {
                "light": {
                    "fees": {
                        "per_trade": 0.0,
                        "bps": 0.0,
                        "per_share": 0.0,
                        "min_fee": 0.0,
                        "max_fee": 0.0,
                    }
                }
            }
        }
    return {
        "execution": {
            "lob": {
                "post_costs": {
                    "per_trade": 0.0,
                    "maker_bps": 0.0,
                    "taker_bps": 0.0,
                    "maker_per_share": 0.0,
                    "taker_per_share": 0.0,
                    "min_fee": 0.0,
                    "max_fee": 0.0,
                }
            }
        }
    }


def execution_cost_flags(cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = resolve_execution_cost_spec(cfg)
    return {
        "exec_mode": spec.mode,
        "fees_per_trade": float(spec.per_trade),
        "fees_bps": float(spec.fee_bps),
        "fees_per_share": float(spec.per_share_fee),
        "fees_min": float(spec.min_fee),
        "fees_max": float(spec.max_fee),
        "exec_cost_enabled": bool(spec.enabled),
        "exec_cost_source": str(spec.source),
    }


__all__ = [
    "ExecutionCostSpec",
    "execution_cost_flags",
    "resolve_execution_cost_spec",
    "zero_fee_patch_for_cfg",
]
