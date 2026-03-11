from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "RiskBorrowPolicy",
    "RiskExposurePolicy",
    "RiskPolicy",
    "RiskSizingPolicy",
    "ShortAvailabilityHeuristic",
    "build_risk_policy",
    "cap_units_by_participation",
    "cap_units_by_trade_notional",
    "is_short_leg",
    "size_units_from_risk_budget",
]


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "none",
            "null",
            "nan",
        }:
            return float(default)
        out = float(value)
        if math.isnan(out):
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _to_float_opt(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "none",
            "null",
            "nan",
        }:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return float(out)
    except Exception:
        return None


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on", "y"}:
            return True
        if s in {"0", "false", "no", "off", "n"}:
            return False
    return bool(default)


@dataclass(frozen=True)
class ShortAvailabilityHeuristic:
    enabled: bool = False
    min_price: float = 0.0
    min_adv_usd: float = 0.0
    block_on_missing: bool = True


@dataclass(frozen=True)
class RiskSizingPolicy:
    risk_per_trade: float = 0.01
    max_trade_pct: float = 0.10
    max_participation: float = 0.10


@dataclass(frozen=True)
class RiskExposurePolicy:
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    max_per_name_pct: float | None = None
    max_open_positions: int | None = None
    max_positions_per_symbol: int | None = None
    strict: bool = False


@dataclass(frozen=True)
class RiskBorrowPolicy:
    require_shortable_flag: bool = True
    cap_by_availability: bool = True


@dataclass(frozen=True)
class RiskPolicy:
    sizing: RiskSizingPolicy
    exposure: RiskExposurePolicy
    borrow: RiskBorrowPolicy
    short_heuristic: ShortAvailabilityHeuristic


def build_risk_policy(
    *,
    risk_cfg: Mapping[str, Any] | None = None,
    backtest_cfg: Mapping[str, Any] | None = None,
    execution_cfg: Mapping[str, Any] | None = None,
) -> RiskPolicy:
    risk = dict(risk_cfg or {})
    bt = dict(backtest_cfg or {})
    ex = dict(execution_cfg or {})
    caps = dict(risk.get("caps") or {}) if isinstance(risk.get("caps"), Mapping) else {}
    short_raw = (
        dict(risk.get("short_availability_heuristic") or {})
        if isinstance(risk.get("short_availability_heuristic"), Mapping)
        else {}
    )

    risk_per_trade = _to_float(
        risk.get("risk_per_trade", bt.get("risk_per_trade", 0.01)), 0.01
    )
    max_trade_pct = _to_float(
        risk.get(
            "max_trade_pct",
            caps.get(
                "per_trade", risk.get("risk_per_trade", bt.get("risk_per_trade", 0.10))
            ),
        ),
        0.10,
    )
    max_participation = _to_float(
        ex.get("max_participation", risk.get("max_participation", 0.10)), 0.10
    )

    max_open_positions_raw = _to_int(risk.get("max_open_positions", 0), 0)
    max_open_positions = max_open_positions_raw if max_open_positions_raw > 0 else None

    max_positions_per_symbol_raw = _to_int(risk.get("max_positions_per_symbol", 0), 0)
    max_positions_per_symbol = (
        max_positions_per_symbol_raw if max_positions_per_symbol_raw > 0 else None
    )

    exposure = RiskExposurePolicy(
        max_gross_exposure=_to_float(
            risk.get("max_gross_exposure", caps.get("max_gross", 2.0)), 2.0
        ),
        max_net_exposure=_to_float(
            risk.get("max_net_exposure", caps.get("max_net", 1.0)), 1.0
        ),
        max_per_name_pct=_to_float_opt(
            risk.get("max_per_name_pct", caps.get("per_name"))
        ),
        max_open_positions=max_open_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        strict=_to_bool(risk.get("strict"), False),
    )
    borrow = RiskBorrowPolicy(
        require_shortable_flag=_to_bool(risk.get("require_shortable_flag"), True),
        cap_by_availability=_to_bool(risk.get("cap_by_availability"), True),
    )
    short = ShortAvailabilityHeuristic(
        enabled=_to_bool(short_raw.get("enabled"), False),
        min_price=max(0.0, _to_float(short_raw.get("min_price", 0.0), 0.0)),
        min_adv_usd=max(0.0, _to_float(short_raw.get("min_adv_usd", 0.0), 0.0)),
        block_on_missing=_to_bool(short_raw.get("block_on_missing"), True),
    )
    sizing = RiskSizingPolicy(
        risk_per_trade=max(0.0, risk_per_trade),
        max_trade_pct=max(0.0, max_trade_pct),
        max_participation=max(0.0, max_participation),
    )
    return RiskPolicy(
        sizing=sizing,
        exposure=exposure,
        borrow=borrow,
        short_heuristic=short,
    )


def size_units_from_risk_budget(
    *,
    capital: float,
    risk_per_trade: float,
    per_unit_risk: float,
    min_units_if_positive: bool = False,
) -> int:
    if capital <= 0.0 or risk_per_trade <= 0.0 or per_unit_risk <= 0.0:
        return 0
    raw = float(capital) * float(risk_per_trade) / float(per_unit_risk)
    if not np.isfinite(raw) or raw <= 0.0:
        return 0
    units = int(math.floor(raw))
    if units <= 0 and min_units_if_positive:
        return 1
    return max(0, units)


def cap_units_by_trade_notional(
    *,
    units: int,
    capital: float,
    max_trade_pct: float,
    per_unit_notional: float,
    min_units_if_positive: bool = False,
) -> int:
    u = int(max(0, units))
    if u <= 0:
        return 0
    if max_trade_pct <= 0.0:
        return u
    if capital <= 0.0 or per_unit_notional <= 0.0:
        return 0
    cap_raw = float(capital) * float(max_trade_pct) / float(per_unit_notional)
    if not np.isfinite(cap_raw):
        return u
    cap_units = int(math.floor(cap_raw))
    if cap_units <= 0:
        return 1 if min_units_if_positive else 0
    return int(min(u, cap_units))


def cap_units_by_participation(
    *,
    units: int,
    max_participation: float,
    adv_sum_usd: float | None,
    per_unit_notional: float,
    require_gt_one_capacity: bool = False,
    min_units_if_positive: bool = False,
) -> int:
    u = int(max(0, units))
    if u <= 0:
        return 0
    if max_participation <= 0.0:
        return u
    if (
        adv_sum_usd is None
        or not np.isfinite(float(adv_sum_usd))
        or float(adv_sum_usd) <= 0.0
    ):
        return 0
    if per_unit_notional <= 0.0:
        return 0
    max_units = float(max_participation) * float(adv_sum_usd) / float(per_unit_notional)
    if not np.isfinite(max_units) or max_units <= 0.0:
        return 0
    if require_gt_one_capacity and max_units <= 1.0:
        return 0
    cap_units = int(math.floor(max_units))
    if cap_units <= 0:
        return 1 if min_units_if_positive else 0
    return int(min(u, cap_units))


def is_short_leg(*, signed_notional: float, units: float) -> bool:
    try:
        if float(signed_notional) < 0.0:
            return True
    except Exception:
        pass
    try:
        if float(units) < 0.0:
            return True
    except Exception:
        pass
    return False
