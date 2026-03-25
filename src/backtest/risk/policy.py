from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = [
    "RiskExposureSnapshot",
    "RiskBorrowPolicy",
    "RiskExposurePolicy",
    "RiskPolicy",
    "RiskSizingPolicy",
    "ShortAvailabilityHeuristic",
    "build_risk_policy",
    "check_pair_admission",
    "cap_units_by_participation",
    "cap_units_by_trade_notional",
    "is_short_leg",
    "short_availability_reason",
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


@dataclass(frozen=True)
class RiskExposureSnapshot:
    open_positions: int
    gross_exposure: float
    net_exposure: float
    per_name_gross: Mapping[str, float]
    positions_per_symbol: Mapping[str, int]


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


def short_availability_reason(
    *,
    heuristic: ShortAvailabilityHeuristic,
    leg_symbols: tuple[str | None, str | None],
    leg_notionals: tuple[float, float],
    leg_units: tuple[float, float],
    leg_entry_prices: tuple[float | None, float | None],
    leg_adv_usd: tuple[float | None, float | None],
    block_on_missing: bool = True,
) -> str:
    if not heuristic.enabled:
        return ""
    needs_price = float(heuristic.min_price) > 0.0
    needs_adv = float(heuristic.min_adv_usd) > 0.0

    for sym, notional, units, price, adv in zip(
        leg_symbols,
        leg_notionals,
        leg_units,
        leg_entry_prices,
        leg_adv_usd,
        strict=False,
    ):
        if not is_short_leg(signed_notional=float(notional), units=float(units)):
            continue
        sym_s = (str(sym).strip().upper() if sym is not None else "") or "UNKNOWN"
        price_val = _to_float_opt(price)
        adv_val = _to_float_opt(adv)
        if needs_price and price_val is None:
            if block_on_missing:
                return f"short_price_missing:{sym_s}"
        elif needs_price and float(price_val) < float(heuristic.min_price):
            return f"short_price:{sym_s}"

        if needs_adv and adv_val is None:
            if block_on_missing and bool(heuristic.block_on_missing):
                return f"short_adv_missing:{sym_s}"
        elif needs_adv and float(adv_val) < float(heuristic.min_adv_usd):
            return f"short_adv:{sym_s}"

    return ""


def check_pair_admission(
    *,
    policy: RiskPolicy,
    capital: float,
    snapshot: RiskExposureSnapshot,
    leg_symbols: tuple[str | None, str | None],
    leg_notionals: tuple[float, float],
) -> bool:
    exposure = policy.exposure
    sizing = policy.sizing
    cap = max(1e-9, float(capital))
    y_sym = (str(leg_symbols[0]).strip().upper() if leg_symbols[0] is not None else "") or None
    x_sym = (str(leg_symbols[1]).strip().upper() if leg_symbols[1] is not None else "") or None
    ny = _to_float(leg_notionals[0], 0.0)
    nx = _to_float(leg_notionals[1], 0.0)
    gross_add = abs(ny) + abs(nx)
    if exposure.strict and (gross_add <= 0.0 or not math.isfinite(gross_add)):
        return False
    if gross_add <= 0.0:
        return True

    if (
        exposure.max_open_positions is not None
        and int(snapshot.open_positions) >= int(exposure.max_open_positions)
    ):
        return False

    if exposure.max_positions_per_symbol is not None:
        counts = snapshot.positions_per_symbol
        seen: set[str] = set()
        for sym in (y_sym, x_sym):
            if sym is None or sym in seen:
                continue
            if int(counts.get(sym, 0)) + 1 > int(exposure.max_positions_per_symbol):
                return False
            seen.add(sym)

    if (
        float(sizing.max_trade_pct) > 0.0
        and gross_add > float(sizing.max_trade_pct) * cap
    ):
        return False

    if float(snapshot.gross_exposure) + gross_add > float(exposure.max_gross_exposure) * cap:
        return False

    net_after = float(snapshot.net_exposure) + ny + nx
    if abs(net_after) > float(exposure.max_net_exposure) * cap:
        return False

    if exposure.max_per_name_pct is not None:
        per_name = snapshot.per_name_gross
        if (
            y_sym is not None
            and float(per_name.get(y_sym, 0.0) + abs(ny))
            > float(exposure.max_per_name_pct) * cap
        ):
            return False
        if (
            x_sym is not None
            and float(per_name.get(x_sym, 0.0) + abs(nx))
            > float(exposure.max_per_name_pct) * cap
        ):
            return False

    return True
