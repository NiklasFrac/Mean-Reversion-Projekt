from __future__ import annotations

import math
from typing import Any, Mapping


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
    except Exception:
        return None
    return num if math.isfinite(num) else None


def is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "force"}
    return bool(value)


def coerce_bool(raw: Any, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    return is_truthy(raw)


def clamp01(value: Any, default: float = 0.0) -> float:
    parsed = safe_float(value)
    base = float(default) if parsed is None else float(parsed)
    return max(0.0, min(1.0, base))


def coerce_int(
    raw: Any,
    default: int,
    *,
    min_value: int | None = None,
    logger: Any | None = None,
    field_name: str = "value",
) -> int:
    try:
        val = int(raw)
    except Exception:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; using default %d.",
                field_name,
                raw,
                int(default),
            )
        val = int(default)
    if min_value is not None and val < min_value:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; clamping to %d.",
                field_name,
                raw,
                int(min_value),
            )
        val = int(min_value)
    return val


def coerce_float(
    raw: Any,
    default: float,
    *,
    min_value: float | None = None,
    strictly_positive: bool = False,
    logger: Any | None = None,
    field_name: str = "value",
) -> float:
    parsed = safe_float(raw)
    if parsed is None:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; using default %.6g.",
                field_name,
                raw,
                float(default),
            )
        return float(default)
    if strictly_positive and parsed <= 0:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; using default %.6g.",
                field_name,
                raw,
                float(default),
            )
        return float(default)
    if min_value is not None and parsed < min_value:
        if logger is not None:
            logger.warning(
                "Invalid %s=%r; clamping to %.6g.",
                field_name,
                raw,
                float(min_value),
            )
        return float(min_value)
    return float(parsed)


def cfg_int(
    section: Mapping[str, Any],
    key: str,
    default: int,
    *,
    min_value: int | None = None,
    logger: Any | None = None,
    section_name: str = "data",
) -> int:
    raw = section.get(key, default)
    return coerce_int(
        raw,
        default,
        min_value=min_value,
        logger=logger,
        field_name=f"{section_name}.{key}",
    )


def cfg_float(
    section: Mapping[str, Any],
    key: str,
    default: float,
    *,
    min_value: float | None = None,
    strictly_positive: bool = False,
    logger: Any | None = None,
    section_name: str = "data",
) -> float:
    raw = section.get(key, default)
    return coerce_float(
        raw,
        default,
        min_value=min_value,
        strictly_positive=strictly_positive,
        logger=logger,
        field_name=f"{section_name}.{key}",
    )


def cfg_bool(
    section: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    raw = section.get(key, default)
    return coerce_bool(raw, default=default)
