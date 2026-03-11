from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class PerfResult(Generic[T]):
    """Ergebnis einer zeitlich gemessenen Funktionsausfuehrung."""

    value: T
    runtime_sec: float


def measure_runtime(fn: Callable[[], T]) -> PerfResult[T]:
    """Fuehrt ``fn`` aus und misst die Wall-Clock-Laufzeit in Sekunden."""
    start = time.perf_counter()
    value = fn()
    end = time.perf_counter()
    return PerfResult(value=value, runtime_sec=end - start)


__all__ = ["PerfResult", "measure_runtime"]
