from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading
from typing import Any, Callable, TypeVar

from universe.coercion import safe_float

T = TypeVar("T")

# Reuse a bounded worker pool for timeout-wrapped calls.
# This avoids spawning an unbounded number of daemon threads when repeated calls time out.
_TIMEOUT_MAX_WORKERS = 16
_TIMEOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_TIMEOUT_MAX_WORKERS,
    thread_name_prefix="universe-timeout",
)
_TIMEOUT_SLOTS = threading.BoundedSemaphore(value=_TIMEOUT_MAX_WORKERS)


def run_with_timeout(
    func: Callable[[], T],
    timeout: float | None,
    *,
    err_prefix: str = "Call",
) -> T:
    timeout_val = safe_float(timeout)
    if timeout_val is None or timeout_val <= 0:
        return func()

    acquired = _TIMEOUT_SLOTS.acquire(timeout=timeout_val)
    if not acquired:
        raise TimeoutError(f"{err_prefix} timed out waiting for timeout-worker slot.")

    try:
        future = _TIMEOUT_EXECUTOR.submit(func)
    except BaseException:
        _TIMEOUT_SLOTS.release()
        raise

    def _release_slot(_future: Any) -> None:
        try:
            _TIMEOUT_SLOTS.release()
        except Exception:
            pass

    future.add_done_callback(_release_slot)
    try:
        return future.result(timeout=timeout_val)
    except FuturesTimeoutError:
        future.cancel()
        raise TimeoutError(f"{err_prefix} timed out after {timeout_val:.3f}s")
