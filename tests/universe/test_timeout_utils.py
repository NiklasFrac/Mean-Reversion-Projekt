from __future__ import annotations

import threading
import time

import pytest

import universe.timeout_utils as timeout_utils
from universe.timeout_utils import run_with_timeout


def test_run_with_timeout_returns_value() -> None:
    assert run_with_timeout(lambda: 7, 0.1) == 7


def test_run_with_timeout_times_out() -> None:
    def _sleep() -> int:
        time.sleep(0.05)
        return 1

    with pytest.raises(TimeoutError):
        run_with_timeout(_sleep, 0.01, err_prefix="test")


def test_run_with_timeout_propagates_base_exception() -> None:
    def _raise_system_exit() -> None:
        raise SystemExit(3)

    with pytest.raises(SystemExit):
        run_with_timeout(_raise_system_exit, 0.1)


def test_run_with_timeout_uses_bounded_worker_pool() -> None:
    def _slow() -> int:
        time.sleep(0.2)
        return 1

    # Trigger many short timeouts; this should not spawn unbounded threads.
    for _ in range(timeout_utils._TIMEOUT_MAX_WORKERS * 3):
        with pytest.raises(TimeoutError):
            run_with_timeout(_slow, 0.005, err_prefix="pool")

    workers = [
        t for t in threading.enumerate() if str(t.name).startswith("universe-timeout")
    ]
    assert len(workers) <= timeout_utils._TIMEOUT_MAX_WORKERS
