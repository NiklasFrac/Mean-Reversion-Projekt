import time

from backtest.reporting import perf_timer


def test_measure_runtime_returns_value_and_runtime() -> None:
    res = perf_timer.measure_runtime(lambda: 7)
    assert res.value == 7
    assert res.runtime_sec >= 0.0


def test_measure_runtime_captures_sleep() -> None:
    res = perf_timer.measure_runtime(lambda: time.sleep(0.001))
    assert res.value is None
    assert res.runtime_sec > 0.0
