from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from universe import monitoring


class DummyMetric:
    def __init__(self) -> None:
        self.set_calls: list[float] = []
        self.inc_calls: list[int] = []
        self.label_kwargs: dict | None = None
        self.observe_calls: list[float] = []

    def set(self, value: float) -> None:
        self.set_calls.append(value)

    def inc(self, n: int = 1) -> None:
        self.inc_calls.append(n)

    def labels(self, **kwargs):
        self.label_kwargs = kwargs
        return self

    def observe(self, value: float) -> None:
        self.observe_calls.append(value)


def test_setup_prometheus_registers_metrics(monkeypatch):
    metrics: list[DummyMetric] = []

    def make_metric(*args, **kwargs):
        metric = DummyMetric()
        metrics.append(metric)
        return metric

    called_port = {}

    def fake_start_http_server(port):
        called_port["port"] = port

    monkeypatch.setattr(monitoring, "_HAS_PROM", True)
    monkeypatch.setattr(monitoring, "_PROM_ENABLED", False)
    monkeypatch.setattr(monitoring, "start_http_server", fake_start_http_server)
    monkeypatch.setattr(monitoring, "Gauge", make_metric)
    monkeypatch.setattr(monitoring, "Counter", make_metric)
    monkeypatch.setattr(monitoring, "Histogram", make_metric)

    monitoring.setup_prometheus(
        {"monitoring": {"prometheus": {"enabled": True, "port": 9999}}}
    )
    monitoring.prom_set_total(10)
    monitoring.prom_set_final(5)
    monitoring.prom_inc_failed(2)

    with monitoring.stage_timer("dummy"):
        pass

    assert called_port["port"] == 9999
    # Order: Gauge total, Gauge final, Counter failed, Histogram stage
    assert metrics[0].set_calls == [10.0]
    assert metrics[1].set_calls == [5.0]
    assert metrics[2].inc_calls == [2]
    assert metrics[3].label_kwargs == {"stage": "dummy"}
    assert metrics[3].observe_calls  # observed at least once


def test_setup_prometheus_parses_string_enabled_and_invalid_port(monkeypatch):
    called_port = {}

    def fake_start_http_server(port):
        called_port["port"] = port

    monkeypatch.setattr(monitoring, "_HAS_PROM", True)
    monkeypatch.setattr(monitoring, "_PROM_ENABLED", False)
    monkeypatch.setattr(monitoring, "start_http_server", fake_start_http_server)
    monkeypatch.setattr(monitoring, "Gauge", lambda *a, **k: DummyMetric())
    monkeypatch.setattr(monitoring, "Counter", lambda *a, **k: DummyMetric())
    monkeypatch.setattr(monitoring, "Histogram", lambda *a, **k: DummyMetric())

    monitoring.setup_prometheus(
        {"monitoring": {"prometheus": {"enabled": "true", "port": "bad"}}}
    )

    assert called_port["port"] == 9108


def test_setup_logging_coerces_rotation_and_retention(monkeypatch, tmp_path):
    monkeypatch.setattr(
        monitoring, "logger", logging.getLogger("runner_universe_test_monitoring")
    )
    log_path = tmp_path / "universe.log"

    log = monitoring.setup_logging(
        {
            "logging": {
                "level": "INFO",
                "file": {
                    "enabled": "true",
                    "path": str(log_path),
                    "rotate_mb": "bad",
                    "retention": "0",
                },
            }
        }
    )

    file_handlers = [h for h in log.handlers if isinstance(h, RotatingFileHandler)]
    assert file_handlers
    fh = file_handlers[0]
    assert fh.maxBytes == 50 * 1024 * 1024
    assert fh.backupCount == 1
