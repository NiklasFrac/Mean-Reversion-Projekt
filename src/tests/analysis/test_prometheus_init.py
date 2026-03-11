# src/tests/analysis/test_prometheus_init.py
import analysis.data_analysis as da


def test_init_prometheus_enabled(monkeypatch):
    calls = {"server": 0}

    def fake_server(*a, **k):
        calls["server"] += 1

    # Patch beide möglichen Targets
    if hasattr(da, "start_http_server"):
        monkeypatch.setattr(da, "start_http_server", fake_server, raising=False)
    try:
        import prometheus_client as pc

        monkeypatch.setattr(pc, "start_http_server", fake_server, raising=False)
    except Exception:
        pass

    class FakeReg: ...

    class FakeGauge:
        def __init__(self, *a, **k): ...
        def set(self, v): ...

    class FakeHist:
        def __init__(self, *a, **k): ...
        def observe(self, v): ...
        def labels(self, **k):
            return self

    monkeypatch.setattr(da, "CollectorRegistry", FakeReg, raising=False)
    monkeypatch.setattr(da, "ProcessCollector", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(da, "PlatformCollector", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(da, "Gauge", FakeGauge, raising=False)
    monkeypatch.setattr(da, "Histogram", FakeHist, raising=False)

    cfg = {
        "monitoring": {
            "prometheus": {"enabled": True, "namespace": "test", "port": 9999}
        }
    }
    da.init_prometheus(cfg)
    assert calls["server"] == 1
