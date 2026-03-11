from types import SimpleNamespace

import analysis.prometheus_metrics as pm


def test_prometheus_start_failure(monkeypatch):
    # Inject stub prometheus_client module that raises on start_http_server.
    fake_mod = SimpleNamespace(
        CollectorRegistry=lambda: "registry",
        Gauge=lambda *args, **kwargs: None,
        Histogram=lambda *args, **kwargs: None,
        PlatformCollector=lambda **kwargs: None,
        ProcessCollector=lambda **kwargs: None,
        start_http_server=lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("port busy")
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "prometheus_client", fake_mod)

    cfg = {
        "monitoring": {
            "prometheus": {"enabled": True, "namespace": "test", "port": 9999}
        }
    }
    pm.init_prometheus(cfg)
    # On port failure, registry should be cleared/disabled.
    assert pm._PROM == {}
    assert pm._PROM_REG is None
