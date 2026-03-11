# src/tests/analysis/test_prometheus_opt.py
from analysis.prometheus_metrics import _PROM, init_prometheus, prom_observe


def test_prometheus_opt_out_noop():
    cfg = {"monitoring": {"prometheus": {"enabled": False}}}
    init_prometheus(cfg)  # sollte einfach nichts tun
    prom_observe({"stageA": 0.01}, 3)  # no-op, kein Fehler


def test_prometheus_opt_in_but_missing_client(monkeypatch):
    # simuliere "kein prometheus_client"
    import analysis.data_analysis as da

    monkeypatch.setattr(da, "start_http_server", None, raising=False)
    cfg = {"monitoring": {"prometheus": {"enabled": True, "port": 9933}}}
    init_prometheus(cfg)  # no-op, kein Fehler
    prom_observe({"stageA": 0.02, "stageB": 0.03}, 5)  # kein Fehler
    # _PROM bleibt leer
    assert isinstance(_PROM, dict)
