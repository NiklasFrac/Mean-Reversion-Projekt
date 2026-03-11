"""Optional Prometheus instrumentation."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from analysis.config_io import dict_hash_sha256
from analysis.logging_config import logger

_PROM_REG: Any = None
_PROM: dict[str, Any] = {}
_PROM_RUN_ID: Optional[str] = None


def init_prometheus(cfg: dict) -> None:
    mon = (cfg.get("monitoring") or {}).get("prometheus") or {}
    if not mon or not mon.get("enabled"):
        return
    try:  # lazy import
        from prometheus_client import (
            CollectorRegistry,
            Gauge,
            Histogram,
            PlatformCollector,
            ProcessCollector,
            start_http_server,
        )
    except Exception:
        return

    global _PROM_REG, _PROM, _PROM_RUN_ID
    _PROM_REG = CollectorRegistry()
    ProcessCollector(registry=_PROM_REG)
    PlatformCollector(registry=_PROM_REG)

    try:
        cfg_hash = dict_hash_sha256(cfg)
    except Exception:
        cfg_hash = hashlib.sha256(repr(cfg).encode("utf-8")).hexdigest()
    _PROM_RUN_ID = f"{cfg_hash[:8]}-{int(time.time())}"

    ns = str(mon.get("namespace", "analysis"))
    _PROM.clear()
    _PROM.update(
        {
            "runtime": Histogram(
                f"{ns}_runtime_seconds",
                "Total analysis runtime",
                registry=_PROM_REG,
                buckets=(1, 3, 5, 10, 30, 60, 120, 300, 600, 1200),
            ),
            "stage_seconds": Histogram(
                f"{ns}_stage_seconds",
                "Stage duration seconds",
                ["stage"],
                registry=_PROM_REG,
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 30, 60),
            ),
            "n_selected": Gauge(
                f"{ns}_n_selected",
                "Selected pairs",
                registry=_PROM_REG,
            ),
            "runtime_by_run": Histogram(
                f"{ns}_runtime_seconds_by_run",
                "Total analysis runtime (by run)",
                ["run_id"],
                registry=_PROM_REG,
                buckets=(1, 3, 5, 10, 30, 60, 120, 300, 600, 1200),
            ),
            "stage_seconds_by_run": Histogram(
                f"{ns}_stage_seconds_by_run",
                "Stage duration seconds (by run)",
                ["stage", "run_id"],
                registry=_PROM_REG,
                buckets=(0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 30, 60),
            ),
            "n_selected_by_run": Gauge(
                f"{ns}_n_selected_by_run",
                "Selected pairs (by run)",
                ["run_id"],
                registry=_PROM_REG,
            ),
        }
    )

    port = int(mon.get("port", 9108))
    try:
        start_http_server(port, registry=_PROM_REG)
        logger.info("Prometheus metrics server on :%d (run_id=%s)", port, _PROM_RUN_ID)
    except OSError as e:
        logger.warning("Prometheus disabled (port %d busy: %s)", port, e)
        _PROM.clear()
        _PROM_REG = None
        _PROM_RUN_ID = None


def prom_observe(timings: dict[str, float], n_selected: int) -> None:
    if not _PROM:
        return
    try:
        total = sum(float(v) for v in timings.values())
        if "runtime" in _PROM:
            _PROM["runtime"].observe(total)
        if "n_selected" in _PROM:
            _PROM["n_selected"].set(float(n_selected))
        if "stage_seconds" in _PROM:
            for k, v in timings.items():
                _PROM["stage_seconds"].labels(stage=str(k)).observe(float(v))
        if _PROM_RUN_ID is not None:
            if "runtime_by_run" in _PROM:
                _PROM["runtime_by_run"].labels(run_id=_PROM_RUN_ID).observe(total)
            if "n_selected_by_run" in _PROM:
                _PROM["n_selected_by_run"].labels(run_id=_PROM_RUN_ID).set(
                    float(n_selected)
                )
            if "stage_seconds_by_run" in _PROM:
                for k, v in timings.items():
                    _PROM["stage_seconds_by_run"].labels(
                        stage=str(k), run_id=_PROM_RUN_ID
                    ).observe(float(v))
    except Exception:
        logger.debug("Prometheus observe failed", exc_info=True)


__all__ = ["_PROM_REG", "_PROM", "_PROM_RUN_ID", "init_prometheus", "prom_observe"]
