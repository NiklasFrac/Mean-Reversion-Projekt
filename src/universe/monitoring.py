from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from universe.coercion import cfg_bool, cfg_int

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _HAS_PROM = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_PROM = False

__all__ = [
    "logger",
    "setup_logging",
    "setup_prometheus",
    "prom_set_total",
    "prom_set_final",
    "prom_inc_failed",
    "stage_timer",
]

logger = logging.getLogger("runner_universe")

_PROM_ENABLED = False
_MET_TICKERS_TOTAL: Gauge | None = None
_MET_TICKERS_FINAL: Gauge | None = None
_MET_FUNDA_FAILED: Counter | None = None
_MET_STAGE_SEC: Histogram | None = None


def _prometheus_settings(cfg: dict[str, Any]) -> tuple[bool, int]:
    mon = cfg.get("monitoring", {}) or {}
    pcfg = mon.get("prometheus", {}) or {}
    if not isinstance(pcfg, dict):
        pcfg = {}
    enabled = cfg_bool(pcfg, "enabled", False)
    port = cfg_int(
        pcfg,
        "port",
        9108,
        min_value=1,
        logger=logger,
        section_name="monitoring.prometheus",
    )
    return enabled, port


def setup_logging(cfg: dict[str, Any]) -> logging.Logger:
    log_cfg = cfg.get("logging", {}) or {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    logger.handlers.clear()
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(ch)

    file_cfg = log_cfg.get("file", {})
    if (
        isinstance(file_cfg, dict)
        and cfg_bool(file_cfg, "enabled", False)
        and file_cfg.get("path")
    ):
        Path(file_cfg["path"]).parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            file_cfg["path"],
            maxBytes=cfg_int(
                file_cfg,
                "rotate_mb",
                50,
                min_value=1,
                logger=logger,
                section_name="logging.file",
            )
            * 1024
            * 1024,
            backupCount=cfg_int(
                file_cfg,
                "retention",
                10,
                min_value=1,
                logger=logger,
                section_name="logging.file",
            ),
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(fh)

    json_cfg = log_cfg.get("json", {})
    if (
        isinstance(json_cfg, dict)
        and cfg_bool(json_cfg, "enabled", False)
        and json_cfg.get("path")
    ):
        jp = Path(json_cfg["path"])
        jp.parent.mkdir(parents=True, exist_ok=True)
        jh = RotatingFileHandler(
            jp, maxBytes=100 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        jh.setLevel(level)
        jh.setFormatter(JsonFormatter())
        logger.addHandler(jh)

    logging.captureWarnings(True)
    try:
        for name in [
            "yfinance",
            "yfinance.shared",
            "yfinance.scrapers",
            "yfinance.ticker",
            "yfinance.multi",
        ]:
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
    except Exception:
        pass
    return logger


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _now_utc_iso(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_prometheus(cfg: dict[str, Any]) -> None:
    global \
        _PROM_ENABLED, \
        _MET_TICKERS_TOTAL, \
        _MET_TICKERS_FINAL, \
        _MET_FUNDA_FAILED, \
        _MET_STAGE_SEC
    if not _HAS_PROM or _PROM_ENABLED:
        return
    enabled, port = _prometheus_settings(cfg)
    if not enabled:
        return
    try:
        start_http_server(port)
        _MET_TICKERS_TOTAL = Gauge(
            "universe_tickers_total", "Total tickers before filtering"
        )
        _MET_TICKERS_FINAL = Gauge("universe_tickers_final", "Tickers after filtering")
        _MET_FUNDA_FAILED = Counter(
            "universe_funda_failed", "Failed fundamentals count"
        )
        _MET_STAGE_SEC = Histogram(
            "universe_stage_seconds",
            "Seconds per stage",
            labelnames=("stage",),
        )
        _PROM_ENABLED = True
        logger.info("Prometheus gestartet auf Port %d", port)
    except Exception as e:
        logger.warning("Failed to start Prometheus: %s", e)


def prom_set_total(n: int) -> None:
    if _PROM_ENABLED and _MET_TICKERS_TOTAL is not None:
        try:
            _MET_TICKERS_TOTAL.set(float(n))
        except Exception:
            pass


def prom_set_final(n: int) -> None:
    if _PROM_ENABLED and _MET_TICKERS_FINAL is not None:
        try:
            _MET_TICKERS_FINAL.set(float(n))
        except Exception:
            pass


def prom_inc_failed(n: int = 1) -> None:
    if _PROM_ENABLED and _MET_FUNDA_FAILED is not None:
        try:
            _MET_FUNDA_FAILED.inc(n)
        except Exception:
            pass


class stage_timer:
    def __init__(self, stage: str):
        self.stage = stage
        self.t0 = 0.0

    def __enter__(self) -> "stage_timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        dt_sec = time.perf_counter() - self.t0
        if _PROM_ENABLED and _MET_STAGE_SEC is not None:
            try:
                _MET_STAGE_SEC.labels(stage=self.stage).observe(dt_sec)
            except Exception:
                pass
        logger.info("[stage:%s] %.3fs", self.stage, dt_sec)
