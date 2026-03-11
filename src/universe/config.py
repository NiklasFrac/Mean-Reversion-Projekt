from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

import yaml
from universe.artifact_defaults import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_TICKERS_CSV,
    apply_artifact_path_defaults,
)
from universe.symbol_filter_defaults import (
    DEFAULT_DROP_CONTAINS,
    DEFAULT_DROP_PREFIXES,
    DEFAULT_DROP_REGEX,
    DEFAULT_DROP_SUFFIXES,
)

__all__ = [
    "RunnerConfig",
    "load_cfg_or_default",
    "validate_cfg",
    "ensure_runner_config",
]

_HAS_PYDANTIC = False
BaseModel: Any
ValidationError: type[Exception]
try:
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import (
        Field,
        ValidationError as _PydanticValidationError,
        field_validator,
    )

    validator = field_validator
    BaseModel = _PydanticBaseModel
    ValidationError = _PydanticValidationError
    _HAS_PYDANTIC = True
except Exception:  # pragma: no cover - optional dependency
    BaseModel = object
    ValidationError = ValueError

    def validator(*args: Any, **kwargs: Any):
        def _inner(fn: Any) -> Any:
            return fn

        return _inner


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return {}


def load_cfg_or_default(path: Path | str | None) -> dict[str, Any]:
    """Loads YAML config or returns a minimal default structure."""
    if path:
        p = Path(path)
        if p.exists():
            return _load_yaml(p)
    return {
        "universe": {
            "output_tickers_csv": DEFAULT_OUTPUT_TICKERS_CSV,
            "manifest": DEFAULT_MANIFEST_PATH,
        },
        "filters": {},
        "runtime": {},
    }


def _with_default_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    val = raw.get(key, {})
    if not isinstance(val, dict):
        val = {}
    copy = dict(val)
    raw[key] = copy
    return copy


if _HAS_PYDANTIC:  # pragma: no cover - exercised implicitly via validate_cfg

    class FiltersCfg(BaseModel):
        drop_na: bool = True
        drop_zero: bool = True
        drop_prefixes: list[str] = list(DEFAULT_DROP_PREFIXES)
        drop_suffixes: list[str] = list(DEFAULT_DROP_SUFFIXES)
        drop_regex: list[str] = list(DEFAULT_DROP_REGEX)
        drop_contains: list[str] = list(DEFAULT_DROP_CONTAINS)

        min_price: float | None = None
        max_price: float | None = None
        min_market_cap: float | None = None
        max_market_cap: float | None = None
        min_avg_volume: float | None = None
        max_avg_volume: float | None = None
        min_dollar_adv: float | None = None

        min_float_pct: float | None = None
        treat_missing_float_as_pass: bool = False
        min_free_float_shares: float | None = None
        min_free_float_dollar_cap: float | None = None

        require_dividend: bool = False
        symbol_whitelist: list[str] = []
        symbol_blacklist: list[str] = []

        @validator("min_float_pct")
        @classmethod
        def _pct(cls, value: float | None) -> float | None:
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError("min_float_pct must be within [0, 1]")
            return value

    class UniverseCfg(BaseModel):
        output_tickers_csv: str
        manifest: str
        fundamentals_out: str | None = None
        adv_cache: str | None = None
        screener_glob: str | None = "runs/data/nasdaq_screener_*.csv"
        screener_selection_mode: Literal["error_if_ambiguous", "latest_mtime"] = (
            "error_if_ambiguous"
        )

    class RuntimeCfg(BaseModel):
        progress_bar: bool = True
        workers: int = 16
        fundamentals_heartbeat_logging: bool = True
        request_timeout: int = 20
        request_retries: int = 3
        request_backoff: float = 1.7
        fundamentals_postfill_mode: Literal["drop", "fill"] = "drop"
        # None means "derive from workers" in builder (legacy behavior).
        max_inflight_requests: int | None = None
        force_rebuild: bool = False
        reuse_exchange_seed: bool = True
        allow_cached_seed_without_screener: bool = False
        fail_fast: dict[str, Any] = {"enabled": False, "max_consecutive_failures": 50}
        checkpoint_path: str | None = DEFAULT_CHECKPOINT_PATH
        persist_run_scoped_outputs: bool = False
        use_hashed_artifacts: bool = True

    class RootCfgModel(BaseModel):
        universe: UniverseCfg
        filters: FiltersCfg
        runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
        data: dict[str, Any] = Field(default_factory=dict)
        logging: dict[str, Any] = Field(default_factory=dict)
        monitoring: dict[str, Any] = Field(default_factory=dict)
        vendor: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RunnerConfig(Mapping[str, Any]):
    """Validated configuration exposing typed helpers while remaining Mapping-like."""

    _data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    @property
    def universe(self) -> dict[str, Any]:
        return dict(self._data.get("universe", {}) or {})

    @property
    def filters(self) -> dict[str, Any]:
        return dict(self._data.get("filters", {}) or {})

    @property
    def runtime(self) -> dict[str, Any]:
        return dict(self._data.get("runtime", {}) or {})

    @property
    def data(self) -> dict[str, Any]:
        return dict(self._data.get("data", {}) or {})

    @property
    def logging(self) -> dict[str, Any]:
        return dict(self._data.get("logging", {}) or {})

    @property
    def monitoring(self) -> dict[str, Any]:
        return dict(self._data.get("monitoring", {}) or {})

    @property
    def vendor(self) -> dict[str, Any]:
        return dict(self._data.get("vendor", {}) or {})


def _drop_deprecated_knobs(cfg: dict[str, Any]) -> None:
    runtime = cfg.get("runtime")
    if isinstance(runtime, dict):
        for key in (
            "parallel_exchange_load",
            "rate_limit_per_sec",
            "rng_seed",
            "skip_if_fresh_hours",
            "adv_executor",
            "adv_workers",
            "profile",
            "persist_snapshots",
            "snapshots_dir",
        ):
            runtime.pop(key, None)

    universe = cfg.get("universe")
    if isinstance(universe, dict):
        for key in ("metadata_cache", "default_name", "adv_path"):
            universe.pop(key, None)

    data = cfg.get("data")
    if isinstance(data, dict):
        for key in ("fundamentals_cache_dir", "fundamentals_use_major_holders"):
            data.pop(key, None)

    vendor = cfg.get("vendor")
    if isinstance(vendor, dict):
        vendor.pop("cache_root", None)

    monitoring = cfg.get("monitoring")
    if isinstance(monitoring, dict):
        for key in ("enabled", "heartbeat_sec"):
            monitoring.pop(key, None)
        prom = monitoring.get("prometheus")
        if isinstance(prom, dict):
            for key in ("namespace", "keep_alive_sec"):
                prom.pop(key, None)

    cfg.pop("mlflow", None)


def validate_cfg(raw: dict[str, Any]) -> RunnerConfig:
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/dict.")

    cfg = dict(raw)

    universe = _with_default_mapping(cfg, "universe")
    if "output_tickers_csv" not in universe or "manifest" not in universe:
        raise ValueError(
            "YAML: universe.output_tickers_csv and universe.manifest are required."
        )
    universe.setdefault("screener_glob", "runs/data/nasdaq_screener_*.csv")
    universe.setdefault("screener_selection_mode", "error_if_ambiguous")

    filters = _with_default_mapping(cfg, "filters")
    filters.setdefault("symbol_whitelist", [])
    filters.setdefault("symbol_blacklist", [])
    filters.setdefault("drop_prefixes", list(DEFAULT_DROP_PREFIXES))
    filters.setdefault("drop_suffixes", list(DEFAULT_DROP_SUFFIXES))
    filters.setdefault("drop_regex", list(DEFAULT_DROP_REGEX))
    filters.setdefault("drop_contains", list(DEFAULT_DROP_CONTAINS))
    filters.setdefault("drop_na", True)
    filters.setdefault("drop_zero", True)
    filters.setdefault("min_dollar_adv", None)
    filters.setdefault("min_avg_volume", None)
    filters.setdefault("treat_missing_float_as_pass", False)
    filters.setdefault("min_free_float_shares", None)
    filters.setdefault("min_free_float_dollar_cap", None)

    data = _with_default_mapping(cfg, "data")
    # Separate cache for warmup (pre-filter) unadjusted prices to avoid being overwritten by filtered runs.
    data.setdefault("adjust_dividends", True)
    data.setdefault("strict_unadjusted_validation", False)
    data.setdefault("adv_cache_ttl_days", 30.0)
    data.setdefault("adv_cache_min_coverage_ratio", 0.8)
    data.setdefault("adv_min_valid_ratio", 0.7)

    runtime = _with_default_mapping(cfg, "runtime")
    runtime.setdefault("use_hashed_artifacts", True)
    runtime.setdefault("persist_run_scoped_outputs", False)
    runtime.setdefault("reuse_exchange_seed", True)
    runtime.setdefault("allow_cached_seed_without_screener", False)
    runtime.setdefault("fundamentals_postfill_mode", "drop")
    runtime.setdefault("fundamentals_heartbeat_logging", True)
    apply_artifact_path_defaults(
        universe_cfg=universe,
        data_cfg=data,
        runtime_cfg=runtime,
    )

    cfg.setdefault("logging", {})
    cfg.setdefault("monitoring", {})
    cfg.setdefault("vendor", {})
    _drop_deprecated_knobs(cfg)

    if _HAS_PYDANTIC:
        try:
            model = RootCfgModel(**cfg)
            fixed = model.model_dump()
            # Merge user extras back to keep custom keys (e.g., runtime.canary)
            for section in cfg:
                if isinstance(cfg[section], dict):
                    merged = dict(cfg[section])
                    merged.update(fixed.get(section, {}))
                    cfg[section] = merged
                else:
                    cfg[section] = fixed.get(section, cfg[section])
        except ValidationError as exc:
            # Fail fast: invalid config must not silently continue with partial defaults.
            raise ValueError(f"YAML validation failed: {exc}") from exc

    return RunnerConfig(cfg)


def ensure_runner_config(obj: RunnerConfig | dict[str, Any]) -> RunnerConfig:
    if isinstance(obj, RunnerConfig):
        return obj
    return validate_cfg(obj)
