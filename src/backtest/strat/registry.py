from __future__ import annotations

from typing import Any, cast

from backtest.config.cfg import AppConfig, config_to_dict
from backtest.strat.baseline import BaselineZScoreStrategy

_STRATEGY_TYPES: dict[str, type[BaselineZScoreStrategy]] = {
    "baseline": BaselineZScoreStrategy
}


def _resolve_strategy_cls(name: str | None) -> type[BaselineZScoreStrategy]:
    key = str(name or "baseline").strip().lower()
    cls = _STRATEGY_TYPES.get(key)
    if cls is None:
        raise KeyError(
            f"Unknown strategy.name={name!r}. Available: {sorted(_STRATEGY_TYPES)}"
        )
    return cls


def build_strategy_from_cfg(cfg: dict[str, Any] | AppConfig) -> Any:
    cfg_dict = config_to_dict(cfg) if isinstance(cfg, AppConfig) else cfg
    scfg = (
        cfg_dict.get("strategy", {})
        if isinstance(cfg_dict.get("strategy"), dict)
        else {}
    )
    name = scfg.get("name")
    params = scfg.get("params", {}) if isinstance(scfg.get("params"), dict) else {}
    cls = _resolve_strategy_cls(cast(str | None, name))
    try:
        return cls(cfg_dict, **params) if params else cls(cfg_dict)
    except TypeError as e:
        if params:
            raise TypeError(
                f"Strategy {name!r} does not accept strategy.params={list(params.keys())}"
            ) from e
        raise
