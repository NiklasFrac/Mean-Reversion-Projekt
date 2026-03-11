from __future__ import annotations

from typing import Any

from backtest.strat.baseline import BaselineZScoreStrategy

_STRATEGY_REGISTRY: dict[str, type] = {
    "baseline": BaselineZScoreStrategy,
    "BaselineZScoreStrategy": BaselineZScoreStrategy,
}


def list_strategy_keys() -> list[str]:
    """Return canonical registry keys (lowercase)."""
    return sorted({k for k in _STRATEGY_REGISTRY.keys() if k == k.lower()})


def get_strategy_cls(name: str | None) -> type:
    """Resolve a strategy class from the registry (case-insensitive)."""
    norm = str(name or "baseline").strip()
    key = norm.lower()
    cls = _STRATEGY_REGISTRY.get(key) or _STRATEGY_REGISTRY.get(norm)
    if cls is None:
        keys = list_strategy_keys()
        raise KeyError(f"Unknown strategy.name={norm!r}. Available: {keys}")
    return cls


def build_strategy_instance(
    cfg: dict[str, Any],
    *,
    borrow_ctx: Any,
    params: dict[str, Any] | None = None,
    name: str | None = None,
) -> Any:
    """Construct a strategy instance from cfg and optional explicit name/params."""
    scfg = cfg.get("strategy", {}) if isinstance(cfg.get("strategy"), dict) else {}
    resolved_name = name if name is not None else scfg.get("name")
    cls = get_strategy_cls(resolved_name)
    params_eff = (
        params
        if params is not None
        else (scfg.get("params", {}) if isinstance(scfg.get("params"), dict) else {})
    )
    try:
        return (
            cls(cfg, borrow_ctx=borrow_ctx, **params_eff)
            if params_eff
            else cls(cfg, borrow_ctx=borrow_ctx)
        )
    except TypeError as e:
        if params_eff:
            raise TypeError(
                f"Strategy {resolved_name!r} does not accept strategy.params={list(params_eff.keys())}"
            ) from e
        raise
