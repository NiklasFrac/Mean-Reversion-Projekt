from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from backtest.strat.registry import build_strategy_instance, list_strategy_keys

_LOB_LEGACY_KEYS = (
    "book",
    "step",
    "orders",
    "policy",
    "fees",
    "impact",
    "diagnostics",
    "pricing",
)
_LOB_DEAD_POST_COST_KEYS = (
    "borrow_bps",
    "borrow_day_basis",
    "borrow_short_only",
    "report_slippage_impact",
)


def _build_strategy(cfg_eff: dict[str, Any], *, borrow_ctx: Any) -> Any:
    """
    Construct a strategy instance from cfg.

    Config schema:
      strategy:
        name: "baseline"   # registry key (default: baseline)
        params: {}      # optional extra ctor kwargs
    """
    scfg = (
        cfg_eff.get("strategy", {}) if isinstance(cfg_eff.get("strategy"), dict) else {}
    )
    name = str(scfg.get("name") or "baseline").strip()
    params = scfg.get("params", {}) if isinstance(scfg.get("params"), dict) else {}
    try:
        return build_strategy_instance(
            cfg_eff, borrow_ctx=borrow_ctx, params=params, name=name
        )
    except KeyError:
        keys = list_strategy_keys()
        raise KeyError(f"Unknown strategy.name={name!r}. Available: {keys}") from None


def _validate_cfg_strict(cfg: Mapping[str, Any]) -> None:
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a dict-like mapping")

    if "reports" in cfg:
        raise ValueError(
            "Legacy config key 'reports' is no longer supported. Use 'reporting'."
        )

    bt = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    wf = bt.get("walkforward", {}) if isinstance(bt.get("walkforward"), dict) else {}
    legacy_keys: list[str] = []
    if "window_end_policy" in bt:
        legacy_keys.append("backtest.window_end_policy")
    if "carry_open_trades" in wf:
        legacy_keys.append("backtest.walkforward.carry_open_trades")
    if "stateful_sizing" in wf:
        legacy_keys.append("backtest.walkforward.stateful_sizing")
    sig = cfg.get("signal", {}) if isinstance(cfg.get("signal"), dict) else {}
    if "execution_lag_bars" in sig:
        legacy_keys.append("signal.execution_lag_bars")
    bo = cfg.get("bo", {}) if isinstance(cfg.get("bo"), dict) else {}
    for key in (
        "write_reference",
        "reference_filename",
        "copy_summary",
        "summary_filename",
    ):
        if key in bo:
            legacy_keys.append(f"bo.{key}")
    for key in (
        "cv",
        "rescore",
        "stage1",
        "stage2",
        "z_window_range",
        "precompute_beta_z",
    ):
        if key in bo:
            legacy_keys.append(f"bo.{key}")
    if legacy_keys:
        raise ValueError(
            "Legacy config keys are no longer supported: " + ", ".join(legacy_keys)
        )

    ex = cfg.get("execution", {}) if isinstance(cfg.get("execution"), dict) else {}
    if "exec_lob" in ex:
        legacy_keys.append("execution.exec_lob")
    if "override_pnl" in ex:
        legacy_keys.append("execution.override_pnl")
    lob = ex.get("lob", {}) if isinstance(ex.get("lob"), dict) else {}
    for key in _LOB_LEGACY_KEYS:
        if key in lob:
            legacy_keys.append(f"execution.lob.{key}")
    post_costs = (
        lob.get("post_costs", {}) if isinstance(lob.get("post_costs"), dict) else {}
    )
    for key in _LOB_DEAD_POST_COST_KEYS:
        if key in post_costs:
            legacy_keys.append(f"execution.lob.post_costs.{key}")
    if legacy_keys:
        raise ValueError(
            "Legacy config keys are no longer supported: "
            + ", ".join(legacy_keys)
            + ". LOB now always uses execution-authoritative PnL, so execution.override_pnl was removed."
        )
    mode = str(ex.get("mode", "lob")).strip().lower()
    if mode not in {"lob", "light"}:
        raise ValueError(
            f"execution.mode must be one of ('lob', 'light') (got {mode!r})"
        )

    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    prices_path = data.get("prices_path")
    pairs_path = data.get("pairs_path")
    if not prices_path:
        raise KeyError("Config missing data.prices_path")
    if not pairs_path:
        raise KeyError("Config missing data.pairs_path")
    if not Path(str(prices_path)).exists():
        raise FileNotFoundError(f"data.prices_path not found: {prices_path}")
    if not Path(str(pairs_path)).exists():
        raise FileNotFoundError(f"data.pairs_path not found: {pairs_path}")
