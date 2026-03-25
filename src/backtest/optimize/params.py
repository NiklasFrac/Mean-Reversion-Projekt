from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import numpy as np

from backtest.config.cfg import AppConfig
from backtest.config.types import BOConfig, BOCVConfig

BOMode = Literal["fast", "realistic"]
BOMetric = Literal["sharpe", "cagr", "calmar"]

_BO_RUNTIME_KEYS = ("init_points", "n_iter", "patience")
_BO_KEY_FIELDS = (
    "enabled",
    "mode",
    "entry_z_range",
    "exit_z_range",
    "stop_z_range",
    "min_revert_prob_range",
    "horizon_days_range",
    "fast",
    "realistic",
)
_VALID_MODES: tuple[BOMode, ...] = ("fast", "realistic")


@dataclass(frozen=True)
class BOBudget:
    init_points: int
    n_iter: int
    patience: int


@dataclass(frozen=True)
class BOResolvedConfig:
    enabled: bool
    mode: BOMode
    out_dir: Path
    cv: BOCVConfig
    realistic_metric: BOMetric
    signal_budget: BOBudget
    markov_budget: BOBudget


@dataclass(frozen=True)
class SignalParams:
    entry_z: float
    exit_z: float
    stop_z: float

    def as_dict(self) -> dict[str, float]:
        return {
            "entry_z": float(self.entry_z),
            "exit_z": float(self.exit_z),
            "stop_z": float(self.stop_z),
        }


@dataclass(frozen=True)
class MarkovParams:
    min_revert_prob: float
    horizon_days: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_revert_prob": float(self.min_revert_prob),
            "horizon_days": int(self.horizon_days),
        }


@dataclass(frozen=True)
class SignalSearchSpace:
    entry_z: tuple[float, float]
    exit_z: tuple[float, float]
    stop_z: tuple[float, float]


@dataclass(frozen=True)
class MarkovDefaults:
    enabled: bool
    min_revert_prob: float
    horizon_days: int


@dataclass(frozen=True)
class MarkovSearchSpace:
    min_revert_prob: tuple[float, float]
    horizon_days: tuple[float, float]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def clean_bo_cfg_for_key(bo_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bo_cfg, Mapping):
        return {}
    out = {key: bo_cfg[key] for key in _BO_KEY_FIELDS if key in bo_cfg}
    for key in _BO_RUNTIME_KEYS:
        out.pop(key, None)
    return out


def _bo_config(cfg: AppConfig) -> BOConfig:
    return cfg.bo


def resolve_bo_config(cfg: AppConfig) -> BOResolvedConfig:
    bo = _bo_config(cfg)
    cv = bo.fast.cv if bo.mode == "fast" else bo.realistic.cv
    signal_budget = BOBudget(
        init_points=int(bo.init_points),
        n_iter=int(bo.n_iter),
        patience=int(bo.patience),
    )
    markov_budget = BOBudget(
        init_points=int(
            bo.markov_init_points
            if bo.markov_init_points is not None
            else signal_budget.init_points
        ),
        n_iter=int(
            bo.markov_n_iter if bo.markov_n_iter is not None else signal_budget.n_iter
        ),
        patience=int(
            bo.markov_patience
            if bo.markov_patience is not None
            else signal_budget.patience
        ),
    )
    return BOResolvedConfig(
        enabled=bool(bo.enabled),
        mode=cast(BOMode, bo.mode),
        out_dir=Path(bo.out_dir),
        cv=cv,
        realistic_metric=cast(BOMetric, bo.realistic.metric),
        signal_budget=signal_budget,
        markov_budget=markov_budget,
    )


def resolve_bo_mode(cfg: AppConfig) -> BOMode:
    return resolve_bo_config(cfg).mode


def resolve_bo_cv(cfg: AppConfig, *, mode: BOMode | None = None) -> BOCVConfig:
    bo = _bo_config(cfg)
    mode_eff = cast(BOMode, mode or bo.mode)
    if mode_eff not in _VALID_MODES:
        keys = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"Unsupported bo.mode={mode_eff!r}. Expected one of: {keys}.")
    return bo.fast.cv if mode_eff == "fast" else bo.realistic.cv


def resolve_realistic_metric(cfg: AppConfig) -> BOMetric:
    return resolve_bo_config(cfg).realistic_metric


def signal_search_space(
    cfg: AppConfig,
    *,
    entry0: float,
    exit0: float,
    stop0: float,
) -> SignalSearchSpace:
    bo = _bo_config(cfg)
    entry_range = bo.entry_z_range or (float(entry0), float(entry0))
    exit_range = bo.exit_z_range or (float(exit0), float(exit0))
    stop_range = bo.stop_z_range or (float(stop0), float(stop0))
    return SignalSearchSpace(
        entry_z=(float(entry_range[0]), float(entry_range[1])),
        exit_z=(float(exit_range[0]), float(exit_range[1])),
        stop_z=(float(stop_range[0]), float(stop_range[1])),
    )


def markov_bo_defaults(cfg: AppConfig) -> MarkovDefaults:
    raw = cfg.markov_filter
    return MarkovDefaults(
        enabled=bool(raw.enabled),
        min_revert_prob=float(np.clip(raw.min_revert_prob, 0.0, 1.0)),
        horizon_days=max(1, int(raw.horizon_days)),
    )


def markov_search_space(cfg: AppConfig, defaults: MarkovDefaults) -> MarkovSearchSpace:
    bo = _bo_config(cfg)
    p_bounds = bo.min_revert_prob_range or (
        float(defaults.min_revert_prob),
        float(defaults.min_revert_prob),
    )
    h_bounds = bo.horizon_days_range or (
        int(defaults.horizon_days),
        int(defaults.horizon_days),
    )
    p_lo = float(np.clip(min(p_bounds[0], p_bounds[1]), 0.0, 1.0))
    p_hi = float(np.clip(max(p_bounds[0], p_bounds[1]), 0.0, 1.0))
    h_lo = float(max(1.0, min(h_bounds[0], h_bounds[1])))
    h_hi = float(max(h_lo, max(h_bounds[0], h_bounds[1])))
    return MarkovSearchSpace(
        min_revert_prob=(p_lo, p_hi),
        horizon_days=(h_lo, h_hi),
    )


def cfg_with_markov_overrides(
    cfg: Mapping[str, Any], markov_overrides: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    if not markov_overrides:
        return cfg
    out = dict(cfg)
    markov = dict(_as_mapping(out.get("markov_filter")))
    markov.update(dict(markov_overrides))
    out["markov_filter"] = markov
    return out


def candidate_bo_payload(theta: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(theta.get("theta_sig_hat"), Mapping) or isinstance(
        theta.get("theta_markov_hat"), Mapping
    ):
        out: dict[str, Any] = {}
        if isinstance(theta.get("theta_sig_hat"), Mapping):
            out["theta_sig_hat"] = dict(
                cast(Mapping[str, Any], theta.get("theta_sig_hat"))
            )
        if isinstance(theta.get("theta_markov_hat"), Mapping):
            out["theta_markov_hat"] = dict(
                cast(Mapping[str, Any], theta.get("theta_markov_hat"))
            )
        return out
    return {"theta_sig_hat": dict(theta)}


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return cast(Mapping[str, Any], value)


def _require_finite_float(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return float(out)


def _require_positive_int(value: Any, *, field: str) -> int:
    try:
        out = int(round(float(value)))
    except Exception as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if out < 1:
        raise ValueError(f"{field} must be >= 1")
    return int(out)


def apply_bo_params_to_cfg(
    cfg: dict[str, Any], bo_best: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(bo_best, Mapping):
        raise TypeError("bo_best must be a mapping")

    out = dict(cfg)
    sig = dict(_as_mapping(out.get("signal")))
    markov = dict(_as_mapping(out.get("markov_filter")))

    theta_sig = bo_best.get("theta_sig_hat")
    if theta_sig is not None:
        theta_sig_map = _require_mapping(theta_sig, field="theta_sig_hat")
        sig["entry_z"] = _require_finite_float(
            theta_sig_map.get("entry_z", sig.get("entry_z", 2.0)),
            field="theta_sig_hat.entry_z",
        )
        sig["exit_z"] = _require_finite_float(
            theta_sig_map.get("exit_z", sig.get("exit_z", 0.5)),
            field="theta_sig_hat.exit_z",
        )
        sig["stop_z"] = _require_finite_float(
            theta_sig_map.get("stop_z", sig.get("stop_z", 2.0)),
            field="theta_sig_hat.stop_z",
        )

    theta_markov = bo_best.get("theta_markov_hat")
    if theta_markov is not None:
        theta_markov_map = _require_mapping(theta_markov, field="theta_markov_hat")
        p_min = _require_finite_float(
            theta_markov_map.get(
                "min_revert_prob", markov.get("min_revert_prob", 0.5)
            ),
            field="theta_markov_hat.min_revert_prob",
        )
        markov["min_revert_prob"] = float(np.clip(p_min, 0.0, 1.0))
        markov["horizon_days"] = _require_positive_int(
            theta_markov_map.get("horizon_days", markov.get("horizon_days", 10)),
            field="theta_markov_hat.horizon_days",
        )
        out["markov_filter"] = markov

    out["signal"] = sig
    return out


def selected_bo_component(bo_res: Mapping[str, Any] | None) -> str | None:
    if not isinstance(bo_res, Mapping):
        return None
    theta_markov = (
        bo_res.get("theta_markov_hat")
        if isinstance(bo_res.get("theta_markov_hat"), Mapping)
        else {}
    )
    if theta_markov:
        return "theta_markov"
    theta_sig = (
        bo_res.get("theta_sig_hat")
        if isinstance(bo_res.get("theta_sig_hat"), Mapping)
        else {}
    )
    if theta_sig:
        return "theta_sig"
    return None


def target_trial_params(bo_res: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    component = selected_bo_component(bo_res)
    if component == "theta_markov":
        theta_hat = (
            bo_res.get("theta_markov_hat")
            if isinstance(bo_res.get("theta_markov_hat"), Mapping)
            else {}
        )
        if theta_hat:
            return component, dict(theta_hat)
    theta_hat = (
        bo_res.get("theta_sig_hat")
        if isinstance(bo_res.get("theta_sig_hat"), Mapping)
        else {}
    )
    if theta_hat:
        return "theta_sig", dict(theta_hat)
    return None
