from __future__ import annotations

from typing import Any, Mapping

from backtest.utils.run.io import _file_key_payload
from backtest.utils.run.data import _pair_prefilter_cfg


_LEGACY_BO_STAGE_KEYS = ("stage1", "stage2", "z_window_range", "precompute_beta_z")
_BO_RUNTIME_KEYS = ("init_points", "n_iter", "patience")


def _assert_single_stage_bo_cfg(bo_cfg: Mapping[str, Any]) -> None:
    if not isinstance(bo_cfg, Mapping):
        return
    legacy = [key for key in _LEGACY_BO_STAGE_KEYS if key in bo_cfg]
    if legacy:
        joined = ", ".join(f"bo.{key}" for key in legacy)
        raise ValueError(
            f"Legacy BO config keys are no longer supported: {joined}. "
            "Use root-level bo search keys such as bo.entry_z_range and bo.init_points."
        )


def _sanitize_bo_cfg_for_key(bo_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bo_cfg, Mapping):
        return {}
    _assert_single_stage_bo_cfg(bo_cfg)
    out = dict(bo_cfg)
    out.pop("out_dir", None)
    for key in _BO_RUNTIME_KEYS:
        out.pop(key, None)
    return out


def _bo_key_payload(cfg: Mapping[str, Any]) -> dict[str, Any]:
    def _as_dict(v: Any) -> dict[str, Any]:
        return dict(v) if isinstance(v, Mapping) else {}

    data = _as_dict(cfg.get("data"))
    input_mode = str(data.get("input_mode", "explicit") or "explicit").strip().lower()
    bt = _as_dict(cfg.get("backtest"))
    splits = _as_dict(bt.get("splits"))
    train = _as_dict(splits.get("train"))
    pair_prefilter = _pair_prefilter_cfg(cfg)
    bo_cfg = _as_dict(cfg.get("bo"))
    _assert_single_stage_bo_cfg(bo_cfg)
    return {
        "version": 9,
        "seed": int(cfg.get("seed", 42)),
        "data": {
            "prices": _file_key_payload(
                data.get("prices_path"), include_path=True, include_hash=False
            ),
            "pairs": _file_key_payload(
                data.get("pairs_path"),
                include_path=(input_mode != "analysis_meta"),
                include_hash=True,
            ),
            "adv_map": _file_key_payload(
                data.get("adv_map_path"), include_path=False, include_hash=True
            ),
        },
        "train": {
            "start": str(train.get("start", "")),
            "end": str(train.get("end", "")),
        },
        "backtest": {
            "initial_capital": bt.get("initial_capital"),
            "risk_per_trade": bt.get("risk_per_trade"),
            "calendar_mapping": bt.get("calendar_mapping"),
            "settlement_lag_bars": bt.get("settlement_lag_bars"),
            "annualization_factor": bt.get("annualization_factor"),
        },
        "strategy": _as_dict(cfg.get("strategy")),
        "signal": _as_dict(cfg.get("signal")),
        "markov_filter": _as_dict(cfg.get("markov_filter")),
        "spread_zscore": _as_dict(cfg.get("spread_zscore")),
        "pair_prefilter": pair_prefilter,
        "risk": _as_dict(cfg.get("risk")),
        "execution": _as_dict(cfg.get("execution")),
        "borrow": _as_dict(cfg.get("borrow")),
        "cv": _as_dict(cfg.get("cv")),
        "bo": _sanitize_bo_cfg_for_key(bo_cfg),
    }
