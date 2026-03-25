from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.config.cfg import AppConfig, config_to_dict, parse_config
from backtest.optimize.inputs import prices_frame_from_pairs_data
from backtest.optimize.optimizer import run_bo_conservative
from backtest.optimize.params import (
    apply_bo_params_to_cfg,
    clean_bo_cfg_for_key,
    resolve_bo_config,
    target_trial_params,
)
from backtest.runner.runtime import file_key_payload, pair_prefilter_cfg
from backtest.runner.window_run import prepare_pairs_data_for_cfg

logger = logging.getLogger("backtest.optimize.runner")


def _bo_key_payload(cfg: AppConfig) -> dict[str, Any]:
    cfg_dict = config_to_dict(cfg)

    def _as_dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    data = _as_dict(cfg_dict.get("data"))
    input_mode = str(data.get("input_mode", "explicit") or "explicit").strip().lower()
    bt = _as_dict(cfg_dict.get("backtest"))
    splits = _as_dict(bt.get("splits"))
    train = _as_dict(splits.get("train"))
    prefilter = pair_prefilter_cfg(cfg)
    bo_cfg = _as_dict(cfg_dict.get("bo"))
    return {
        "version": 9,
        "seed": int(((cfg_dict.get("runtime") or {}).get("seed", 42))),
        "data": {
            "prices": file_key_payload(
                data.get("prices_path"), include_path=True, include_hash=False
            ),
            "pairs": file_key_payload(
                data.get("pairs_path"),
                include_path=(input_mode != "analysis_meta"),
                include_hash=True,
            ),
            "adv_map": file_key_payload(
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
        "strategy": _as_dict(cfg_dict.get("strategy")),
        "signal": _as_dict(cfg_dict.get("signal")),
        "markov_filter": _as_dict(cfg_dict.get("markov_filter")),
        "spread_zscore": _as_dict(cfg_dict.get("spread_zscore")),
        "pair_prefilter": prefilter,
        "risk": _as_dict(cfg_dict.get("risk")),
        "execution": _as_dict(cfg_dict.get("execution")),
        "borrow": _as_dict(cfg_dict.get("borrow")),
        "cv": _as_dict(cfg_dict.get("cv")),
        "bo": clean_bo_cfg_for_key(bo_cfg),
    }


@dataclass(frozen=True)
class BORunResult:
    cfg_eff: AppConfig
    pairs_data: dict[str, Any] | None
    bo_res: dict[str, Any] | None
    bo_id: str | None
    bo_out: Path | None
    bo_key_payload: dict[str, Any] | None
    selected_cv_scores: pd.DataFrame | None
    selection_metric: str | None

    @property
    def bo_meta(self) -> dict[str, Any] | None:
        if self.bo_id is None or self.bo_out is None or self.bo_key_payload is None:
            return None
        return {
            "bo_id": self.bo_id,
            "bo_out_dir": str(self.bo_out),
            "bo_key_payload": self.bo_key_payload,
        }


def _clamp_bo_for_quick(cfg_eff: AppConfig) -> AppConfig:
    out = config_to_dict(cfg_eff)
    bo = dict(out.get("bo") or {})
    if "init_points" in bo:
        bo["init_points"] = int(min(int(bo.get("init_points", 0) or 0), 3))
    if "n_iter" in bo:
        bo["n_iter"] = int(min(int(bo.get("n_iter", 0) or 0), 6))
    if "markov_init_points" in bo:
        bo["markov_init_points"] = int(
            min(int(bo.get("markov_init_points", 0) or 0), 3)
        )
    if "markov_n_iter" in bo:
        bo["markov_n_iter"] = int(min(int(bo.get("markov_n_iter", 0) or 0), 6))
    out["bo"] = bo
    return parse_config(out)


def _selection_metric_from_cfg(cfg_eff: AppConfig) -> str:
    resolved = resolve_bo_config(cfg_eff)
    if resolved.mode == "realistic":
        return str(resolved.realistic_metric)
    return "sharpe"


def _params_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for key in a.keys():
        va = a.get(key)
        vb = b.get(key)
        if va is None or vb is None:
            if va != vb:
                return False
            continue
        try:
            fa = float(va)
            fb = float(vb)
            if abs(fa - fb) > 1e-9:
                return False
        except Exception:
            if str(va) != str(vb):
                return False
    return True


def _extract_selected_cv_scores(
    bo_out: Path | None, bo_res: Mapping[str, Any] | None, *, selection_metric: str
) -> pd.DataFrame:
    if bo_out is None or not isinstance(bo_res, Mapping):
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )
    trials_path = bo_out / "bo_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )

    target = target_trial_params(bo_res)
    if target is None:
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )
    component, target_params = target

    try:
        df = pd.read_csv(trials_path)
    except Exception:
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )
    if df.empty or "component" not in df.columns or "params_json" not in df.columns:
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if str(row.get("component", "")) != component:
            continue
        fold_id = row.get("fold_id")
        if pd.isna(fold_id):
            continue
        params_raw = row.get("params_json")
        try:
            params = json.loads(str(params_raw))
        except Exception:
            continue
        if not isinstance(params, Mapping) or not _params_match(
            dict(params), target_params
        ):
            continue
        score_raw = row.get("oos_score", row.get("score"))
        try:
            score = float(score_raw)
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        rows.append(
            {
                "fold_id": int(fold_id),
                "score": float(score),
                "selection_metric": selection_metric,
                "component": component,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["fold_id", "score", "selection_metric", "component"]
        )
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def load_bo_trials(bo_out: Path | None) -> pd.DataFrame:
    cols = [
        "timestamp",
        "component",
        "model_id",
        "params_json",
        "metric",
        "score",
        "sharpe",
        "fold",
        "fold_id",
        "is_score",
        "oos_score",
    ]
    if bo_out is None:
        return pd.DataFrame(columns=cols)
    trials_path = Path(bo_out) / "bo_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(trials_path)
    except Exception:
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df.loc[:, cols].copy()


def run_bo_if_enabled(
    *,
    cfg_eff: AppConfig,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    adv_map: dict[str, float] | None,
    out_dir: Path,
    quick: bool,
    pairs_data: dict[str, Any] | None = None,
    persist_quick_budget: bool = False,
) -> BORunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bool(cfg_eff.bo.enabled):
        return BORunResult(
            cfg_eff=cfg_eff,
            pairs_data=pairs_data,
            bo_res=None,
            bo_id=None,
            bo_out=None,
            bo_key_payload=None,
            selected_cv_scores=pd.DataFrame(
                columns=["fold_id", "score", "selection_metric", "component"]
            ),
            selection_metric=None,
        )

    cfg_bo = _clamp_bo_for_quick(cfg_eff) if quick else cfg_eff
    pairs_data_local = pairs_data
    if pairs_data_local is None:
        pairs_data_local = prepare_pairs_data_for_cfg(
            prices=prices,
            pairs=pairs,
            cfg=cfg_bo,
            adv_map=adv_map,
        )

    bo_key_payload = _bo_key_payload(cfg_eff)
    bo_id = hashlib.sha256(
        json.dumps(bo_key_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    bo_base = resolve_bo_config(cfg_eff).out_dir
    bo_out = bo_base / f"BO-{bo_id}"
    if quick:
        bo_out = bo_out / "_quick"

    bo_res = run_bo_conservative(
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        pairs_data=pairs_data_local,
        adv_map=adv_map,
        cfg=cfg_bo,
        out_dir=bo_out,
    )

    cfg_base = cfg_bo if persist_quick_budget else cfg_eff
    cfg_applied = apply_bo_params_to_cfg(config_to_dict(cfg_base), bo_res)
    cfg_applied_obj = parse_config(cfg_applied)
    selection_metric = _selection_metric_from_cfg(cfg_applied_obj)
    selected_cv_scores = _extract_selected_cv_scores(
        bo_out, bo_res, selection_metric=selection_metric
    )

    return BORunResult(
        cfg_eff=cfg_applied_obj,
        pairs_data=pairs_data_local,
        bo_res=cast(dict[str, Any], bo_res),
        bo_id=bo_id,
        bo_out=bo_out,
        bo_key_payload=bo_key_payload,
        selected_cv_scores=selected_cv_scores,
        selection_metric=selection_metric,
    )


def _prices_frame_from_pairs_data(
    pairs_data: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    return prices_frame_from_pairs_data(pairs_data)
