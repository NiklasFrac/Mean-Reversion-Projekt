from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from backtest.optimize.paper_bo_parts.pipeline import (
    _build_train_inputs_from_pairs_data,
    _calendar_from_pairs_data,
)
from backtest.utils.run.bo import _assert_single_stage_bo_cfg, _bo_key_payload
from backtest.utils.run.data import _pair_prefilter_inputs, _prepare_pairs_data
from backtest.utils.tz import NY_TZ, get_ex_tz

logger = logging.getLogger("backtest.run.bo")


@dataclass(frozen=True)
class BORunResult:
    cfg_eff: dict[str, Any]
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


def _clamp_bo_for_quick(cfg_eff: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(cfg_eff)
    bo = dict(out.get("bo") or {})
    _assert_single_stage_bo_cfg(bo)
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
    return out


def _selection_metric_from_cfg(cfg_eff: Mapping[str, Any]) -> str:
    bo_cfg = cfg_eff.get("bo", {}) if isinstance(cfg_eff.get("bo"), Mapping) else {}
    mode = str(bo_cfg.get("mode", "realistic")).strip().lower()
    if mode == "realistic":
        realistic = (
            bo_cfg.get("realistic", {})
            if isinstance(bo_cfg.get("realistic"), Mapping)
            else {}
        )
        metric = str(realistic.get("metric", "sharpe")).strip().lower()
        return metric or "sharpe"
    return "sharpe"


def _selected_bo_component(bo_res: Mapping[str, Any] | None) -> str | None:
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


def _target_trial_params(
    bo_res: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    component = _selected_bo_component(bo_res)
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

    target = _target_trial_params(bo_res)
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


def run_bo_if_enabled(
    *,
    cfg_eff: dict[str, Any],
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
    bo_cfg = cfg_eff.get("bo", {}) if isinstance(cfg_eff.get("bo"), dict) else {}
    _assert_single_stage_bo_cfg(bo_cfg)
    if not bool(bo_cfg.get("enabled", False)):
        return BORunResult(
            cfg_eff=dict(cfg_eff),
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

    from backtest.optimize.paper_bo import (
        apply_bo_params_to_cfg,
        run_paper_bo_conservative,
    )

    cfg_bo = _clamp_bo_for_quick(cfg_eff) if quick else dict(cfg_eff)
    pairs_data_local = pairs_data
    if pairs_data_local is None:
        disable_prefilter, prefilter_range = _pair_prefilter_inputs(cfg_bo)
        pairs_data_local = _prepare_pairs_data(
            prices=prices,
            prices_panel=prices_panel,
            pairs=pairs,
            cfg=cfg_bo,
            adv_map=adv_map,
            disable_prefilter=disable_prefilter,
            prefilter_range=prefilter_range,
        )

    bo_key_payload = _bo_key_payload(cfg_eff)
    bo_id = hashlib.sha256(
        json.dumps(bo_key_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    bo_base = Path(str(bo_cfg.get("out_dir", "runs/results/bo")))
    bo_out = bo_base / f"BO-{bo_id}"
    if quick:
        bo_out = bo_out / "_quick"

    bo_res = run_paper_bo_conservative(
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        pairs_data=pairs_data_local,
        adv_map=adv_map,
        cfg=cfg_bo,
        out_dir=bo_out,
    )

    cfg_base = cfg_bo if persist_quick_budget else cfg_eff
    cfg_applied = apply_bo_params_to_cfg(dict(cfg_base), bo_res)
    selection_metric = _selection_metric_from_cfg(cfg_applied)
    selected_cv_scores = _extract_selected_cv_scores(
        bo_out, bo_res, selection_metric=selection_metric
    )

    return BORunResult(
        cfg_eff=cfg_applied,
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
    *,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    ex_tz = get_ex_tz(cfg or {}, None, default=NY_TZ) if cfg is not None else None
    cal = _calendar_from_pairs_data(pairs_data, ex_tz=ex_tz)
    if cal.empty:
        raise ValueError("pairs_data does not contain usable price series.")

    per_pair_prices = _build_train_inputs_from_pairs_data(pairs_data, cal=cal)
    if not per_pair_prices:
        raise ValueError("pairs_data does not contain usable price series.")

    series_by_sym: dict[str, pd.Series] = {}
    pairs_map: dict[str, dict[str, str]] = {}

    def _sym(v: Any) -> str | None:
        s = str(v or "").strip().upper()
        return s or None

    def _merge_series(existing: pd.Series | None, new: pd.Series) -> pd.Series:
        if existing is None:
            return new
        try:
            merged = existing.combine_first(new)
        except Exception:
            return existing if existing.count() >= new.count() else new
        try:
            if merged.count() >= max(existing.count(), new.count()):
                return merged
        except Exception:
            return merged
        return existing if existing.count() >= new.count() else new

    for pair, meta in (pairs_data or {}).items():
        if not isinstance(meta, Mapping):
            continue
        raw_meta = meta.get("meta") if isinstance(meta.get("meta"), Mapping) else meta
        t1 = _sym(raw_meta.get("t1") if isinstance(raw_meta, Mapping) else None)
        t2 = _sym(raw_meta.get("t2") if isinstance(raw_meta, Mapping) else None)
        if t1 is None or t2 is None:
            p = str(pair)
            for sep in ("-", "/", "_", "|", ":"):
                if sep in p:
                    a, b = p.split(sep, 1)
                    t1 = _sym(a)
                    t2 = _sym(b)
                    break
        if t1 is None or t2 is None:
            continue
        pair_key = str(pair)
        if pair_key not in per_pair_prices:
            continue
        yz = per_pair_prices[pair_key]
        y = yz.get("y")
        x = yz.get("x")
        if not isinstance(y, pd.Series) or not isinstance(x, pd.Series):
            continue
        series_by_sym[t1] = _merge_series(series_by_sym.get(t1), y)
        series_by_sym[t2] = _merge_series(series_by_sym.get(t2), x)
        pairs_map[pair_key] = {"t1": t1, "t2": t2}

    if not series_by_sym or not pairs_map:
        raise ValueError("pairs_data does not contain usable price series.")

    prices_df = pd.concat(
        {k: pd.to_numeric(v, errors="coerce") for k, v in series_by_sym.items()}, axis=1
    )
    return prices_df, pairs_map
