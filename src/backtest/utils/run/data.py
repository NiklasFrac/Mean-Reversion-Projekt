from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtest.utils.common.prices import as_price_map as _as_price_map
from backtest.loader import prepare_pairs_data
from backtest.utils.alpha import resolve_half_life_cfg
from backtest.utils.run.io import _load_json, _sha256_file

__all__ = [
    "_as_price_map",
    "_pair_prefilter_cfg",
    "_pair_prefilter_inputs",
    "_pairs_prep_inputs",
    "_prepare_pairs_data",
    "_resolve_data_inputs",
]


def _resolve_data_inputs(cfg: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """
    Resolve backtest inputs to avoid drift across Universe/Processing/Analysis/Backtest.

    Supported modes (cfg.data.input_mode):
      - "explicit" (default): use cfg.data.* paths as provided
      - "analysis_meta": derive immutable inputs from an Analysis meta JSON
          - analysis_meta_path: optional (default: pairs_path with suffix .meta.json)
          - uses meta.outputs.run_scoped_pairs_path and meta.config.resolved_config_json
          - sets prices_path to analysis_cfg.data.panel_prices_path when present (LOB requires panel)
          - sets adv_map_path to sibling processed adv_map.pkl when present (keeps inputs consistent)
          - freezes pairs file into <out_dir>/inputs/ to ensure reproducibility even if "latest" is overwritten later
    """
    if not isinstance(cfg, dict):
        return cfg

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    mode = str(data_cfg.get("input_mode", "explicit") or "explicit").strip().lower()
    if mode not in {"explicit", "analysis_meta"}:
        mode = "explicit"
    if mode != "analysis_meta":
        return cfg

    pairs_path_raw = data_cfg.get("pairs_path")
    if not pairs_path_raw:
        raise KeyError(
            "data.pairs_path is required for data.input_mode='analysis_meta'"
        )
    pairs_path = Path(str(pairs_path_raw))
    meta_path = (
        Path(str(data_cfg.get("analysis_meta_path")))
        if data_cfg.get("analysis_meta_path")
        else pairs_path.with_suffix(".meta.json")
    )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Analysis meta JSON not found: {meta_path}. "
            "Run analysis first (it writes *.meta.json next to pairs_path), or set data.analysis_meta_path."
        )

    meta = _load_json(meta_path)
    resolved_cfg_path_raw = (
        (meta.get("config") or {}) if isinstance(meta.get("config"), dict) else {}
    ).get("resolved_config_json")
    if not resolved_cfg_path_raw:
        raise KeyError(
            f"Analysis meta missing config.resolved_config_json: {meta_path}"
        )
    resolved_cfg_path = Path(str(resolved_cfg_path_raw))
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(
            f"Analysis resolved config JSON not found: {resolved_cfg_path}"
        )
    anl_cfg = _load_json(resolved_cfg_path)

    anl_data = anl_cfg.get("data", {}) if isinstance(anl_cfg.get("data"), dict) else {}
    panel_prices_path = anl_data.get("panel_prices_path") or anl_data.get("prices_path")
    if not panel_prices_path:
        raise KeyError(
            f"Analysis resolved config missing data.panel_prices_path/prices_path: {resolved_cfg_path}"
        )
    panel_prices_path = str(panel_prices_path)
    if not Path(panel_prices_path).exists():
        raise FileNotFoundError(
            f"Resolved processing prices path not found: {panel_prices_path}"
        )

    outputs = meta.get("outputs", {}) if isinstance(meta.get("outputs"), dict) else {}
    run_scoped_pairs = (
        outputs.get("run_scoped_pairs_path")
        or outputs.get("latest_pairs_path")
        or str(pairs_path)
    )
    run_scoped_pairs_p = Path(str(run_scoped_pairs))
    if not run_scoped_pairs_p.exists():
        raise FileNotFoundError(
            f"Resolved analysis pairs path not found: {run_scoped_pairs_p}"
        )

    # Freeze pairs into the backtest output directory to prevent drift even when using "latest" analysis outputs.
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    frozen_pairs_path = inputs_dir / run_scoped_pairs_p.name
    if not frozen_pairs_path.exists():
        shutil.copyfile(run_scoped_pairs_p, frozen_pairs_path)

    # Prefer processing run-scoped ADV map if present next to the panel prices.
    adv_candidate = Path(panel_prices_path).parent / "adv_map.pkl"
    adv_map_path = (
        str(adv_candidate)
        if adv_candidate.exists()
        else (
            str(data_cfg.get("adv_map_path")) if data_cfg.get("adv_map_path") else None
        )
    )

    # Record provenance for audits.
    prov = {
        "mode": "analysis_meta",
        "analysis_meta_path": str(meta_path),
        "analysis_run": meta.get("run"),
        "analysis_outputs": outputs,
        "analysis_resolved_config_json": str(resolved_cfg_path),
        "resolved_inputs": {
            "prices_panel_path": panel_prices_path,
            "pairs_path_frozen": str(frozen_pairs_path),
            "adv_map_path": adv_map_path,
        },
        "sha256": {
            "pairs_frozen": _sha256_file(frozen_pairs_path),
            "analysis_meta": _sha256_file(meta_path),
            "analysis_resolved_config_json": _sha256_file(resolved_cfg_path),
        },
    }
    (out_dir / "inputs_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    data_out = dict(data_cfg)
    data_out["prices_path"] = panel_prices_path
    data_out["pairs_path"] = str(frozen_pairs_path)
    if adv_map_path is not None:
        data_out["adv_map_path"] = str(adv_map_path)
    cfg_out = dict(cfg)
    cfg_out["data"] = data_out
    return cfg_out


def _prefilter_range_from_cfg(cfg: Mapping[str, Any]) -> tuple[Any, Any] | None:
    bt = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
    splits = bt.get("splits") if isinstance(bt.get("splits"), dict) else {}
    if isinstance(splits, dict):
        tr_raw = splits.get("train")
        if isinstance(tr_raw, Mapping):
            if tr_raw.get("start") and tr_raw.get("end"):
                return (tr_raw.get("start"), tr_raw.get("end"))
    return None


def _pair_prefilter_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    pair_prefilter = (
        cfg.get("pair_prefilter", {})
        if isinstance(cfg.get("pair_prefilter"), dict)
        else {}
    )
    out = dict(pair_prefilter)
    if "prefilter_active" not in out:
        pairs_prep = (
            cfg.get("pairs_prep", {}) if isinstance(cfg.get("pairs_prep"), dict) else {}
        )
        out["prefilter_active"] = not bool(pairs_prep.get("disable_prefilter", True))
    else:
        out["prefilter_active"] = bool(out["prefilter_active"])
    out["min_obs"] = max(2, int(out.get("min_obs", 30)))
    out["half_life"] = resolve_half_life_cfg(out.get("half_life"))
    return out


def _pair_prefilter_inputs(
    cfg: Mapping[str, Any],
) -> tuple[bool, tuple[Any, Any] | None]:
    pair_prefilter = _pair_prefilter_cfg(cfg)
    disable_prefilter = not bool(pair_prefilter.get("prefilter_active", False))
    prefilter_range = _prefilter_range_from_cfg(cfg)
    return disable_prefilter, prefilter_range


def _pairs_prep_inputs(cfg: Mapping[str, Any]) -> tuple[bool, tuple[Any, Any] | None]:
    # Legacy alias: pair prefilter settings now live under cfg.pair_prefilter.
    return _pair_prefilter_inputs(cfg)


def _prepare_pairs_data(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    cfg: dict[str, Any],
    adv_map: dict[str, float] | None,
    disable_prefilter: bool,
    prefilter_range: tuple[Any, Any] | None,
) -> dict[str, dict[str, Any]]:
    pair_prefilter = _pair_prefilter_cfg(cfg)
    pairs_data = prepare_pairs_data(
        prices,
        pairs,
        adv_map=adv_map,
        verbose=False,
        disable_prefilter=disable_prefilter,
        attach_prices_df=True,
        prefilter_range=prefilter_range,
        pair_prefilter_cfg=pair_prefilter,
    )
    return pairs_data
