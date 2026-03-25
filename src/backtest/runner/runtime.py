from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from backtest.runner.calendars import build_trading_calendar
from backtest.config.cfg import AppConfig, config_to_dict, parse_config
from backtest.config.types import DataConfig
from backtest.runner.loader import (
    load_adv_map,
    load_filtered_pairs,
    load_price_panel,
    select_field_from_panel,
)
from backtest.utils.io import load_json_dict, write_json
from backtest.utils.pair_analysis import resolve_half_life_cfg
from backtest.utils.prices import as_price_map as _as_price_map
from backtest.config.validation import validate_runtime_config


@dataclass(frozen=True)
class RuntimeContext:
    cfg: AppConfig
    out_dir: Path
    data_cfg: DataConfig
    prices_path: Path
    pairs_path: Path
    calendar_name: str
    prefer_col: str
    prices_panel: pd.DataFrame
    prices: pd.DataFrame
    pairs: dict[str, Any]
    adv_map: dict[str, float] | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(str(path))
    try:
        st = p.stat()
    except Exception:
        return {"path": str(p)}
    return {"path": str(p), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def file_key_payload(
    path: str | Path | None,
    *,
    include_path: bool = True,
    include_hash: bool = False,
) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(str(path))
    try:
        st = p.stat()
    except Exception:
        return {"path": str(p)} if include_path else None
    out: dict[str, Any] = {"size": int(st.st_size)}
    if include_path:
        out["path"] = str(p)
    if include_hash:
        try:
            out["sha256"] = sha256_file(p)
        except Exception:
            out["sha256"] = None
    return out


def resolve_data_inputs(cfg: AppConfig, *, out_dir: Path) -> AppConfig:
    if not isinstance(cfg, AppConfig):
        raise TypeError("cfg must be an AppConfig")

    data_cfg = cfg.data
    if data_cfg.input_mode != "analysis_meta":
        return cfg

    pairs_path_raw = data_cfg.pairs_path
    if not pairs_path_raw:
        raise KeyError(
            "data.pairs_path is required for data.input_mode='analysis_meta'"
        )
    pairs_path = Path(str(pairs_path_raw))
    meta_path = (
        Path(str(data_cfg.analysis_meta_path))
        if data_cfg.analysis_meta_path
        else pairs_path.with_suffix(".meta.json")
    )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Analysis meta JSON not found: {meta_path}. "
            "Run analysis first (it writes *.meta.json next to pairs_path), or set data.analysis_meta_path."
        )

    meta = load_json_dict(meta_path)
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
    anl_cfg = load_json_dict(resolved_cfg_path)

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

    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    frozen_pairs_path = inputs_dir / run_scoped_pairs_p.name
    if not frozen_pairs_path.exists():
        shutil.copyfile(run_scoped_pairs_p, frozen_pairs_path)

    adv_candidate = Path(panel_prices_path).parent / "adv_map.pkl"
    adv_map_path = (
        str(adv_candidate)
        if adv_candidate.exists()
        else (
            str(data_cfg.adv_map_path) if data_cfg.adv_map_path else None
        )
    )

    provenance = {
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
            "pairs_frozen": sha256_file(frozen_pairs_path),
            "analysis_meta": sha256_file(meta_path),
            "analysis_resolved_config_json": sha256_file(resolved_cfg_path),
        },
    }
    write_json(out_dir / "inputs_provenance.json", provenance)

    data_out = replace(
        data_cfg,
        prices_path=panel_prices_path,
        pairs_path=str(frozen_pairs_path),
        adv_map_path=str(adv_map_path) if adv_map_path is not None else None,
    )
    return replace(cfg, data=data_out)


def _prefilter_range_from_cfg(
    cfg: AppConfig | Mapping[str, Any],
) -> tuple[Any, Any] | None:
    if isinstance(cfg, AppConfig):
        train = cfg.backtest.splits.get("train")
        if train is not None and train.start and train.end:
            return (train.start, train.end)
        return None

    backtest = cfg.get("backtest") if isinstance(cfg, Mapping) else None
    splits = backtest.get("splits") if isinstance(backtest, Mapping) else None
    train = splits.get("train") if isinstance(splits, Mapping) else None
    if isinstance(train, Mapping):
        start = train.get("start")
        end = train.get("end")
        if start and end:
            return (start, end)
    return None


def pair_prefilter_cfg(cfg: AppConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, AppConfig):
        out = config_to_dict(cfg).get("pair_prefilter", {})
        out["prefilter_active"] = bool(cfg.pair_prefilter.prefilter_active)
        out["min_obs"] = max(2, int(cfg.pair_prefilter.min_obs))
        out["half_life"] = resolve_half_life_cfg(out.get("half_life"))
        return out

    pair_prefilter_raw = (
        cfg.get("pair_prefilter", {}) if isinstance(cfg, Mapping) else {}
    )
    out = dict(pair_prefilter_raw) if isinstance(pair_prefilter_raw, Mapping) else {}

    if "prefilter_active" in out:
        prefilter_active = bool(out.get("prefilter_active", False))
    else:
        pairs_prep = cfg.get("pairs_prep", {}) if isinstance(cfg, Mapping) else {}
        if isinstance(pairs_prep, Mapping) and "disable_prefilter" in pairs_prep:
            prefilter_active = not bool(pairs_prep.get("disable_prefilter", False))
        else:
            prefilter_active = True

    out["prefilter_active"] = bool(prefilter_active)
    out["min_obs"] = max(2, int(out.get("min_obs", 30)))
    out["half_life"] = resolve_half_life_cfg(out.get("half_life"))
    return out


def pair_prefilter_inputs(
    cfg: AppConfig | Mapping[str, Any],
) -> tuple[bool, tuple[Any, Any] | None]:
    prefilter = pair_prefilter_cfg(cfg)
    disable_prefilter = not bool(prefilter.get("prefilter_active", False))
    return disable_prefilter, _prefilter_range_from_cfg(cfg)


def load_runtime_context(cfg: AppConfig, *, out_dir: Path) -> RuntimeContext:
    cfg_eff = resolve_data_inputs(cfg, out_dir=out_dir)
    validate_runtime_config(cfg_eff)

    data_cfg = cfg_eff.data
    prices_path = Path(str(data_cfg.prices_path))
    pairs_path = Path(str(data_cfg.pairs_path))
    calendar_name = str(data_cfg.calendar_name)
    prefer_col = str(data_cfg.prefer_col)

    prices_panel = load_price_panel(str(prices_path))
    prices = select_field_from_panel(prices_panel, field=prefer_col)
    pairs = load_filtered_pairs(str(pairs_path))

    adv_map: dict[str, float] | None = None
    adv_path = data_cfg.adv_map_path
    if adv_path:
        adv_p = Path(str(adv_path))
        if not adv_p.exists():
            raise FileNotFoundError(f"data.adv_map_path not found: {adv_p}")
        adv_map = load_adv_map(adv_p)

    return RuntimeContext(
        cfg=cfg_eff,
        out_dir=out_dir,
        data_cfg=data_cfg,
        prices_path=prices_path,
        pairs_path=pairs_path,
        calendar_name=calendar_name,
        prefer_col=prefer_col,
        prices_panel=prices_panel,
        prices=prices,
        pairs=pairs,
        adv_map=adv_map,
    )


def limit_runtime_pairs(ctx: RuntimeContext, *, limit: int) -> RuntimeContext:
    if limit <= 0 or len(ctx.pairs) <= limit:
        return ctx
    keys = list(ctx.pairs.keys())[: int(limit)]
    return replace(ctx, pairs={k: ctx.pairs[k] for k in keys})


def build_runtime_calendar(ctx: RuntimeContext) -> pd.DatetimeIndex:
    return build_trading_calendar(
        _as_price_map(ctx.prices), calendar_name=ctx.calendar_name
    )
