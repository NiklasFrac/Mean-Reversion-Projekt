from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.calendars import build_trading_calendar
from backtest.utils.common.prices import as_price_map as _as_price_map
from backtest.loader import (
    load_adv_map,
    load_filtered_pairs,
    load_price_panel,
    select_field_from_panel,
)
from backtest.utils.run.data import _resolve_data_inputs
from backtest.utils.run.strategy import _validate_cfg_strict


@dataclass(frozen=True)
class RuntimeContext:
    cfg: dict[str, Any]
    out_dir: Path
    data_cfg: dict[str, Any]
    prices_path: Path
    pairs_path: Path
    calendar_name: str
    prefer_col: str
    prices_panel: pd.DataFrame
    prices: pd.DataFrame
    pairs: dict[str, Any]
    adv_map: dict[str, float] | None


def load_runtime_context(cfg: dict[str, Any], *, out_dir: Path) -> RuntimeContext:
    cfg_eff = _resolve_data_inputs(dict(cfg), out_dir=out_dir)
    _validate_cfg_strict(cfg_eff)

    data_cfg = cfg_eff.get("data", {}) if isinstance(cfg_eff.get("data"), dict) else {}
    prices_path = Path(str(data_cfg.get("prices_path")))
    pairs_path = Path(str(data_cfg.get("pairs_path")))
    calendar_name = str(data_cfg.get("calendar_name", "XNYS"))
    prefer_col = str(data_cfg.get("prefer_col", "close"))

    prices_panel = load_price_panel(
        str(prices_path),
        coerce_timezone="keep",
    )
    prices = select_field_from_panel(prices_panel, field=prefer_col)
    pairs = load_filtered_pairs(str(pairs_path))

    adv_map: dict[str, float] | None = None
    adv_path = data_cfg.get("adv_map_path")
    if adv_path:
        adv_p = Path(str(adv_path))
        if not adv_p.exists():
            raise FileNotFoundError(f"data.adv_map_path not found: {adv_p}")
        adv_map = load_adv_map(adv_p)

    return RuntimeContext(
        cfg=cfg_eff,
        out_dir=out_dir,
        data_cfg=dict(data_cfg),
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
