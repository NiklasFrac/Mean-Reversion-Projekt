from __future__ import annotations

from pathlib import Path
from .types import AppConfig

__all__ = ["validate_runtime_config"]


def validate_runtime_config(cfg: AppConfig) -> None:
    if not isinstance(cfg, AppConfig):
        raise TypeError("cfg must be an AppConfig")

    prices_path = cfg.data.prices_path
    pairs_path = cfg.data.pairs_path
    if not prices_path:
        raise KeyError("Config missing data.prices_path")
    if not pairs_path:
        raise KeyError("Config missing data.pairs_path")
    if not Path(str(prices_path)).exists():
        raise FileNotFoundError(f"data.prices_path not found: {prices_path}")
    if not Path(str(pairs_path)).exists():
        raise FileNotFoundError(f"data.pairs_path not found: {pairs_path}")
    if cfg.data.adv_map_path and not Path(cfg.data.adv_map_path).exists():
        raise FileNotFoundError(f"data.adv_map_path not found: {cfg.data.adv_map_path}")
    if cfg.data.input_mode == "analysis_meta":
        meta_path = cfg.data.analysis_meta_path
        if not meta_path:
            raise KeyError(
                "data.analysis_meta_path is required for data.input_mode='analysis_meta'"
            )
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"data.analysis_meta_path not found: {meta_path}")
