from __future__ import annotations

from typing import Any, Final, Mapping, MutableMapping

DEFAULT_OUTPUT_TICKERS_CSV: Final[str] = "runs/data/tickers_universe.csv"
DEFAULT_MANIFEST_PATH: Final[str] = "runs/data/universe_manifest.json"
DEFAULT_FUNDAMENTALS_OUT: Final[str | None] = None
DEFAULT_ADV_CACHE_PATH: Final[str] = "runs/data/ticker_adv.pkl"

DEFAULT_ADV_CSV_PATH: Final[str] = "runs/data/adv_map_usd.csv"
DEFAULT_ADV_FILTERED_CSV_PATH: Final[str | None] = None
DEFAULT_RAW_PRICES_PATH: Final[str] = "runs/data/raw_prices.pkl"
DEFAULT_RAW_VOLUMES_PATH: Final[str] = "runs/data/raw_volume.pkl"
DEFAULT_RAW_PRICES_UNADJ_WARMUP_PATH: Final[str] = (
    "runs/data/raw_prices_unadj_warmup.pkl"
)
DEFAULT_RAW_PRICES_UNADJ_PATH: Final[str] = "runs/data/raw_prices_unadj.pkl"
DEFAULT_RAW_VOLUMES_UNADJ_PATH: Final[str] = "runs/data/raw_volume_unadj.pkl"

DEFAULT_CHECKPOINT_PATH: Final[str] = "runs/data/universe_checkpoint.json"
DEFAULT_RUN_SCOPED_OUTPUTS_DIR: Final[str] = "runs/data/by_run"

UNIVERSE_ARTIFACT_PATH_DEFAULTS: Final[Mapping[str, str | None]] = {
    "output_tickers_csv": DEFAULT_OUTPUT_TICKERS_CSV,
    "manifest": DEFAULT_MANIFEST_PATH,
    "fundamentals_out": DEFAULT_FUNDAMENTALS_OUT,
    "adv_cache": DEFAULT_ADV_CACHE_PATH,
}

DATA_ARTIFACT_PATH_DEFAULTS: Final[Mapping[str, str | None]] = {
    "adv_path": DEFAULT_ADV_CSV_PATH,
    "adv_filtered_path": DEFAULT_ADV_FILTERED_CSV_PATH,
    "raw_prices_cache": DEFAULT_RAW_PRICES_PATH,
    "volume_path": DEFAULT_RAW_VOLUMES_PATH,
    "raw_prices_unadj_warmup_cache": DEFAULT_RAW_PRICES_UNADJ_WARMUP_PATH,
    "raw_prices_unadj_cache": DEFAULT_RAW_PRICES_UNADJ_PATH,
    "raw_volume_unadj_cache": DEFAULT_RAW_VOLUMES_UNADJ_PATH,
}

RUNTIME_ARTIFACT_PATH_DEFAULTS: Final[Mapping[str, str | None]] = {
    "checkpoint_path": DEFAULT_CHECKPOINT_PATH,
    "run_scoped_outputs_dir": DEFAULT_RUN_SCOPED_OUTPUTS_DIR,
}


def apply_artifact_path_defaults(
    *,
    universe_cfg: MutableMapping[str, Any],
    data_cfg: MutableMapping[str, Any],
    runtime_cfg: MutableMapping[str, Any],
) -> None:
    for key, default in UNIVERSE_ARTIFACT_PATH_DEFAULTS.items():
        universe_cfg.setdefault(key, default)
    for key, default in DATA_ARTIFACT_PATH_DEFAULTS.items():
        data_cfg.setdefault(key, default)
    for key, default in RUNTIME_ARTIFACT_PATH_DEFAULTS.items():
        runtime_cfg.setdefault(key, default)
