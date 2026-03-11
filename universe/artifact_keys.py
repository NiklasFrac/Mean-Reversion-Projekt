from __future__ import annotations

from pathlib import Path
from typing import MutableMapping

KEY_ADV_CACHE = "adv_cache"
KEY_ADV_CSV = "adv_csv"
KEY_ADV_CSV_FILTERED = "adv_csv_filtered"
KEY_PRICES = "prices"
KEY_PRICES_CANONICAL = "prices_canonical"
KEY_PRICES_UNADJUSTED = "prices_unadjusted"
KEY_PRICES_UNADJUSTED_WARMUP = "prices_unadjusted_warmup"
KEY_TICKERS_EXT_CSV = "tickers_ext_csv"
KEY_VOLUMES = "volumes"
KEY_VOLUMES_CANONICAL = "volumes_canonical"
KEY_VOLUMES_UNADJUSTED = "volumes_unadjusted"

ALL_ARTIFACT_KEYS = frozenset(
    {
        KEY_ADV_CACHE,
        KEY_ADV_CSV,
        KEY_ADV_CSV_FILTERED,
        KEY_PRICES,
        KEY_PRICES_CANONICAL,
        KEY_PRICES_UNADJUSTED,
        KEY_PRICES_UNADJUSTED_WARMUP,
        KEY_TICKERS_EXT_CSV,
        KEY_VOLUMES,
        KEY_VOLUMES_CANONICAL,
        KEY_VOLUMES_UNADJUSTED,
    }
)


def set_artifact_path(
    artifacts: MutableMapping[str, str],
    *,
    key: str,
    path: str | Path | None,
) -> None:
    if path is None:
        return
    artifacts[key] = str(path)
