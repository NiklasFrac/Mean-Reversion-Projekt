from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import pandas as pd
from universe.artifact_keys import (
    KEY_ADV_CACHE,
    KEY_ADV_CSV,
    KEY_ADV_CSV_FILTERED,
    set_artifact_path,
)
from universe.storage import ArtifactPaths, resolve_artifact_paths

__all__ = ["emit_adv_cache_and_filtered_csv"]
logger = logging.getLogger("runner_universe")


def emit_adv_cache_and_filtered_csv(
    *,
    data_cfg: Mapping[str, Any],
    universe_cfg: Mapping[str, Any],
    tickers_final: list[str],
    artifacts: MutableMapping[str, str],
    norm_symbol_fn: Callable[[str], str],
    atomic_write_pickle_fn: Callable[[object, Path], None],
    artifact_paths: ArtifactPaths | None = None,
) -> tuple[Path | None, Path | None]:
    paths = artifact_paths or resolve_artifact_paths(
        universe_cfg=universe_cfg, data_cfg=data_cfg
    )
    # Keep adv_path untouched (produced by builder); only emit the per-ticker pickle cache if possible.
    adv_csv_val = paths.adv_csv
    adv_cache_val = paths.adv_cache
    adv_csv_path: Path | None = None
    adv_csv_filtered_path: Path | None = None
    if adv_csv_val and Path(str(adv_csv_val)).exists():
        adv_csv_path = Path(str(adv_csv_val))
        set_artifact_path(artifacts, key=KEY_ADV_CSV, path=adv_csv_path)
        try:
            adv_df = pd.read_csv(adv_csv_path)
        except Exception as exc:
            logger.warning("Failed to read ADV CSV %s: %s", adv_csv_path, exc)
            adv_df = pd.DataFrame()
        if not adv_df.empty and "ticker" in adv_df.columns:
            adv_df["ticker"] = adv_df["ticker"].map(norm_symbol_fn)
            tick_set = set(tickers_final)
            filtered = adv_df[adv_df["ticker"].isin(tick_set)]
            filtered_csv_path = paths.adv_csv_filtered
            if not filtered.empty:
                try:
                    filtered_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    filtered.to_csv(filtered_csv_path, index=False)
                    set_artifact_path(
                        artifacts,
                        key=KEY_ADV_CSV_FILTERED,
                        path=filtered_csv_path,
                    )
                    adv_csv_filtered_path = filtered_csv_path
                except Exception as exc:
                    logger.warning(
                        "Failed to write filtered ADV CSV %s: %s",
                        filtered_csv_path,
                        exc,
                    )
            key = (
                "adv_dollar_median_usd"
                if "adv_dollar_median_usd" in filtered.columns
                else None
            )
            if key is None:
                for cand in ("adv", "dollar_adv", "dollar_adv_hist"):
                    if cand in filtered.columns:
                        key = cand
                        break
            if key is not None:
                mapping: dict[str, float] = {}
                for _, row in filtered.iterrows():
                    try:
                        ticker_raw = row.get("ticker")
                        if ticker_raw is None or pd.isna(ticker_raw):
                            continue
                        sym = norm_symbol_fn(str(ticker_raw))
                        if not sym:
                            continue
                        mapping[sym] = float(row[key])
                    except Exception:
                        continue
                if adv_cache_val is not None:
                    atomic_write_pickle_fn(mapping, Path(str(adv_cache_val)))
                    set_artifact_path(artifacts, key=KEY_ADV_CACHE, path=adv_cache_val)

    return adv_csv_path, adv_csv_filtered_path
