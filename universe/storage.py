from __future__ import annotations

import logging
import pickle
import shutil as _sh
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from universe.artifact_defaults import (
    DEFAULT_ADV_CACHE_PATH,
    DEFAULT_ADV_CSV_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_TICKERS_CSV,
    DEFAULT_RAW_PRICES_PATH,
    DEFAULT_RAW_PRICES_UNADJ_PATH,
    DEFAULT_RAW_PRICES_UNADJ_WARMUP_PATH,
    DEFAULT_RAW_VOLUMES_PATH,
    DEFAULT_RAW_VOLUMES_UNADJ_PATH,
    DEFAULT_RUN_SCOPED_OUTPUTS_DIR,
)

logger = logging.getLogger("runner_universe")

__all__ = [
    "ArtifactPaths",
    "resolve_artifact_paths",
    "hashed_artifact_siblings",
    "ensure_updated_at_column",
    "load_fundamentals_store",
    "save_fundamentals_store",
    "artifact_targets",
    "build_event_frame",
    "write_event_artifact",
    "hash_bytes",
]


@dataclass(frozen=True)
class ArtifactPaths:
    output_tickers_csv: Path
    output_tickers_ext_csv: Path | None
    manifest: Path
    fundamentals_out: Path | None
    adv_cache: Path | None
    adv_csv: Path
    adv_csv_filtered: Path
    raw_prices_cache: Path
    volume_path: Path
    raw_prices_unadj_warmup_cache: Path
    raw_prices_unadj_cache: Path
    raw_volume_unadj_cache: Path
    checkpoint_path: Path | None
    run_scoped_outputs_dir: Path


def _coerce_path(
    value: Any,
    *,
    default: str | None = None,
    optional: bool = False,
    allow_disabled_literals: bool = False,
) -> Path | None:
    raw: str | None
    if allow_disabled_literals and value is None:
        return None
    if value is None:
        raw = default
    else:
        raw = str(value).strip()
        if allow_disabled_literals and raw.lower() in {"", "none", "null"}:
            return None
        if not raw:
            raw = default
    if raw is None:
        return None if optional else Path("")
    if optional and str(raw).strip().lower() in {"none", "null"}:
        return None
    return Path(str(raw))


def resolve_artifact_paths(
    *,
    universe_cfg: Mapping[str, Any] | None = None,
    data_cfg: Mapping[str, Any] | None = None,
    runtime_cfg: Mapping[str, Any] | None = None,
) -> ArtifactPaths:
    universe = dict(universe_cfg or {})
    data = dict(data_cfg or {})
    runtime = dict(runtime_cfg or {})

    output_tickers_csv = _coerce_path(
        universe.get("output_tickers_csv"),
        default=DEFAULT_OUTPUT_TICKERS_CSV,
    )
    output_tickers_ext_csv = _coerce_path(
        universe.get("output_tickers_ext_csv"),
        optional=True,
    )
    manifest = _coerce_path(
        universe.get("manifest"),
        default=DEFAULT_MANIFEST_PATH,
    )
    fundamentals_out = _coerce_path(
        universe.get("fundamentals_out"),
        optional=True,
    )
    adv_cache = _coerce_path(
        universe.get("adv_cache"),
        default=DEFAULT_ADV_CACHE_PATH,
        optional=True,
    )

    adv_csv = _coerce_path(
        data.get("adv_path"),
        default=DEFAULT_ADV_CSV_PATH,
    )
    if adv_csv is None:
        adv_csv = Path(DEFAULT_ADV_CSV_PATH)
    adv_csv_filtered = _coerce_path(
        data.get("adv_filtered_path"),
        optional=True,
    )
    if adv_csv_filtered is None:
        adv_csv_filtered = adv_csv.with_name(
            adv_csv.stem + "_filtered" + adv_csv.suffix
        )

    raw_prices_cache = _coerce_path(
        data.get("raw_prices_cache"),
        default=DEFAULT_RAW_PRICES_PATH,
    )
    volume_path = _coerce_path(
        data.get("volume_path"),
        default=DEFAULT_RAW_VOLUMES_PATH,
    )
    raw_prices_unadj_warmup_cache = _coerce_path(
        data.get("raw_prices_unadj_warmup_cache"),
        default=DEFAULT_RAW_PRICES_UNADJ_WARMUP_PATH,
    )
    raw_prices_unadj_cache = _coerce_path(
        data.get("raw_prices_unadj_cache"),
        default=DEFAULT_RAW_PRICES_UNADJ_PATH,
    )
    try:
        if (
            raw_prices_unadj_cache is not None
            and raw_prices_unadj_warmup_cache is not None
            and raw_prices_unadj_cache.resolve(strict=False)
            == raw_prices_unadj_warmup_cache.resolve(strict=False)
        ):
            raw_prices_unadj_cache = raw_prices_unadj_cache.with_name(
                raw_prices_unadj_cache.stem
                + "_filtered"
                + raw_prices_unadj_cache.suffix
            )
    except Exception:
        pass
    raw_volume_unadj_cache = _coerce_path(
        data.get("raw_volume_unadj_cache"),
        default=DEFAULT_RAW_VOLUMES_UNADJ_PATH,
    )

    checkpoint_value: Any
    if "checkpoint_path" in runtime:
        checkpoint_value = runtime.get("checkpoint_path")
    else:
        checkpoint_value = DEFAULT_CHECKPOINT_PATH
    checkpoint_path = _coerce_path(
        checkpoint_value,
        default=DEFAULT_CHECKPOINT_PATH,
        optional=True,
        allow_disabled_literals=True,
    )
    run_scoped_outputs_dir = _coerce_path(
        runtime.get("run_scoped_outputs_dir"),
        default=DEFAULT_RUN_SCOPED_OUTPUTS_DIR,
    )

    assert output_tickers_csv is not None
    assert manifest is not None
    assert adv_csv is not None
    assert adv_csv_filtered is not None
    assert raw_prices_cache is not None
    assert volume_path is not None
    assert raw_prices_unadj_warmup_cache is not None
    assert raw_prices_unadj_cache is not None
    assert raw_volume_unadj_cache is not None
    assert run_scoped_outputs_dir is not None

    return ArtifactPaths(
        output_tickers_csv=output_tickers_csv,
        output_tickers_ext_csv=output_tickers_ext_csv,
        manifest=manifest,
        fundamentals_out=fundamentals_out,
        adv_cache=adv_cache,
        adv_csv=adv_csv,
        adv_csv_filtered=adv_csv_filtered,
        raw_prices_cache=raw_prices_cache,
        volume_path=volume_path,
        raw_prices_unadj_warmup_cache=raw_prices_unadj_warmup_cache,
        raw_prices_unadj_cache=raw_prices_unadj_cache,
        raw_volume_unadj_cache=raw_volume_unadj_cache,
        checkpoint_path=checkpoint_path,
        run_scoped_outputs_dir=run_scoped_outputs_dir,
    )


def hashed_artifact_siblings(canonical: Path) -> list[Path]:
    parent = canonical.parent
    stem = canonical.stem
    suffix = canonical.suffix
    if not parent.exists():
        return []
    return sorted(parent.glob(f"{stem}.*{suffix}"))


def ensure_updated_at_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "updated_at" not in df.columns:
        df["updated_at"] = float("nan")
    df["updated_at"] = pd.to_numeric(df["updated_at"], errors="coerce")
    return df


def load_fundamentals_store(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    suffix = path.suffix.lower()
    candidates = [path]
    if suffix in {".parquet", ".pq"}:
        candidates.append(path.with_suffix(".pkl"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            suffix_cand = candidate.suffix.lower()
            if suffix_cand in {".parquet", ".pq"}:
                df = pd.read_parquet(candidate)
            elif suffix_cand in {".pkl", ".pickle"}:
                df = pd.read_pickle(candidate)
            else:
                df = pd.read_csv(candidate, index_col=0)
        except Exception as e:
            logger.warning(
                "Fundamentals-Cache konnte nicht geladen werden (%s): %s", candidate, e
            )
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        df = df.copy()
        df.index = df.index.map(lambda x: str(x).strip())
        if df.index.name is None:
            df.index.name = "ticker"
        df = df.sort_index()
        return ensure_updated_at_column(df)
    return pd.DataFrame()


def save_fundamentals_store(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    used_path = path
    try:
        if suffix in {".parquet", ".pq"}:
            df.to_parquet(path)
            used_path = path
        elif suffix in {".pkl", ".pickle"}:
            df.to_pickle(path)
            used_path = path
        else:
            df.to_csv(path, index=True)
            used_path = path
        logger.info(
            "Fundamentals gespeichert: %s (rows=%d)", used_path, int(df.shape[0])
        )
    except Exception as e:
        if suffix in {".parquet", ".pq"}:
            fallback = path.with_suffix(".pkl")
            logger.warning(
                "Parquet-Write fehlgeschlagen (%s), fallback Pickle: %s", e, fallback
            )
            df.to_pickle(fallback)
            used_path = fallback
            logger.info(
                "Fundamentals gespeichert (Fallback Pickle): %s (rows=%d)",
                fallback,
                int(df.shape[0]),
            )
        else:
            raise
    return used_path


def hash_bytes(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def artifact_targets(
    *,
    hashed: bool,
    data_cfg: dict[str, Any],
    price_bytes: bytes | None = None,
    vol_bytes: bytes | None = None,
) -> tuple[Path, Path, Path | None, Path | None]:
    paths = resolve_artifact_paths(data_cfg=data_cfg)
    canonical_prices = paths.raw_prices_cache
    canonical_vols = paths.volume_path

    if hashed:
        ph = hash_bytes(price_bytes or b"")
        vh = hash_bytes(vol_bytes or b"")
        prices_runs = canonical_prices.with_name(
            f"{canonical_prices.stem}.{ph}{canonical_prices.suffix}"
        )
        vols_runs = canonical_vols.with_name(
            f"{canonical_vols.stem}.{vh}{canonical_vols.suffix}"
        )
        mirror_prices = canonical_prices
        mirror_vols = canonical_vols
    else:
        prices_runs = canonical_prices
        vols_runs = canonical_vols
        mirror_prices = canonical_prices
        mirror_vols = canonical_vols
    return prices_runs, vols_runs, mirror_prices, mirror_vols


def build_event_frame(event_map: dict[str, pd.Series]) -> pd.DataFrame:
    if not event_map:
        return pd.DataFrame()
    normalized: dict[str, pd.Series] = {}
    for sym, series in event_map.items():
        try:
            s = pd.Series(series)
            idx = pd.to_datetime(s.index, errors="coerce")
            if isinstance(idx, pd.DatetimeIndex):
                # Normalize timezone to avoid tz-aware vs tz-naive unions
                idx = (
                    idx.tz_convert("UTC")
                    if idx.tz is not None
                    else idx.tz_localize("UTC")
                ).tz_localize(None)
            s.index = idx
            normalized[str(sym)] = s
        except Exception:
            continue

    df = pd.DataFrame(normalized)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df


def write_event_artifact(
    *,
    name: str,
    df: pd.DataFrame,
    canonical_path: Path,
    hashed: bool,
) -> Path | None:
    if df.empty:
        return None
    runs_dir = Path("runs/data")
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
    if hashed:
        hashed_path = runs_dir / f"{name}.{hash_bytes(payload)}.pkl"
    else:
        hashed_path = runs_dir / f"{name}.pkl"
    hashed_path.parent.mkdir(parents=True, exist_ok=True)
    hashed_path.write_bytes(payload)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if canonical_path.resolve() != hashed_path.resolve():
            _sh.copy2(str(hashed_path), str(canonical_path))
    except Exception:
        _sh.copy2(str(hashed_path), str(canonical_path))
    return hashed_path
