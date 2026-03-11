# src/backtest/utils/io.py
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Optional, cast

import pandas as pd

from backtest.reporting.artifacts import (
    ArtifactStore,
    DataFrameParquetSerializer,
    JSONSerializer,
    Stage,
    load_or_build,
)

# SSOT timezone utilities
from backtest.utils.tz import (
    NY_TZ,
    coerce_series_to_tz,
    ensure_index_tz,
    get_ex_tz,
    to_naive_local,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Small, focused helpers
# =============================================================================


def cut_float(d: Mapping[str, float]) -> dict[str, float]:
    """Return a plain-JSONable {str: float} copy (no numpy scalars)."""
    return {str(k): float(v) for k, v in d.items()}


def _safe_write_json(path: Path, obj: Any) -> None:
    """Write compact UTF-8 JSON and create parent dirs. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception as e:  # pragma: no cover
        logger.warning("Write JSON failed (%s): %s", path, e)


def _safe_write_df(path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to .csv/.parquet (fallback to CSV). Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        suf = path.suffix.lower()
        if suf == ".csv":
            df.to_csv(path, index=True)
        elif suf == ".parquet":
            df.to_parquet(path, index=True)
        else:
            df.to_csv(path.with_suffix(".csv"), index=True)
    except Exception as e:  # pragma: no cover
        logger.warning("Write DF failed (%s): %s", path, e)


# =============================================================================
# Artifact cache convenience
# =============================================================================


def load_or_build_df(
    *,
    stage: Stage,
    config: Mapping[str, Any],
    builder: Callable[[], pd.DataFrame],
    store: ArtifactStore | None = None,
    extra: Mapping[str, Any] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for DataFrame artifacts (Parquet by default).

    Example:
        df = load_or_build_df(
            stage=Stage.FEATURES,
            config={"symbol": "AAPL", "day": "2025-01-31", "bar": "1m"},
            builder=lambda: make_features(...),
            extra={"dataset_id": "intraday-v3"},
        )
    """
    return load_or_build(
        stage=stage,
        config=config,
        builder=builder,
        serializer=DataFrameParquetSerializer(),
        store=store,
        extra=extra,
        force_rebuild=force_rebuild,
    )


def load_or_build_json(
    *,
    stage: Stage,
    config: Mapping[str, Any],
    builder: Callable[[], Any],
    store: ArtifactStore | None = None,
    extra: Mapping[str, Any] | None = None,
    force_rebuild: bool = False,
) -> Any:
    """Convenience wrapper for JSON artifacts (reports/manifests)."""
    return load_or_build(
        stage=stage,
        config=config,
        builder=builder,
        serializer=JSONSerializer(),
        store=store,
        extra=extra,
        force_rebuild=force_rebuild,
    )


# =============================================================================
# Domain I/O with explicit timezone policy (SSOT)
# =============================================================================

# Environment override, but default to our project policy (NY_TZ).
_EX_TZ: str = get_ex_tz({}, None, default=NY_TZ)
TZPolicy = Literal["exchange", "naive"]  # type alias for public API


def _normalize_prices_index_awaretz(
    df: pd.DataFrame,
    *,
    ex_tz: str,
    context: str = "exchange",
) -> pd.DataFrame:
    """
    Ensure a DataFrame has a DatetimeIndex localized/converted to ex_tz.

    Contract:
      - If index is tz-naive → treat as local wall clock and tz_localize(ex_tz).
      - If index is tz-aware → tz_convert(ex_tz).
      - No in-place mutation; returns a new DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{context}: expected DatetimeIndex, got {type(df.index).__name__}"
        )

    out = ensure_index_tz(df, ex_tz)
    idx = cast(pd.DatetimeIndex, out.index)
    tzname = getattr(idx.tz, "key", None) or str(idx.tz)
    logger.info("Price index timezone normalized: %s (policy=exchange)", tzname)
    return out


def normalize_prices_for_module(
    obj: pd.DataFrame | pd.Series,
    *,
    policy: TZPolicy = "exchange",
    ex_tz: Optional[str] = None,
    context: str = "",
) -> pd.DataFrame | pd.Series:
    """
    Normalize a Series/DataFrame time index for a given consumer.

    - policy="exchange" (default): return tz-aware America/New_York.
    - policy="naive": return tz-naive view (local wall clock).
    - ex_tz: optional override; if None, falls back to `_EX_TZ`.
    """
    tz = ex_tz or _EX_TZ
    if isinstance(obj, pd.Series):
        # Lift Series to DataFrame for index normalization; drop back to Series.
        df = obj.to_frame()
        df_norm = _normalize_prices_index_awaretz(
            df, ex_tz=tz, context=context or "series"
        )
        return (
            to_naive_local(df_norm[obj.name])
            if policy == "naive"
            else df_norm[obj.name]
        )

    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj.copy()

    df_norm = _normalize_prices_index_awaretz(obj, ex_tz=tz, context=context or "frame")
    return to_naive_local(df_norm) if policy == "naive" else df_norm


def read_prices_frame(
    path: str | os.PathLike[str],
    *,
    cfg: Mapping[str, Any] | None = None,
    policy: TZPolicy = "exchange",
) -> pd.DataFrame:
    """
    Read a *daily* price matrix and normalize the index timezone per SSOT.

    Supports .pkl/.pickle, .parquet, and .csv with a datetime index.
    Never mutates in-place and always returns a copy.

    Notes:
    - If the file holds tz-naive dates, interpret as *exchange-local*
      and localize to America/New_York (not UTC).
    - If the file holds UTC or any tz-aware index, convert to ex_tz.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # Load
    if p.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(p)
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported price file type: {p.suffix}")

    # Derive target tz from cfg, env, or default (NY)
    ex_tz = get_ex_tz(cfg or {}, df, default=_EX_TZ)

    # Enforce project policy for internal use
    df_norm = cast(
        pd.DataFrame,
        normalize_prices_for_module(
            df, policy="exchange", ex_tz=ex_tz, context=f"read:{p.name}"
        ),
    )

    # Provide a naive view only if explicitly requested by the caller
    return df_norm if policy == "exchange" else to_naive_local(df_norm)


def read_intraday_ohlcv(
    symbol: str,
    day: str,
    base_dir: str | None = None,
) -> pd.DataFrame:
    """
    Read intraday OHLCV in a standard layout and convert to exchange tz.

    Expected file schema:
      {base_dir}/{YYYY-MM-DD}/{SYMBOL}.parquet
      columns: ['ts', 'open', 'high', 'low', 'close', 'volume']
      'ts' is UTC nanoseconds (or any tz-aware timestamp).

    Returns:
      DataFrame indexed in exchange timezone (America/New_York by default).
    """
    base = Path(base_dir or "data/intraday")
    p = base / day / f"{symbol}.parquet"
    df = pd.read_parquet(p)

    if "ts" not in df.columns:
        raise KeyError(f"missing 'ts' column in {p}")

    ts = coerce_series_to_tz(
        df["ts"], "UTC", naive_is_utc=True, utc_hint="auto", errors="raise"
    )
    # Convert UTC → exchange tz (SSOT)
    ts_local = coerce_series_to_tz(ts, _EX_TZ)
    out = df.drop(columns=["ts"]).set_index(ts_local).sort_index()
    logger.info(
        "Intraday %s %s loaded: rows=%d | idx.tz=%s", symbol, day, len(out), _EX_TZ
    )
    return out
