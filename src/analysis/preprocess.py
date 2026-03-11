"""Time handling, schema normalization, and price loading utilities."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from analysis.constants import PRICE_COL_CANDIDATES
from analysis.logging_config import logger
from analysis.utils import _guard


def _is_parseable_timestamp_column(
    values: pd.Series, *, ts_unit: Optional[str] = None
) -> bool:
    """
    Probe timestamp parseability without surfacing pandas inference warnings.

    This is only used for auto-detecting a timestamp column. The actual selected
    timestamp column is still parsed normally afterwards.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format, so each element will be parsed individually.*",
                category=UserWarning,
            )
            pd.to_datetime(values, utc=False, errors="raise", unit=ts_unit)
        return True
    except Exception:
        return False


def ensure_utc_index(
    df: pd.DataFrame, *, int_unit: Optional[str] = None
) -> pd.DataFrame:
    """
    Return df with a parsed DatetimeIndex, sorted and duplicate-safe (keep last).

    Despite the legacy name, this function preserves the input timezone if present.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            idx_arr = np.asarray(df.index)
            if np.issubdtype(idx_arr.dtype, np.integer) and int_unit is None:
                raise ValueError(
                    "Integer index detected; provide data.timestamp_unit (e.g., 's','ms')."
                )
            df.index = pd.to_datetime(
                df.index, utc=False, errors="raise", unit=int_unit
            )
        except Exception as e:
            raise ValueError(
                "Failed to parse index to DatetimeIndex. "
                "Ensure ISO8601 timestamps or provide timestamp_unit for integer epochs."
            ) from e
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if not df.index.is_unique:
        dup = int(df.index.duplicated(keep="last").sum())
        if dup:
            logger.warning("Dropping %d duplicate timestamps (keep='last')", dup)
        df = df[~df.index.duplicated(keep="last")]
    return df


def _dedup_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to str and drop duplicate labels by keeping the first column."""
    cols_str = pd.Index(map(str, df.columns))
    if cols_str.is_unique:
        df.columns = cols_str
        return df
    mask = ~cols_str.duplicated(keep="first")
    if (~mask).any():
        dups = cols_str[~mask].unique().tolist()
        logger.warning("Dropping duplicate columns after str-cast: %s", dups[:10])
    df = df.loc[:, mask]
    df.columns = cols_str[mask]
    return df


def select_price_columns(
    df: pd.DataFrame, preferred_field: Optional[str] = None
) -> pd.DataFrame:
    """Return a wide price matrix with string asset columns."""
    if isinstance(df.columns, pd.MultiIndex):
        last = pd.Index(df.columns.get_level_values(-1)).map(lambda s: str(s).lower())
        available_fields = pd.Index(last.unique())

        field_to_use: Optional[str] = None
        if preferred_field:
            pref = preferred_field.lower()
            if pref in available_fields:
                field_to_use = pref
            else:
                logger.warning(
                    "Preferred price field '%s' missing; falling back to default candidates.",
                    preferred_field,
                )
        if field_to_use is None:
            for cand in PRICE_COL_CANDIDATES:
                if cand in available_fields:
                    field_to_use = cand
                    break

        _guard(
            field_to_use is not None,
            "MultiIndex without price field (close/price/etc).",
        )
        mask_arr = np.asarray(last == field_to_use, dtype=bool)
        out = df.loc[:, mask_arr].copy()
        out.columns = out.columns.droplevel(-1)
        if not out.columns.is_unique:
            out = out.T.groupby(level=0, sort=False).last().T
            logger.info(
                "Aggregated duplicate asset columns after droplevel via groupby(...).last()"
            )
        out.columns = pd.Index(map(str, out.columns))
        return out

    cols = set(map(str, df.columns))
    if {"asset_id", "ticker"}.intersection(cols) and any(
        str(c).lower() in PRICE_COL_CANDIDATES for c in cols
    ):
        asset_col = "asset_id" if "asset_id" in cols else "ticker"
        lower_map = {str(c).lower(): c for c in df.columns}
        price_col = next(
            (lower_map[pref] for pref in PRICE_COL_CANDIDATES if pref in lower_map),
            None,
        )
        _guard(price_col is not None, "No recognized price column in long-form input.")

        idx_any = cast(Any, df.index)
        wide = df.pivot_table(
            index=idx_any,
            columns=asset_col,
            values=price_col,
            aggfunc="last",
            observed=True,
            sort=False,
        )
        wide = _dedup_str_columns(wide)
        return wide

    non_price_tokens = ("volume", "vol", "turnover", "adv", "shares", "count", "qty")
    maybe_price = [
        c
        for c in df.columns
        if not any(tok in str(c).lower() for tok in non_price_tokens)
    ]
    numeric = df[maybe_price].select_dtypes(include=[np.number])
    _guard(numeric.shape[1] > 0, "No numeric price columns found.")
    numeric = _dedup_str_columns(numeric)
    return numeric


def load_filled_data(
    path: str | Path,
    ts_col_cfg: Optional[str] = None,
    ts_unit: Optional[str] = None,
    field_preference: Optional[str] = None,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Price file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        with p.open("rb") as f:
            payload = pickle.load(f)
        df = (
            payload.copy()
            if isinstance(payload, pd.DataFrame)
            else pd.DataFrame(payload)
        )
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        ts_col = ts_col_cfg or next(
            (
                c
                for c in df.columns
                if str(c).lower() in ("timestamp", "ts", "date", "datetime")
            ),
            None,
        )
        if ts_col is None and df.columns.size >= 1:
            candidate = df.columns[0]
            if _is_parseable_timestamp_column(df[candidate], ts_unit=ts_unit):
                ts_col = candidate
                logger.debug(
                    "Auto-detected timestamp column '%s' for CSV input.", candidate
                )
            else:
                ts_col = None
        _guard(
            ts_col is not None,
            "CSV/TSV requires a timestamp column; set data.timestamp_column",
        )
        df[ts_col] = pd.to_datetime(df[ts_col], utc=False, errors="raise", unit=ts_unit)
        df = df.set_index(ts_col).sort_index()
    elif suffix in (".feather",):
        read = pd.read_feather(p)
        _guard(read.shape[1] > 0, "Feather file has no columns.")
        ts_col = ts_col_cfg or next(
            (
                c
                for c in read.columns
                if str(c).lower() in ("timestamp", "ts", "date", "datetime")
            ),
            None,
        )
        if ts_col is None and read.columns.size >= 1:
            candidate = read.columns[0]
            if _is_parseable_timestamp_column(read[candidate], ts_unit=ts_unit):
                ts_col = candidate
                logger.debug(
                    "Auto-detected timestamp column '%s' for Feather input.", candidate
                )
            else:
                ts_col = None
        _guard(
            ts_col is not None,
            "Feather must include a timestamp column (e.g., 'timestamp'); set data.timestamp_column",
        )
        read[ts_col] = pd.to_datetime(
            read[ts_col], utc=False, errors="raise", unit=ts_unit
        )
        df = read.set_index(ts_col)
    else:
        raise ValueError(f"Unsupported file type for price input: {p}")

    if not isinstance(df.index, pd.DatetimeIndex):
        candidates = ("timestamp", "ts", "date", "datetime")
        ts_col = ts_col_cfg or next(
            (c for c in df.columns if str(c).lower() in candidates), None
        )
        if ts_col is not None:
            df[ts_col] = pd.to_datetime(
                df[ts_col], utc=False, errors="raise", unit=ts_unit
            )
            df = df.set_index(ts_col)
    df = ensure_utc_index(df, int_unit=ts_unit)
    df = select_price_columns(df, preferred_field=field_preference)
    field_tag = field_preference.lower() if field_preference else "auto"
    logger.info(
        "Loaded filled prices: %s [field=%s] (rows=%d, cols=%d)",
        p,
        field_tag,
        df.shape[0],
        df.shape[1],
    )
    return df


__all__ = [
    "ensure_utc_index",
    "select_price_columns",
    "load_filled_data",
    "_dedup_str_columns",
    "_is_parseable_timestamp_column",
]
