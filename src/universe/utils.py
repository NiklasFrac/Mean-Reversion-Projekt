from __future__ import annotations

import datetime as dt
import os
import pickle
import secrets
import threading
import time
from decimal import Decimal
from numbers import Number
from pathlib import Path
from typing import Any, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from universe.coercion import clamp01
from universe.fs_atomic import atomic_replace_or_raise


def _series_is_numeric(series: pd.Series) -> bool:
    if is_numeric_dtype(series):
        return True
    if series.dtype == "object":
        sample = series.dropna().head(5)
        if sample.empty:
            return False
        return all(
            isinstance(val, Number) or isinstance(val, Decimal) for val in sample
        )
    return False


def _enforce_canary(
    df_filtered: pd.DataFrame,
    *,
    min_valid_tickers: int,
    max_nan_pct: float,
) -> dict[str, float | int | str | None]:
    """
    Canonical canary:
    - size check: rows >= min_valid_tickers
    - NaN check enabled if 0.0 <= max_nan_pct <= 1.0
    - core fields ('price', 'market_cap', 'volume') are always checked first
      when present so core NaNs cannot be masked by a complete float column
    - optional float coverage check when 'float_pct' exists
    - fallback numeric global share when neither core nor float checks apply

    Returns a dict with check metadata for the manifest.
    Raises RuntimeError on violation.
    """
    n = int(df_filtered.shape[0])
    if min_valid_tickers > 0 and n < int(min_valid_tickers):
        raise RuntimeError(
            f"Canary: only {n} valid tickers < min_valid_tickers={int(min_valid_tickers)}"
        )

    basis = "disabled"
    metric = "disabled"
    nan_share: float | None = None

    if 0.0 <= float(max_nan_pct) <= 1.0:
        checks: list[tuple[str, str, float]] = []

        core_cols = [
            c for c in ("price", "market_cap", "volume") if c in df_filtered.columns
        ]
        if core_cols:
            core = df_filtered[core_cols].apply(pd.to_numeric, errors="coerce")
            core_share = (
                float(core.isna().sum().sum() / core.size) if core.size else 1.0
            )
            checks.append(("core_fields", "global_share", core_share))

        if "float_pct" in df_filtered.columns:
            s = pd.to_numeric(df_filtered["float_pct"], errors="coerce")
            checks.append(("float_pct", "col_mean", float(s.isna().mean())))

        if not checks:
            numeric_cols = [
                col
                for col in df_filtered.columns
                if _series_is_numeric(df_filtered[col])
            ]
            if numeric_cols:
                num = df_filtered[numeric_cols].apply(pd.to_numeric, errors="coerce")
                checks.append(
                    (
                        "numeric",
                        "global_share",
                        float(num.isna().sum().sum() / num.size) if num.size else 1.0,
                    )
                )
            else:
                checks.append(("none", "empty", 1.0))

        failing = [c for c in checks if c[2] > float(max_nan_pct)]

        primary_basis, primary_metric, primary_share = checks[0]
        if len(checks) == 1:
            basis = primary_basis
            metric = primary_metric
        else:
            basis = ",".join(dict.fromkeys(name for name, _, _ in checks))
            metric = ",".join(dict.fromkeys(name for _, name, _ in checks))
        nan_share = float(primary_share)

        if failing:
            detail = ", ".join(
                f"{name}={share * 100.0:.1f}%" for name, _, share in failing
            )
            raise RuntimeError(
                f"Canary: NaN share {detail} "
                f"> max_nan_pct={float(max_nan_pct) * 100.0:.1f}% "
                f"(basis={basis}, metric={metric})"
            )

    return {
        "n_rows": n,
        "min_valid_tickers": int(min_valid_tickers),
        "max_nan_pct": float(max_nan_pct),
        "nan_basis_used": basis,
        "nan_metric_used": metric,
        "nan_share_checked": nan_share,
    }


def _ensure_updated_at_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "updated_at" not in df.columns:
        df["updated_at"] = float("nan")
    df["updated_at"] = pd.to_numeric(df["updated_at"], errors="coerce")
    return df


def _ensure_not_cancelled(stop_event: threading.Event | None) -> None:
    """
    Raise KeyboardInterrupt if a graceful-shutdown signal was received.
    Keeps long-running stages from continuing silently after Ctrl+C.
    """
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt("Universe runner cancelled via signal.")


def _sha1(data: Union[bytes, str, Path], *, chunk_size: int = 1 << 20) -> str | None:
    """
    SHA-1 hasher for either raw bytes or a file path. Returns hex digest.
    For file paths returns None on error (file missing, permissions, etc.).
    """
    import hashlib

    h = hashlib.sha1()
    if isinstance(data, (bytes, bytearray)):
        h.update(data)
        return h.hexdigest()

    try:
        p = Path(data)
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _generate_run_id(cfg_hash: str | None) -> str:
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    token = secrets.token_hex(3).upper()
    base = (cfg_hash or "NOHASH")[:6].upper()
    return f"RUN-{ts}-{base}-{token}"


def _clip01(x: float) -> float:
    return clamp01(x, default=0.0)


def _atomic_write_pickle(obj: Any, path: Path, *, attempts: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        atomic_replace_or_raise(
            tmp,
            path,
            attempts=attempts,
            replace_fn=os.replace,
            sleep_fn=time.sleep,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Atomic pickle write failed for {tmp} -> {path}: {exc}"
        ) from exc
