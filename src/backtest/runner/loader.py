from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import backtest.utils.pair_analysis as _pair_analysis
from backtest.utils.pairs import (
    normalize_pairs_input as _normalize_pairs_input_ssot,
)
from backtest.utils.pairs import parse_pair_symbols as _parse_pair_symbols
from backtest.utils.tz import (
    align_ts_to_index,
    assert_ny_frame,
)

logger = logging.getLogger("backtest.loader")
logger.addHandler(logging.NullHandler())

_PANEL_FIELDS = {"close", "adj_close", "open", "high", "low", "volume", "vwap", "price"}


def _select_field_from_panel(df: pd.DataFrame, *, prefer_col: str) -> pd.DataFrame:
    """
    Select one field from a processing-style OHLCV panel.

    The backtest runtime expects MultiIndex columns `(symbol, field)`.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError(
            "Backtest requires a processing-style OHLCV panel with MultiIndex columns "
            "(symbol, field)."
        )

    mi = df.columns
    names = [str(name).strip().lower() if name is not None else "" for name in mi.names]
    fld_level = names.index("field") if "field" in names else (mi.nlevels - 1)
    field_values = {str(v).lower() for v in mi.get_level_values(fld_level).unique()}
    if not field_values.intersection(_PANEL_FIELDS):
        raise ValueError(
            "Backtest panel columns must use MultiIndex layout (symbol, field) "
            "with field on the last level."
        )

    preferred = str(prefer_col or "").strip()
    try:
        out = cast(
            pd.DataFrame,
            df.xs(preferred, axis=1, level=fld_level, drop_level=True),
        )
    except Exception as exc:
        available = sorted({str(v) for v in mi.get_level_values(fld_level).unique()})
        raise KeyError(
            f"Requested field {prefer_col!r} not present in panel. "
            f"Available fields: {available}"
        ) from exc

    out.columns = pd.Index(map(str, out.columns))
    return out


def load_price_panel(path: str | Path) -> pd.DataFrame:
    """
    Load the processing-stage OHLCV panel used by backtest runtime.

    Supported inputs are the official processing artifacts: pickle or parquet.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Price file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(p)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Price panel pickle must contain a DataFrame: {p}")
        df = obj.copy()
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(
            f"Unsupported price panel type: {p.suffix} (expected .pkl/.pickle/.parquet)"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    assert_ny_frame(df, context=f"load_price_panel:{p.name}")
    _select_field_from_panel(df, prefer_col="close")
    return df


def select_field_from_panel(df: pd.DataFrame, *, field: str = "close") -> pd.DataFrame:
    """Public wrapper for selecting one field from a processing-style OHLCV panel."""
    assert_ny_frame(df, context="select_field_from_panel")
    return _select_field_from_panel(df, prefer_col=field)


def load_filtered_pairs(path: str | Path) -> dict[str, dict[str, str]]:
    """
    Load the Analysis pair-selection artifact.

    The supported contract is the official Analysis output: pickle or CSV with a
    `pair` column containing strings such as `AAA-BBB`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pairs path not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        obj = pd.read_pickle(p)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Pairs pickle must contain a DataFrame: {p}")
        df = obj.copy()
    elif suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported pairs file format (expected .pkl/.pickle/.csv)")

    if df.empty:
        return {}
    if "pair" not in df.columns:
        raise ValueError("Filtered pairs file must contain a 'pair' column.")

    out: dict[str, dict[str, str]] = {}
    for pair_val in df["pair"]:
        pair = str(pair_val).strip()
        if not pair:
            continue
        t1, t2 = _parse_pair_symbols(pair, upper=False)
        if t1 is None or t2 is None:
            raise ValueError(f"Invalid pair string in filtered pairs file: {pair!r}")
        out[pair] = {"t1": t1, "t2": t2}
    return out


def _to_positive_finite_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def load_adv_map(path: str | Path) -> dict[str, float]:
    """
    Load the Processing ADV artifact (`adv_map.pkl`).

    The supported contract is the official Processing output: a pickle mapping
    `{symbol -> float}` or `{symbol -> {'adv': float, ...}}`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ADV map path not found: {p}")
    if p.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("Unsupported ADV map format (expected .pkl/.pickle)")

    obj = pd.read_pickle(p)
    if not isinstance(obj, Mapping):
        raise ValueError("ADV map pickle must contain a mapping.")

    out: dict[str, float] = {}
    for key, value in obj.items():
        symbol = str(key).strip()
        if not symbol:
            continue
        adv: float | None
        if isinstance(value, Mapping):
            adv = _to_positive_finite_float(cast(Mapping[str, Any], value).get("adv"))
        else:
            adv = _to_positive_finite_float(value)
        if adv is not None:
            out[symbol] = adv

    if not out:
        raise ValueError(
            "ADV map pickle did not contain any valid positive ADV values."
        )
    return out


def prepare_pairs_data(  # noqa: C901
    prices: pd.DataFrame,
    pairs: Mapping[str, Any],
    adv_map: dict[str, float] | None = None,
    disable_prefilter: bool = False,
    *,
    prefilter_range: tuple[Any, Any] | None = None,
    pair_prefilter_cfg: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build per-pair runtime structures on top of the cleaned execution-price matrix.

    Processing already owns panel cleaning, alignment, and corporate-action handling.
    This function only prepares per-pair runtime inputs and optional train-window
    cointegration metadata.
    """
    assert_ny_frame(prices, context="prepare_pairs_data(prices)")

    prices_float = prices.apply(pd.to_numeric, errors="coerce")
    idx = cast(pd.DatetimeIndex, prices_float.index)
    idx_prefilter = idx
    if prefilter_range is not None:
        try:
            start_raw, end_raw = prefilter_range
            t0 = align_ts_to_index(start_raw, idx)
            t1_end = align_ts_to_index(end_raw, idx)
            idx_prefilter = idx[(idx >= t0) & (idx <= t1_end)]
        except Exception:
            idx_prefilter = idx
    if idx_prefilter.empty and prefilter_range is not None:
        logger.warning("prepare_pairs_data: prefilter_range produced empty window.")
        idx_prefilter = idx

    pf_cfg = dict(pair_prefilter_cfg or {})
    coint_alpha = float(pf_cfg.get("coint_alpha", 0.05))
    min_obs = max(2, int(pf_cfg.get("min_obs", 30)))
    half_life_cfg = _pair_analysis.resolve_half_life_cfg(pf_cfg.get("half_life"))

    pairs_norm = _normalize_pairs_input_ssot(pairs, upper=False)
    needed: set[str] = set()
    for meta in pairs_norm.values():
        t1 = meta.get("t1")
        t2 = meta.get("t2")
        if t1:
            needed.add(t1)
        if t2:
            needed.add(t2)
    needed_cols = sorted((t for t in needed if t in prices_float.columns), key=str)
    prices_float = prices_float.loc[:, needed_cols]

    def _fetch_metric(src: dict[str, float] | None, key: str | None) -> float:
        if not src or key is None:
            return float("nan")
        try:
            val = src.get(key)
            if val is None:
                return float("nan")
            f = float(val)
            return f if np.isfinite(f) else float("nan")
        except Exception:
            return float("nan")

    out: dict[str, dict[str, Any]] = {}
    filtered_reasons: dict[str, str] = {}

    for pair in sorted(pairs_norm.keys(), key=str):
        meta = pairs_norm[pair]
        t1 = meta.get("t1")
        t2 = meta.get("t2")
        if t1 is None or t2 is None:
            filtered_reasons[pair] = f"missing_ticker({t1},{t2})"
            continue
        if t1 not in prices_float.columns or t2 not in prices_float.columns:
            filtered_reasons[pair] = f"missing_ticker({t1},{t2})"
            continue

        s1 = prices_float[t1]
        s2 = prices_float[t2]
        s1_pref = s1.reindex(idx_prefilter)
        s2_pref = s2.reindex(idx_prefilter)

        coint_diag: dict[str, Any] | None = None
        if not disable_prefilter:
            try:
                coint_diag = _pair_analysis.evaluate_pair_cointegration(
                    pd.DataFrame({"y": s1_pref, "x": s2_pref}),
                    coint_alpha=coint_alpha,
                    min_obs=min_obs,
                    half_life_cfg=half_life_cfg,
                )
                if not bool(coint_diag.get("passed", False)):
                    filtered_reasons[pair] = str(
                        coint_diag.get("reject_reason") or "prefilter_failed"
                    )
                    continue
            except Exception as exc:
                filtered_reasons[pair] = f"prefilter_error:{exc}"
                continue
        else:
            beta_hat, beta_reason = (
                _pair_analysis.estimate_beta_ols_with_intercept_details(
                    s1_pref, s2_pref
                )
            )
            if beta_hat is None:
                filtered_reasons[pair] = str(beta_reason or "beta_estimation_failed")
                continue

        item_meta: dict[str, Any] = {
            "t1": t1,
            "t2": t2,
            "adv_t1": _fetch_metric(adv_map, t1),
            "adv_t2": _fetch_metric(adv_map, t2),
        }
        if coint_diag is not None:
            item_meta["cointegration"] = dict(coint_diag)

        item: dict[str, Any] = {
            "t1_price": s1,
            "t2_price": s2,
            "meta": item_meta,
            "prices": pd.DataFrame({"y": s1, "x": s2}),
        }

        out[pair] = item

    logger.info(
        "prepare_pairs_data: retained %d pairs (from %d)", len(out), len(pairs_norm)
    )
    if filtered_reasons:
        counts = Counter(filtered_reasons.values())
        logger.info("prepare_pairs_data: filtered_reasons=%s", dict(counts))
    return out


__all__ = [
    "load_price_panel",
    "select_field_from_panel",
    "load_filtered_pairs",
    "load_adv_map",
    "prepare_pairs_data",
]
