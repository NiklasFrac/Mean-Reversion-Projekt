from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from backtest.utils.common.pairs import (
    normalize_pairs_input as _normalize_pairs_input_ssot,
)
from backtest.utils.common.prices import (
    as_price_map as _as_price_map,
)
from backtest.utils.common.prices import (
    price_at_or_prior as _price_at_or_prior,
)
from backtest.utils.alpha import (
    evaluate_pair_cointegration,
    pair_prefilter,
    resolve_half_life_cfg,
)
from backtest.utils import strategy as _strategy_helpers
from backtest.utils.tz import (
    NY_TZ,
    align_ts_to_index,
    ensure_dtindex_tz,
    ensure_index_tz,
    to_naive_local,
)

logger = logging.getLogger("data_loader")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# =============================================================================
#                              TIMEZONE POLICY
# =============================================================================
_EX_TZ = NY_TZ  # single exchange tz for loader
_PANEL_FIELDS = {"close", "adj_close", "open", "high", "low", "volume", "vwap", "price"}


def _idx_to_ex_tz(idx: pd.DatetimeIndex, tz: str = _EX_TZ) -> pd.DatetimeIndex:
    """Return a DatetimeIndex localized/converted to the exchange tz."""
    df = pd.DataFrame(index=idx)
    df2 = ensure_index_tz(df, tz)
    return cast(pd.DatetimeIndex, df2.index)


def _series_to_ex_tz(s: pd.Series, tz: str = _EX_TZ) -> pd.Series:
    """Return a Series whose index is localized/converted to the exchange tz."""
    return ensure_index_tz(s, tz)


def _coerce_ts_like_index(ts: Any, idx: pd.DatetimeIndex) -> pd.Timestamp:
    return align_ts_to_index(ts, idx)


def _prefilter_ok(
    y: pd.Series,
    x: pd.Series,
    *,
    coint_alpha: float = 0.05,
    min_obs: int = 30,
) -> bool:
    """Run the Engle-Granger pair prefilter on a standard y/x DataFrame."""
    df = pd.DataFrame({"y": y, "x": x})
    try:
        return bool(
            pair_prefilter(df, coint_alpha=float(coint_alpha), min_obs=int(min_obs))
        )
    except TypeError:
        try:
            return bool(pair_prefilter(df))
        except Exception as e:
            logger.debug(
                "pair_prefilter failed; treating as pass-through (ok). Error: %s", e
            )
            return True  # prefilter is advisory-never hard-fail
    except Exception as e:
        logger.debug(
            "pair_prefilter failed; treating as pass-through (ok). Error: %s", e
        )
        return True  # prefilter is advisory-never hard-fail


# =============================================================================
#                   LIGHTWEIGHT PRICE UTILS (MAPPINGS & SNAPSHOT)
# =============================================================================


def as_price_mapping(
    price_data: pd.DataFrame | Mapping[str, pd.Series | pd.DataFrame],
    *,
    prefer_col: str = "close",
    coerce_numeric: bool = True,
) -> dict[str, pd.Series]:
    """
    Normalize inputs into Mapping[str -> Series] while preserving the existing index timezone.
    """
    return _as_price_map(
        price_data,
        prefer_col=prefer_col,
        coerce_numeric=coerce_numeric,
    )


def series_price_at(
    price_data: pd.Series | Mapping[str, pd.Series | pd.DataFrame],
    symbol: str,
    ts: pd.Timestamp,
    *,
    prefer_col: str = "close",
) -> float | None:
    """
    Price at-or-prior to `ts` (comparison in **America/New_York**).
    """
    if isinstance(price_data, pd.Series):
        s: pd.Series | None = price_data
    elif isinstance(price_data, Mapping):
        v = price_data.get(symbol)
        if v is None:
            return None
        if isinstance(v, pd.DataFrame):
            series = v[prefer_col] if prefer_col in v.columns else v.squeeze()
            s = series if isinstance(series, pd.Series) else None
        else:
            s = v if isinstance(v, pd.Series) else None
    else:
        s = None
    if s is None or len(s) == 0:
        return None

    s = _series_to_ex_tz(pd.to_numeric(s, errors="coerce")).sort_index()
    return _price_at_or_prior(s, ts, allow_zero=False)


# =============================================================================
#                                  IO: PRICES
# =============================================================================


def _select_field_from_panel(df: pd.DataFrame, *, prefer_col: str) -> pd.DataFrame:
    """
    Select a single field from a panel-like DataFrame.

    Supported layouts:
      - MultiIndex columns: (symbol, field)
      - Flat wide price matrices are passed through unchanged
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        mi = df.columns
        if mi.nlevels < 2:
            raise ValueError(
                "Backtest panel columns must use a MultiIndex layout (symbol, field)."
            )
        names = [
            str(name).strip().lower() if name is not None else "" for name in mi.names
        ]
        fld_level = names.index("field") if "field" in names else (mi.nlevels - 1)
        field_values = {str(v).lower() for v in mi.get_level_values(fld_level).unique()}
        if not field_values.intersection(_PANEL_FIELDS):
            raise ValueError(
                "Backtest panel columns must use MultiIndex layout (symbol, field) "
                "with field on the last level."
            )

        def _xs(field: str) -> pd.DataFrame | None:
            try:
                return cast(
                    pd.DataFrame, df.xs(field, axis=1, level=fld_level, drop_level=True)
                )
            except Exception:
                return None

        preferred = str(prefer_col or "").strip()
        if preferred:
            out = _xs(preferred)
            if out is not None:
                out.columns = pd.Index(map(str, out.columns))
                return out

        for fallback in ("adj_close", "close", "price", "vwap"):
            out = _xs(fallback)
            if out is not None:
                out.columns = pd.Index(map(str, out.columns))
                return out

        available = sorted({str(v) for v in mi.get_level_values(fld_level).unique()})
        raise KeyError(
            f"Requested field {prefer_col!r} not present in panel. "
            f"Available fields: {available}"
        )

    return df


def load_price_data(
    path: str | Path | pd.DataFrame,
    *,
    apply_corporate_actions: bool = False,
    prefer_col: str = "close",
    required_symbols: list[str] | None = None,
    coerce_timezone: str = "exchange",
) -> pd.DataFrame:
    """
    Load a wide price matrix from disk (or accept a DataFrame) and normalize:
      - column selection (panel -> field)
      - optional symbol pruning
      - timezone coercion: 'exchange' | 'utc' | 'naive' | 'keep'
    """
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Price file not found: {p}")
        suf = p.suffix.lower()
        if suf in (".pkl", ".pickle"):
            obj = pd.read_pickle(p)
            df = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        elif suf in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif suf in (".csv", ".tsv"):
            sep = "\t" if suf == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported price file type: {p.suffix}")

    if apply_corporate_actions:
        # CA should already be handled upstream (universe/processing). Keep flag for compat only.
        logger.warning(
            "apply_corporate_actions requested, but this loader does not apply CA; ignoring."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]

    # Select preferred field if this is a panel-like frame.
    df = _select_field_from_panel(df, prefer_col=prefer_col)

    # Optional symbol pruning.
    if required_symbols:
        want = {str(s).strip() for s in required_symbols if str(s).strip()}
        if want:
            cols = [c for c in map(str, df.columns) if c in want]
            df = df.loc[:, cols]

    # Timezone coercion.
    pol = str(coerce_timezone or "exchange").strip().lower()
    ex_tz = _EX_TZ
    if pol in {"exchange", "ny", "new_york"}:
        df = ensure_index_tz(df, ex_tz)
    elif pol in {"utc"}:
        df2 = ensure_index_tz(df, ex_tz)
        df2 = df2.copy()
        df2.index = ensure_dtindex_tz(cast(pd.DatetimeIndex, df2.index), "UTC")
        df = df2
    elif pol in {"naive", "local_naive"}:
        df2 = ensure_index_tz(df, ex_tz)
        df2 = df2.copy()
        df2.index = to_naive_local(cast(pd.DatetimeIndex, df2.index))
        df = df2
    elif pol in {"keep", "none", "off"}:
        pass
    else:
        logger.warning("Unknown coerce_timezone=%s; using 'exchange'.", pol)
        df = ensure_index_tz(df, ex_tz)

    # Optional daily exchange-calendar alignment.
    # NOTE: calendar alignment is performed upstream in processing; loader must not
    # reindex again (avoids silent grid drift / extra NaNs).

    return df


def load_price_panel(
    path: str | Path | pd.DataFrame,
    *,
    coerce_timezone: str = "exchange",
) -> pd.DataFrame:
    """
    Load a panel-like OHLCV DataFrame (MultiIndex columns preferred) and normalize:
      - DatetimeIndex + sorting + dedup
      - timezone coercion: 'exchange' | 'utc' | 'naive' | 'keep'

    Unlike `load_price_data`, this keeps *all* fields (close/high/low/open/volume).
    Backtest consumes the processing-stage panel contract directly: MultiIndex
    columns `(symbol, field)` are required.
    """
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Price file not found: {p}")
        suf = p.suffix.lower()
        if suf in (".pkl", ".pickle"):
            obj = pd.read_pickle(p)
            df = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        elif suf in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        elif suf in (".csv", ".tsv"):
            sep = "\t" if suf == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported price file type: {p.suffix}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError(
            "Backtest requires a processing-style OHLCV panel with MultiIndex columns "
            "(symbol, field)."
        )
    _select_field_from_panel(df, prefer_col="close")

    # Timezone coercion.
    pol = str(coerce_timezone or "exchange").strip().lower()
    ex_tz = _EX_TZ
    if pol in {"exchange", "ny", "new_york"}:
        df = ensure_index_tz(df, ex_tz)
    elif pol in {"utc"}:
        df2 = ensure_index_tz(df, ex_tz)
        df2 = df2.copy()
        df2.index = ensure_dtindex_tz(cast(pd.DatetimeIndex, df2.index), "UTC")
        df = df2
    elif pol in {"naive", "local_naive"}:
        df2 = ensure_index_tz(df, ex_tz)
        df2 = df2.copy()
        df2.index = to_naive_local(cast(pd.DatetimeIndex, df2.index))
        df = df2
    elif pol in {"keep", "none", "off"}:
        pass
    else:
        logger.warning("Unknown coerce_timezone=%s; using 'exchange'.", pol)
        df = ensure_index_tz(df, ex_tz)

    # Optional daily exchange-calendar alignment.
    # NOTE: calendar alignment is performed upstream in processing; loader must not
    # reindex again (avoids silent grid drift / extra NaNs).

    return df


def select_field_from_panel(df: pd.DataFrame, *, field: str = "close") -> pd.DataFrame:
    """Public wrapper for selecting one field from a panel-like frame (see `_select_field_from_panel`)."""
    return _select_field_from_panel(df, prefer_col=field)


# =============================================================================
#                                  IO: PAIRS
# =============================================================================


def _normalize_pairs_input(obj: Any) -> dict[str, dict[str, str]]:
    # Preserve loader behavior: keep symbol case as provided (no uppercasing).
    return _normalize_pairs_input_ssot(obj, upper=False)


def load_filtered_pairs(path: str | Path | dict[str, Any] | Iterable) -> dict[str, Any]:
    """
    Load pair definitions from file or in-memory object, in a forgiving way.
    """
    if isinstance(path, (dict, list, tuple)):
        return _normalize_pairs_input(path)

    p = Path(cast(Any, path))
    if not p.exists():
        raise FileNotFoundError(f"Pairs path not found: {p}")

    def _from_df(df: pd.DataFrame) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if {"t1", "t2"}.issubset(df.columns):
            for _, r in df.iterrows():
                k = r.get("pair") or f"{r['t1']}-{r['t2']}"
                out[str(k)] = {"t1": str(r["t1"]).strip(), "t2": str(r["t2"]).strip()}
            return out
        if {"y", "x"}.issubset(df.columns):
            for _, r in df.iterrows():
                k = r.get("pair") or f"{r['y']}-{r['x']}"
                out[str(k)] = {"t1": str(r["y"]).strip(), "t2": str(r["x"]).strip()}
            return out
        if "pair" in df.columns:
            for _, r in df.iterrows():
                pstr = str(r["pair"]).strip()
                for sep in (" - ", "-", " / ", "/", " "):
                    if sep in pstr:
                        a, b = pstr.split(sep, 1)
                        out[pstr] = {"t1": a.strip(), "t2": b.strip()}
                        break
            return out
        if df.shape[1] >= 2:
            a, b = df.columns[:2]
            for _, r in df.iterrows():
                out[f"{r[a]}-{r[b]}"] = {
                    "t1": str(r[a]).strip(),
                    "t2": str(r[b]).strip(),
                }
        return out

    if p.suffix.lower() in (".pkl", ".pickle"):
        obj = pd.read_pickle(p)
        if isinstance(obj, (dict, list, tuple)):
            return _normalize_pairs_input(obj)
        try:
            df = pd.DataFrame(obj)
            return _from_df(df)
        except Exception as e:
            raise ValueError(f"Unsupported pickle content for pairs file: {e}")
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        return _from_df(df)
    else:
        raise ValueError("Unsupported pairs file format (use .pkl/.csv)")


# =============================================================================
#                               ADV Map (external)
# =============================================================================


def _resolve_colcase(like: Mapping[str, str], candidates: Sequence[str]) -> str | None:
    """Find a column name case-insensitively (with synonyms)."""
    for cand in candidates:
        if cand in like:
            return like[cand]
    return None


_ADV_KEYS: tuple[str, ...] = (
    "adv",
    "adv_usd",
    "adv_currency",
    "avg_dollar_volume",
    "dollar_volume",
    "dollar_adv_hist",
    "turnover",
    "avg_turnover",
)


def _to_positive_finite_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def _extract_adv_from_mapping_payload(payload: Mapping[str, Any]) -> float | None:
    for key in _ADV_KEYS:
        if key in payload:
            return _to_positive_finite_float(payload[key])
    return None


def _adv_dict_from_mapping(obj: Mapping[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in obj.items():
        sym = str(k).strip()
        if not sym:
            continue
        if isinstance(v, Mapping):
            adv = _extract_adv_from_mapping_payload(cast(Mapping[str, Any], v))
        else:
            adv = _to_positive_finite_float(v)
        if adv is not None:
            out[sym] = adv
    return out


def _adv_dict_from_series(obj: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in obj.items():
        sym = str(k).strip()
        if not sym:
            continue
        if isinstance(v, Mapping):
            adv = _extract_adv_from_mapping_payload(cast(Mapping[str, Any], v))
        else:
            adv = _to_positive_finite_float(v)
        if adv is not None:
            out[sym] = adv
    return out


def load_adv_map(path: str | Path) -> dict[str, float]:
    """
    Load an ADV table and return {symbol -> adv}.

    This expects a simple tabular file produced by the processing/universe
    stages, with at least a symbol column and one numeric ADV column.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ADV map path not found: {p}")

    def _normalize_df(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=["symbol", "adv"])
        d = df_in.copy()
        # Be defensive: CSVs are sometimes exported with a different delimiter (e.g. ';')
        # which pandas then reads as a single combined column name like "ticker;adv_usd;...".
        # Also tolerate non-string column labels.
        lc: dict[str, str] = {}
        for c in d.columns:
            key = str(c).strip().lower()
            if key:
                lc[key] = str(c)

        # Common pickle/schema case: a "wide" dataframe with symbols as columns and metrics as index,
        # e.g. from pd.DataFrame({"A":{"adv":...,"last_price":...}, ...}).
        if (
            ("adv" not in lc)
            and isinstance(d.index, pd.Index)
            and d.index.size
            and d.columns.size > 1
        ):
            idx_lc = [str(x).strip().lower() for x in list(d.index)]
            if "adv" in idx_lc and all(str(c).strip() for c in d.columns):
                row = d.loc[d.index[idx_lc.index("adv")]]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                wide = pd.DataFrame(
                    {
                        "symbol": row.index.astype(str),
                        "adv": pd.to_numeric(row.values, errors="coerce"),
                    }
                )
                return _normalize_df(wide)

        sym_col = _resolve_colcase(lc, ("symbol", "ticker", "secid", "ric", "isin"))
        adv_candidates = (
            "adv_usd",
            "adv_currency",
            "adv",
            "avg_dollar_volume",
            "dollar_volume",
            "dollar_adv_hist",
            "turnover",
            "avg_turnover",
        )
        adv_col = _resolve_colcase(lc, adv_candidates)

        # Another common format: symbols in the index + numeric adv column.
        if sym_col is None and isinstance(d.index, pd.Index) and d.index.size:
            if adv_col is not None and d.index.dtype == object:
                d2 = d.reset_index().rename(columns={"index": "symbol"})
                return _normalize_df(d2)
        if sym_col is None:
            cols = ", ".join(map(repr, list(d.columns)[:25]))
            raise ValueError(
                "ADV map: need 'symbol'/'ticker'/'secid'/'ric'/'isin' column. "
                f"Got columns: [{cols}]"
            )
        if adv_col is None:
            num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            if not num_cols:
                raise ValueError("ADV map: no numeric ADV column found.")
            adv_col = num_cols[0]

        d = d.rename(columns={sym_col: "symbol", adv_col: "adv"})[["symbol", "adv"]]
        d["symbol"] = d["symbol"].astype(str).str.strip()
        d["adv"] = pd.to_numeric(d["adv"], errors="coerce").astype(float)
        d = cast(
            pd.DataFrame,
            d.dropna(subset=["symbol"]).groupby("symbol", as_index=False)["adv"].last(),
        )
        d = d.dropna(subset=["adv"])
        d = d[np.isfinite(d["adv"]) & (d["adv"] > 0)]
        return d

    if p.suffix.lower() == ".csv":
        # Try robust delimiter sniffing first; fallback to default CSV parsing.
        try:
            df = pd.read_csv(p, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(p, memory_map=True)

        # Common "Excel-export" case: semicolon-separated but read without sniffing
        # (or with a header-only file) -> retry a couple of likely delimiters.
        if df.shape[1] <= 1 and df.columns.size == 1:
            header = str(df.columns[0])
            for sep in (";", "\t", "|"):
                if sep in header:
                    try:
                        df = pd.read_csv(p, sep=sep)
                        break
                    except Exception:
                        pass
    elif p.suffix.lower() in (".pkl", ".pickle"):
        obj = pd.read_pickle(p)
        # Processing may persist ADV as a dict: symbol -> float or symbol -> {"adv":..., "last_price":...}
        if isinstance(obj, Mapping):
            return _adv_dict_from_mapping(obj)

        if isinstance(obj, pd.Series):
            return _adv_dict_from_series(obj)

        if isinstance(obj, pd.DataFrame):
            df = obj
        else:
            try:
                df = pd.DataFrame(obj)
            except Exception as e:
                raise ValueError(
                    f"Unsupported pickle content for ADV map: {type(obj).__name__}"
                ) from e
    elif p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported ADV path suffix: {p.suffix}")

    df = _normalize_df(df)
    return {
        str(sym): float(adv)
        for sym, adv in df[["symbol", "adv"]].itertuples(index=False, name=None)
    }


def save_adv_map(adv: Mapping[str, float], path: str | Path) -> None:
    """Save an ADV map as CSV (symbol,adv)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"symbol": list(adv.keys()), "adv": list(adv.values())})
    df.to_csv(p, index=False)


def filter_by_adv(
    df: pd.DataFrame,
    adv: Mapping[str, float] | None,
    *,
    min_adv: float | None = None,
    keep_missing: bool = False,
) -> pd.DataFrame:
    """
    Filter columns of a price matrix by an ADV threshold.
    """
    if adv is None or min_adv is None:
        return df
    adv_ok = {s for s, v in adv.items() if np.isfinite(v) and v >= float(min_adv)}
    cols_keep: list[str] = []
    for c in map(str, df.columns):
        if c in adv_ok:
            cols_keep.append(c)
        elif keep_missing and c not in adv:
            cols_keep.append(c)
    if not cols_keep:
        logger.warning("filter_by_adv: no columns meet min_adv=%.3f", float(min_adv))
        return df.iloc[:, :0]
    return df.loc[:, cols_keep]


# --- Back-compat shim: resolve_borrow_rate -----------------------------------


def resolve_borrow_rate(
    symbol: str,
    day: Any,
    *,
    borrow_ctx: Any = None,
    default_rate_annual: Optional[float] = None,
    day_basis: Optional[int] = None,  # kept for back-compat; currently unused
    availability_df: Optional[pd.DataFrame] = None,  # kept for back-compat
    **kwargs: Any,
) -> float:
    """
    Backwards-compatible borrow-rate resolver; very forgiving.

    This is intentionally lightweight: it does not construct availability
    panels or do any IO on its own, it just interrogates `borrow_ctx`
    / `availability_df` if provided and otherwise falls back to a default.
    """
    try:
        d = pd.Timestamp(day).normalize()
    except Exception:
        d = cast(pd.Timestamp, pd.NaT)

    rate = None
    if borrow_ctx is not None:
        try:
            resolve_fn = getattr(borrow_ctx, "resolve_borrow_rate", None)
            if callable(resolve_fn):
                r = resolve_fn(symbol, d)
                if r is not None:
                    return float(r)
        except Exception:
            pass

        for meth in ("events_for_range", "get_borrow_events"):
            fn = getattr(borrow_ctx, meth, None)
            if callable(fn):
                try:
                    ev = fn([symbol], d, d)
                    if (
                        isinstance(ev, pd.DataFrame)
                        and not ev.empty
                        and "rate_annual" in ev.columns
                    ):
                        val = pd.to_numeric(ev["rate_annual"], errors="coerce").dropna()
                        if not val.empty:
                            return float(val.iloc[-1])
                except Exception:
                    pass

        try:
            if rate is None and hasattr(borrow_ctx, "default_rate_annual"):
                rate = float(getattr(borrow_ctx, "default_rate_annual"))
        except Exception:
            pass

    if rate is None and availability_df is not None:
        try:
            df = availability_df
            sym_col = df.get("symbol")
            if isinstance(sym_col, pd.Series):
                df = df[sym_col == symbol]
                if not df.empty and "rate_annual" in df.columns:
                    val = pd.to_numeric(df["rate_annual"], errors="coerce").dropna()
                    if not val.empty:
                        rate = float(val.iloc[-1])
        except Exception:
            pass

    if rate is None:
        rate = float(default_rate_annual) if default_rate_annual is not None else 0.0
    return float(rate)


# =============================================================================
#                             Build pair data
# =============================================================================


def prepare_pairs_data(  # noqa: C901
    prices: pd.DataFrame,
    pairs: dict[str, Any] | Iterable[tuple[str, str]],
    adv_map: dict[str, float] | None = None,
    adv_vol_map: dict[str, float] | None = None,
    adv_shares_map: dict[str, float] | None = None,
    adv_currency_map: dict[str, str] | None = None,
    verbose: bool = False,
    disable_prefilter: bool = False,
    *,
    attach_prices_df: bool = True,
    pair_adv_mode: str = "harmonic",
    prefilter_range: tuple[Any, Any] | None = None,
    pair_prefilter_cfg: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build per-pair data structures on top of a cleaned execution-price panel.

    This function assumes that all symbol-level data cleaning, calendar
    alignment, and corporate-actions handling have already been applied
    upstream in the processing/universe modules.
    """
    # --- 0) Prework -----------------------------------------------------------
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices.index = _idx_to_ex_tz(pd.DatetimeIndex(prices.index), _EX_TZ)

    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    # Prices are expected to be pre-cleaned and grid-aligned by processing.
    # Loader must not apply additional panel-level filling (to avoid silently changing
    # the processed dataset semantics).
    prices_float = prices.apply(pd.to_numeric, errors="coerce")
    idx = prices_float.index
    idx_prefilter = idx
    if prefilter_range is not None:
        try:
            start_raw, end_raw = prefilter_range
            t0 = _coerce_ts_like_index(start_raw, idx)
            t1_end = _coerce_ts_like_index(end_raw, idx)
            idx_prefilter = idx[(idx >= t0) & (idx <= t1_end)]
        except Exception:
            idx_prefilter = idx
    if idx_prefilter.empty and prefilter_range is not None:
        logger.warning("prepare_pairs_data: prefilter_range produced empty window.")
    pf_cfg = dict(pair_prefilter_cfg or {})
    coint_alpha = float(pf_cfg.get("coint_alpha", 0.05))
    min_obs = max(2, int(pf_cfg.get("min_obs", 30)))
    half_life_cfg = resolve_half_life_cfg(pf_cfg.get("half_life"))

    pairs_norm = _normalize_pairs_input(pairs)
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
    pair_mode = str(pair_adv_mode or "harmonic").lower()

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

    def _combine_adv(a: float, b: float) -> float:
        vals = [v for v in (a, b) if np.isfinite(v) and v > 0]
        if len(vals) == 2:
            if pair_mode == "harmonic":
                try:
                    return float(2.0 / (1.0 / vals[0] + 1.0 / vals[1]))
                except ZeroDivisionError:
                    return float("nan")
            if pair_mode == "geometric":
                return float(np.sqrt(vals[0] * vals[1]))
            if pair_mode == "max":
                return float(max(vals))
            return float(min(vals))
        if len(vals) == 1:
            return float(vals[0])
        return float("nan")

    # NOTE: processing is the single source of truth for ADV/$ADV/etc.
    # Loader only reads the precomputed maps (adv_map_path) and derives a
    # deterministic pair-level ADV aggregation from it.

    out: dict[str, dict[str, Any]] = {}
    filtered_reasons: dict[str, str] = {}

    # --- 1) Main loop over pairs ----------------------------------------------
    for pair in sorted(pairs_norm.keys(), key=str):
        meta = pairs_norm[pair]
        t1 = meta.get("t1")
        t2 = meta.get("t2")
        if t1 is None or t2 is None:
            filtered_reasons[pair] = f"missing_ticker({t1},{t2})"
            if verbose:
                logger.debug("Pair %s skipped: missing tickers (%s,%s)", pair, t1, t2)
            continue
        if t1 not in prices_float.columns or t2 not in prices_float.columns:
            filtered_reasons[pair] = f"missing_ticker({t1},{t2})"
            if verbose:
                logger.debug("Pair %s skipped: missing tickers (%s,%s)", pair, t1, t2)
            continue

        s1 = prices_float[t1]
        s2 = prices_float[t2]
        s1_pref = s1.reindex(idx_prefilter)
        s2_pref = s2.reindex(idx_prefilter)

        coint_diag: dict[str, Any] | None = None
        if not disable_prefilter:
            try:
                coint_diag = evaluate_pair_cointegration(
                    pd.DataFrame({"y": s1_pref, "x": s2_pref}),
                    coint_alpha=coint_alpha,
                    min_obs=min_obs,
                    half_life_cfg=half_life_cfg,
                )
                if not bool(coint_diag.get("passed", False)):
                    filtered_reasons[pair] = str(
                        coint_diag.get("reject_reason") or "prefilter_failed"
                    )
                    if verbose:
                        logger.debug(
                            "Pair %s failed pair_prefilter: %s",
                            pair,
                            filtered_reasons[pair],
                        )
                    continue
            except Exception as e:
                # wrapper already shields errors, but keep a reason for transparency
                filtered_reasons[pair] = f"prefilter_error:{e}"
                if verbose:
                    logger.debug("Pair %s prefilter error: %s", pair, e)
                continue
        else:
            beta_hat, beta_reason = (
                _strategy_helpers.estimate_beta_ols_with_intercept_details(
                    s1_pref, s2_pref
                )
            )
            if beta_hat is None:
                filtered_reasons[pair] = str(beta_reason or "beta_estimation_failed")
                if verbose:
                    logger.debug(
                        "Pair %s skipped due to invalid beta without prefilter: %s",
                        pair,
                        filtered_reasons[pair],
                    )
                continue

        adv_y = _fetch_metric(adv_map, t1)
        adv_x = _fetch_metric(adv_map, t2)
        meta_out: dict[str, Any] = {
            "t1": t1,
            "t2": t2,
            "adv_usd_t1": adv_y,
            "adv_usd_t2": adv_x,
            "adv_pair_mode": pair_mode,
        }
        meta_out["adv_t1"] = adv_y
        meta_out["adv_t2"] = adv_x
        meta_out["adv_pair_usd"] = _combine_adv(adv_y, adv_x)
        meta_out["adv_pair"] = meta_out["adv_pair_usd"]
        if coint_diag is not None:
            meta_out["cointegration"] = dict(coint_diag)

        item: dict[str, Any] = {
            "t1_price": s1,
            "t2_price": s2,
            "meta": meta_out,
        }
        if attach_prices_df:
            item["prices"] = pd.DataFrame({"y": s1, "x": s2})

        out[pair] = item

    logger.info(
        "prepare_pairs_data: retained %d pairs (from %d)", len(out), len(pairs_norm)
    )
    if filtered_reasons:
        counts = Counter(filtered_reasons.values())
        logger.info("prepare_pairs_data: filtered_reasons=%s", dict(counts))
        if verbose:
            sample = dict(list(filtered_reasons.items())[:20])
            logger.debug("prepare_pairs_data: filtered sample: %s", sample)
    return out


# =============================================================================
#                                 PUBLIC EXPORTS
# =============================================================================

__all__ = [
    # Price helpers
    "as_price_mapping",
    "series_price_at",
    "load_price_data",
    "load_price_panel",
    "select_field_from_panel",
    # Pairs
    "load_filtered_pairs",
    "prepare_pairs_data",
    # ADV
    "load_adv_map",
    "save_adv_map",
    "filter_by_adv",
    # Borrow shim
    "resolve_borrow_rate",
]
