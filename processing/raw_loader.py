from __future__ import annotations

import csv
import glob as globlib
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .logging_utils import logger

__all__ = [
    "UniversePanelBundle",
    "_discover",
    "_load_any_prices",
    "_extract_panel_from_suffixes",
    "load_raw_prices_from_universe",
]


@dataclass
class UniversePanelBundle:
    """Container for optional universe artefacts beyond close/volume."""

    panel: pd.DataFrame | None = None
    fields: dict[str, pd.DataFrame] = field(default_factory=dict)


def _discover(glob_pattern: str) -> Path | None:
    try:
        matches = sorted(
            Path().glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return matches[0] if matches else None
    except Exception:
        return None


def _load_any_prices(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".p"}:
        obj: Any = pd.read_pickle(path)
    elif path.suffix == ".parquet":
        obj = pd.read_parquet(path)
    else:
        obj = pd.read_csv(path)

    if isinstance(obj, dict):
        frames: list[pd.DataFrame] = []
        for t, v in obj.items():
            s = v if isinstance(v, pd.Series) else pd.Series(v)
            frames.append(pd.DataFrame({t: s}))
        df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
        cols_lower = {str(c).lower() for c in df.columns}
        if {"ts", "ticker", "close"}.issubset(cols_lower):
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(
                "America/New_York"
            )
            df = df.pivot(index="ts", columns="ticker", values="close")
    else:
        raise TypeError(f"Unsupported raw object type: {type(obj)}")
    return df


def _expected_symbol_count(data_dir: Path) -> int | None:
    """
    Best-effort expected ticker count for production runs.

    If `tickers_universe.csv` is present, we use it to sanity-check discovered artefacts
    and avoid accidentally using tiny/malformed placeholder files in `runs/data`.
    """
    p = data_dir / "tickers_universe.csv"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        header0 = rows[0][0].strip().lower() if rows[0] else ""
        start = 1 if header0 in {"ticker", "symbol"} else 0
        tickers = [r[0].strip() for r in rows[start:] if r and r[0].strip()]
        return len(set(tickers)) if tickers else None
    except Exception:
        return None


def _extract_panel_from_suffixes(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None, pd.DataFrame | None]:
    """
    Detect single-level columns formatted as <symbol>_<field> and split into per-field
    DataFrames plus an assembled panel MultiIndex (symbol, field). Returns:
      - field_map: dict[field_name -> wide DataFrame of symbols]
      - panel_df: MultiIndex DataFrame with columns (symbol, field) or None
      - close_df: the close field (if present) to preserve legacy close-only behavior
    """
    if df is None or df.empty or isinstance(df.columns, pd.MultiIndex):
        return {}, None, None

    field_map: dict[str, dict[str, pd.Series]] = {}
    known_fields = ["adj_close", "volume", "open", "high", "low", "close"]
    for col in df.columns:
        name = str(col)
        name_lower = name.lower()
        if "_" not in name:
            continue
        matched = False
        # Match longest suffix first (e.g., adj_close before close)
        for fld in sorted(known_fields, key=len, reverse=True):
            suffix = f"_{fld}"
            if name_lower.endswith(suffix):
                base = name[: -len(suffix)]
                if base:  # require a non-empty symbol prefix
                    field_map.setdefault(fld, {})[base] = pd.to_numeric(
                        df[col], errors="coerce"
                    )
                matched = True
                break
        if not matched:
            continue

    if not field_map:
        return {}, None, None

    per_field: dict[str, pd.DataFrame] = {}
    for fld, cols in field_map.items():
        per_field[fld] = pd.DataFrame(cols, index=df.index)

    panel_cols: dict[tuple[str, str], pd.Series] = {}
    for fld, frame in per_field.items():
        for sym in frame.columns:
            panel_cols[(sym, fld)] = frame[sym]

    panel_df = None
    if panel_cols:
        panel_df = pd.DataFrame(panel_cols, index=df.index)
        panel_df.columns = pd.MultiIndex.from_tuples(
            cast(list[tuple[Hashable, Hashable]], list(panel_df.columns)),
            names=["symbol", "field"],
        )
        panel_df = panel_df.sort_index(axis=1)

    close_df = (
        per_field["close"] if "close" in per_field else per_field.get("adj_close")
    )
    return per_field, panel_df, close_df


def load_raw_prices_from_universe(
    data_dir: Path,
    *,
    include_bundle: bool = False,
    price_globs: list[str] | None = None,
    volume_globs: list[str] | None = None,
) -> (
    tuple[pd.DataFrame, pd.DataFrame | None, UniversePanelBundle, dict[str, str | None]]
    | tuple[pd.DataFrame, pd.DataFrame | None, dict[str, str | None]]
):
    expected_syms = _expected_symbol_count(data_dir)
    min_rows = 50 if (expected_syms is not None and expected_syms >= 100) else 0
    min_cols = 50 if (expected_syms is not None and expected_syms >= 100) else 0

    def _discover_many(patterns: list[str]) -> list[Path]:
        # Preserve config intent: patterns are ordered by preference.
        # We still pick the newest match *within* each pattern, but we never let a later
        # "fallback" pattern override an earlier preferred one just because it's newer.
        def _matches_for_pattern(pat: str) -> list[Path]:
            p_pat = Path(pat)
            # pathlib.Path.glob is relative-only on Windows and cannot resolve absolute
            # wildcard patterns reliably. Handle absolute patterns explicitly.
            if p_pat.is_absolute():
                has_wildcards = any(ch in pat for ch in ("*", "?", "[", "]"))
                if has_wildcards:
                    return [Path(m) for m in globlib.glob(pat)]
                return [p_pat] if p_pat.exists() else []
            return list(Path().glob(pat))

        seen: set[Path] = set()
        out: list[Path] = []
        for pat in patterns:
            try:
                matches = _matches_for_pattern(pat)
            except Exception:
                continue
            matches = sorted(
                matches,
                key=lambda p: (
                    -(p.stat().st_mtime if p.exists() else 0.0),
                    str(p),
                ),
            )
            for p in matches:
                if p in seen:
                    continue
                seen.add(p)
                out.append(p)
        return out

    def _pick_first_valid(
        cands: list[Path], *, kind: str
    ) -> tuple[Path | None, pd.DataFrame | None]:
        for p in cands:
            try:
                df = _load_any_prices(p)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load %s from %s: %s", kind, p, exc)
                continue
            col_coverage = (
                (float(df.shape[1]) / float(expected_syms))
                if (expected_syms is not None and expected_syms > 0)
                else 1.0
            )
            suspicious_small = (
                bool(min_rows and min_cols)
                and df.shape[0] < min_rows
                and (df.shape[1] < min_cols or col_coverage < 0.5)
            )
            if suspicious_small:
                logger.warning(
                    "Ignoring suspiciously small %s artefact %s (shape=%s; expected_syms=%s; col_coverage=%.2f)",
                    kind,
                    p,
                    df.shape,
                    expected_syms,
                    col_coverage,
                )
                continue
            return p, df
        return None, None

    price_patterns = price_globs or [
        str(data_dir / "raw_prices.*.pkl"),
        str(data_dir / "raw_prices.pkl"),
        str(data_dir / "raw_prices.parquet"),
    ]
    volume_patterns = volume_globs or [
        str(data_dir / "raw_volume_unadj*.pkl"),
        str(data_dir / "raw_volume.*.pkl"),
        str(data_dir / "raw_volume.pkl"),
        str(data_dir / "raw_volume.parquet"),
    ]

    p_prices, prices = _pick_first_valid(
        _discover_many(price_patterns), kind="raw_prices"
    )
    if not p_prices:
        logger.warning(
            "raw_prices.* artefacts not found (patterns=%s). Will attempt to derive close from panel bundle.",
            price_patterns,
        )

    p_vol, volume = _pick_first_valid(
        _discover_many(volume_patterns), kind="raw_volume"
    )

    panel_patterns = [
        str(data_dir / "raw_prices_panel.*.pkl"),
        str(data_dir / "raw_prices_panel.pkl"),
        str(data_dir / "raw_prices_panel.parquet"),
    ]
    panel_cands = _discover_many(panel_patterns)
    p_panel = panel_cands[0] if panel_cands else None
    panel_df: pd.DataFrame | None = None
    panel_fields: dict[str, pd.DataFrame] = {}
    if p_panel:
        try:
            panel_df = _load_any_prices(p_panel)
            if isinstance(panel_df.columns, pd.MultiIndex):
                cols_mi = panel_df.columns
                if cols_mi.nlevels == 2:
                    names = list(cols_mi.names)
                    field_candidates = {
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj_close",
                        "volume",
                    }
                    field_level: int | None = None
                    symbol_level: int | None = None

                    if names:
                        if "field" in names:
                            field_level = names.index("field")
                        if "symbol" in names:
                            symbol_level = names.index("symbol")

                    if field_level is None or symbol_level is None:
                        lvl0 = cols_mi.get_level_values(0)
                        lvl1 = cols_mi.get_level_values(1)
                        score0 = sum(
                            str(v).lower() in field_candidates for v in lvl0.unique()
                        )
                        score1 = sum(
                            str(v).lower() in field_candidates for v in lvl1.unique()
                        )
                        if score0 > score1:
                            field_level, symbol_level = 0, 1
                        elif score1 > score0:
                            field_level, symbol_level = 1, 0

                    if field_level is None:
                        field_level = 1
                    if symbol_level is None:
                        symbol_level = 1 - field_level

                    if field_level == 0:
                        panel_df = panel_df.swaplevel(0, 1, axis=1)
                    panel_df.columns = panel_df.columns.set_names(["symbol", "field"])

                try:
                    field_level = panel_df.columns.names.index("field")
                except ValueError:
                    field_level = 1
                unique_fields = panel_df.columns.get_level_values(field_level).unique()
                for field_name in unique_fields:
                    try:
                        extracted = cast(
                            pd.DataFrame,
                            panel_df.xs(field_name, axis=1, level=field_level),
                        )
                    except KeyError:  # pragma: no cover - defensive
                        continue
                    panel_fields[str(field_name).lower()] = extracted.astype(float)
        except Exception as exc:
            logger.warning("Failed to load raw_prices_panel from %s: %s", p_panel, exc)
            panel_df = None

    # Parse flat OHLC columns (e.g., AAPL_open, AAPL_close) into panel fields if needed
    parsed_fields: dict[str, pd.DataFrame] = {}
    parsed_panel: pd.DataFrame | None = None
    parsed_close: pd.DataFrame | None = None
    if prices is not None:
        parsed_fields, parsed_panel, parsed_close = _extract_panel_from_suffixes(prices)
        if parsed_fields:
            # Prefer parsed close for legacy close-only downstream expectations
            if parsed_close is not None:
                prices = parsed_close
            # Merge into panel_fields; parsed panel overrides only when no explicit panel was loaded
            for k, v in parsed_fields.items():
                panel_fields.setdefault(k, v)
            if panel_df is None and parsed_panel is not None:
                panel_df = parsed_panel

    if (prices is None or prices.empty) and panel_fields:
        fallback_field = (
            "close"
            if "close" in panel_fields
            else ("adj_close" if "adj_close" in panel_fields else None)
        )
        if fallback_field:
            fallback_df = panel_fields[fallback_field]
            if not fallback_df.empty:
                prices = fallback_df.copy()
                logger.info(
                    "Derived raw close matrix from panel field '%s'.", fallback_field
                )
        if prices is None or prices.empty:
            raise FileNotFoundError(
                "raw_prices artefact missing and no suitable 'close' field could be derived from panel bundle."
            )
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        raise FileNotFoundError(
            f"raw_prices.* not found under {data_dir} and no panel-derived fallback available."
        )

    if (volume is None or volume.empty) and panel_fields.get("volume") is not None:
        vol_df = panel_fields["volume"]
        if not vol_df.empty:
            volume = vol_df.copy()
            logger.info("Derived raw volume matrix from panel bundle.")

    bundle = UniversePanelBundle(panel=panel_df, fields=panel_fields)

    used_paths: dict[str, str | None] = {
        "prices": str(p_prices)
        if p_prices
        else (f"{p_panel}[close]" if p_panel else None),
        "volume": str(p_vol) if p_vol else None,
        "panel": str(p_panel) if p_panel else None,
    }
    if include_bundle:
        return prices, volume, bundle, used_paths
    return prices, volume, used_paths
