from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Mapping

from universe.symbol_filter_defaults import (
    DEFAULT_DROP_CONTAINS,
    DEFAULT_DROP_PREFIXES,
    DEFAULT_DROP_REGEX,
    DEFAULT_DROP_SUFFIXES,
)

logger = logging.getLogger("runner_universe")

__all__ = [
    "pre_filter_symbols",
    "load_exchange_tickers",
    "get_last_screener_meta",
]

_LAST_SCREENER_META: dict[str, Any] = {}


def get_last_screener_meta() -> Mapping[str, Any]:
    """
    Returns metadata about the last screener CSV successfully used by
    `load_exchange_tickers()` in the current Python process.

    This is used for run provenance (manifest/run-scoped artifacts). It is best-effort:
    when no screener has been loaded yet, returns an empty mapping.
    """
    return dict(_LAST_SCREENER_META)


def _load_screener_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            peek = fh.readline()
            if not peek:
                return []
            delimiter = "|" if "|" in peek else ","
            fh.seek(0)
            reader = csv.DictReader(fh, delimiter=delimiter)
            if not reader.fieldnames:
                return []
            symbol_col = None
            for candidate in ("Symbol", "symbol", "Ticker", "ticker"):
                if candidate in reader.fieldnames:
                    symbol_col = candidate
                    break
            if symbol_col is None:
                symbol_col = reader.fieldnames[0]
            rows: list[tuple[str, dict[str, Any]]] = []
            for row in reader:
                val = row.get(symbol_col)
                if not val:
                    continue
                rows.append((str(val).strip().upper(), dict(row)))
            return rows
    except Exception as exc:  # pragma: no cover - IO errors exercised indirectly
        logger.warning("Could not read symbol file %s: %s", path, exc)
        return []


def pre_filter_symbols(
    symbols: list[str],
    drop_prefixes: list[str],
    drop_suffixes: list[str] | None = None,
    drop_regex: list[str] | None = None,
    drop_contains: list[str] | None = None,
) -> list[str]:
    import re

    pref = tuple(p.upper() for p in (drop_prefixes or []))
    suff = tuple(s.upper() for s in (drop_suffixes or []))
    regs = [re.compile(r, re.I) for r in (drop_regex or [])]
    cont = [c.upper() for c in (drop_contains or [])]
    out: list[str] = []
    for sym in symbols:
        token = str(sym).strip().upper()
        if pref and token.startswith(pref):
            continue
        if suff and token.endswith(suff):
            continue
        if cont and any(chunk in token for chunk in cont):
            continue
        if regs and any(r.search(token) for r in regs):
            continue
        out.append(token)
    return out


def _is_common_equity(row: dict[str, Any]) -> bool:
    """
    Heuristic: keep operating common equity, drop ETFs/ETNs/CEFs/funds/
    warrants/preferreds/units and obvious finance investment vehicles.

    Important:
    - "Common Stock"/"Common Shares" alone is NOT sufficient evidence.
      Many closed-end funds have exactly that wording in the screener.
    """
    lowered = {str(k).lower(): v for k, v in row.items()}

    # Boolean/flag columns that explicitly mark ETFs/funds
    for key in ("etf", "is_etf", "fund", "isfund"):
        if key in lowered:
            val = str(lowered[key]).strip().lower()
            if val in {"1", "true", "yes", "y", "etf", "fund"}:
                return False

    # Type-like columns (if present)
    type_keys = [
        "type",
        "security type",
        "security_type",
        "asset type",
        "asset_type",
        "class",
    ]
    type_val = None
    for key in type_keys:
        if key in lowered:
            type_val = str(lowered[key]).strip()
            break
    if type_val:
        tv = type_val.upper()
        blocked = {
            "ETF",
            "ETN",
            "EXCHANGE TRADED FUND",
            "EXCHANGE TRADED NOTE",
            "CLOSED END FUND",
            "CLOSED-END FUND",
            "CEF",
            "FUND",
            "MUTUAL FUND",
            "PREFERRED",
            "PREFERRED STOCK",
            "PFD",
            "WARRANT",
            "RIGHT",
            "RIGHTS",
            "UNIT",
            "UNITS",
        }
        if any(blk in tv for blk in blocked):
            return False

    name_val = str(lowered.get("name") or lowered.get("company name") or "").upper()
    sector_val = str(lowered.get("sector") or "").upper()
    industry_val = str(lowered.get("industry") or "").upper()

    # Hard name-level blockers
    blocked_name_hints = (
        " ETF",
        " ETN",
        "CLOSED END FUND",
        "CLOSED-END FUND",
        "TRUST PREFERRED",
        "PREFERRED SECURITIES",
        "PREFERRED STOCK",
        "PREFERRED SHARES",
        "DEPOSITARY SHARE",
        "DEPOSITARY SHARES",
        "WARRANT",
        " RIGHT",
        " RIGHTS",
        " UNIT",
        " UNITS",
        "SENIOR NOTE",
        "SENIOR NOTES",
        " NOTE DUE",
        " NOTES DUE",
        " BOND",
        "BENEFICIAL INTEREST",
        "SHARES OF BENEFICIAL INTEREST",
    )
    if any(hint in name_val for hint in blocked_name_hints):
        return False

    # Explicit fund detection (word boundary so "FUNDING" does not match)
    if re.search(r"\bFUNDS?\b", name_val):
        return False

    # "Trust" names: some are real operating companies (banks, REITs), many are not.
    equity_name_hints = (
        "COMMON STOCK",
        "COMMON SHARES",
        "ORDINARY SHARES",
        "CLASS A COMMON",
        "CLASS B COMMON",
        "CLASS C COMMON",
    )

    safe_trust_industries = {
        "REAL ESTATE INVESTMENT TRUSTS",
        "MAJOR BANKS",
        "REGIONAL BANKS",
        "S&LS/SAVINGS BANKS",
        "BUILDING OPERATORS",
    }

    safe_trust_name_hints = (
        "REALTY TRUST",
        "NORTHERN TRUST",
        "TRUSTCO BANK",
        "TRUSTMARK",
        "WASHINGTON TRUST",
        "WINTRUST",
        "COMMUNITY TRUST BANCORP",
        "ADAMAS TRUST",
    )

    obvious_trust_vehicle_hints = (
        " INCOME TRUST",
        " MUNICIPAL TRUST",
        " OPPORTUNITY TRUST",
        " EQUITY TRUST",
        " CREDIT TRUST",
        " TRUST FOR INVESTMENT",
        " DYNAMIC INCOME TRUST",
        " SMALL-CAP TRUST",
        " MICRO-CAP TRUST",
    )

    if "TRUST" in name_val:
        if any(h in name_val for h in obvious_trust_vehicle_hints):
            return False

        if industry_val in safe_trust_industries:
            return True

        if any(h in name_val for h in safe_trust_name_hints):
            return True

        # Finance-sector trust names are guilty until proven innocent.
        if sector_val == "FINANCE":
            return False

        # Outside finance, require at least an explicit equity marker.
        if not any(h in name_val for h in equity_name_hints):
            return False

    # Catch bland finance-vehicle names without explicit "Fund"/"Trust".
    suspicious_finance_industries = {
        "INVESTMENT MANAGERS",
        "TRUSTS EXCEPT EDUCATIONAL RELIGIOUS AND CHARITABLE",
        "FINANCE/INVESTORS SERVICES",
        "FINANCE COMPANIES",
    }

    suspicious_vehicle_style_hints = (
        "TOTAL RETURN",
        "TAX-ADVANTAGE",
        "DIVIDEND OPP",
        "DIVIDEND OPPORTUNIT",
        "MUNICIPAL",
        "CONVERTIBLE",
        "HIGH YIELD",
        "HIGH INCOME",
        "STRATEGIC",
        "ALLOCATION",
        "INFRASTRUCTURE",
        "OPPORTUNITY",
        "GLOBAL DYNAMIC",
        "NEXTGEN",
    )

    if (
        sector_val == "FINANCE"
        and industry_val in suspicious_finance_industries
        and any(h in name_val for h in suspicious_vehicle_style_hints)
    ):
        return False

    return True


def load_exchange_tickers(
    *,
    filters_cfg: Mapping[str, Any],
    universe_cfg: Mapping[str, Any],
) -> list[str]:
    glob_pattern = (
        universe_cfg.get("screener_glob") or "runs/data/nasdaq_screener_*.csv"
    )
    selection_mode_raw = str(
        universe_cfg.get("screener_selection_mode", "error_if_ambiguous")
    ).strip()
    selection_mode = selection_mode_raw.lower() or "error_if_ambiguous"
    if selection_mode not in {"error_if_ambiguous", "latest_mtime"}:
        logger.warning(
            "Unknown universe.screener_selection_mode=%r; using 'error_if_ambiguous'.",
            selection_mode_raw,
        )
        selection_mode = "error_if_ambiguous"
    glob_path = Path(glob_pattern)
    base_dir = glob_path.parent if glob_path.parent != Path("") else Path(".")
    pattern = glob_path.name
    candidates = sorted(
        base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise RuntimeError(f"No screener CSVs matching {glob_pattern}")
    if len(candidates) > 1 and selection_mode == "error_if_ambiguous":
        examples = ", ".join(str(p) for p in candidates[:5])
        raise RuntimeError(
            "Ambiguous universe.screener_glob matched multiple files "
            f"({len(candidates)}): {examples}. Pin a single file path or set "
            "universe.screener_selection_mode='latest_mtime' to allow newest-file selection."
        )
    if len(candidates) > 1 and selection_mode == "latest_mtime":
        logger.warning(
            "Screener glob matched %d files; selecting newest by mtime because "
            "universe.screener_selection_mode=latest_mtime.",
            len(candidates),
        )

    candidates_to_try = (
        candidates if selection_mode == "latest_mtime" else candidates[:1]
    )

    screener_symbols: list[str] = []
    chosen_path: Path | None = None
    chosen_rows_total: int | None = None
    chosen_rows_kept: int | None = None
    chosen_rows_dropped: int | None = None
    for cand in candidates_to_try:
        rows = _load_screener_rows(cand)
        if not rows:
            continue
        filtered_rows = [(sym, row) for sym, row in rows if _is_common_equity(row)]
        if not filtered_rows:
            continue
        screener_symbols = [sym for sym, _ in filtered_rows]
        chosen_path = cand
        chosen_rows_total = len(rows)
        chosen_rows_kept = len(filtered_rows)
        chosen_rows_dropped = len(rows) - len(filtered_rows)
        logger.info(
            "Screener CSV selected: %s | rows_total=%d | kept_instrument=%d | dropped_instrument=%d",
            cand,
            int(chosen_rows_total),
            int(chosen_rows_kept),
            int(chosen_rows_dropped),
        )
        break
    if not screener_symbols:
        raise RuntimeError(
            f"Found {len(candidates)} screener files under {base_dir} but no symbols parsed."
        )

    prefilter_before = len(screener_symbols)
    filtered = pre_filter_symbols(
        screener_symbols,
        drop_prefixes=filters_cfg.get("drop_prefixes", list(DEFAULT_DROP_PREFIXES)),
        drop_suffixes=filters_cfg.get("drop_suffixes", list(DEFAULT_DROP_SUFFIXES)),
        drop_regex=filters_cfg.get("drop_regex", list(DEFAULT_DROP_REGEX)),
        drop_contains=filters_cfg.get("drop_contains", list(DEFAULT_DROP_CONTAINS)),
    )
    prefilter_after = len(filtered)
    prefilter_dropped = prefilter_before - prefilter_after
    if prefilter_dropped:
        logger.info(
            "Symbol prefilter: %d -> %d (dropped=%d).",
            int(prefilter_before),
            int(prefilter_after),
            int(prefilter_dropped),
        )

    dedup_before = len(filtered)
    seen = set()
    uniq: list[str] = []
    for sym in filtered:
        if sym not in seen:
            seen.add(sym)
            uniq.append(sym)
    dedup_after = len(uniq)
    dedup_dropped = dedup_before - dedup_after
    if dedup_dropped:
        logger.info(
            "Symbol dedup: %d -> %d (dropped_dupes=%d).",
            int(dedup_before),
            int(dedup_after),
            int(dedup_dropped),
        )

    manual_drop_symbols = {
        str(x).strip().upper()
        for x in universe_cfg.get("manual_drop_symbols", []) or []
        if str(x).strip()
    }
    if manual_drop_symbols:
        before_manual = len(uniq)
        uniq = [sym for sym in uniq if sym not in manual_drop_symbols]
        dropped_manual = before_manual - len(uniq)
        if dropped_manual:
            logger.info(
                "Manual denylist: %d -> %d (dropped=%d).",
                int(before_manual),
                int(len(uniq)),
                int(dropped_manual),
            )

    # ---- Provenance: persist which screener CSV was selected (best-effort). ----
    # This module-level metadata is consumed by builder/runner for manifest/run-scoped inputs.
    try:
        meta: dict[str, Any] = {}
        if chosen_path is not None and chosen_path.exists():
            from universe.utils import _sha1

            stat = chosen_path.stat()
            meta = {
                "path": str(chosen_path),
                "mtime_utc": None,
                "sha1": _sha1(chosen_path),
                # Counts for paper-ready flow reporting.
                "rows_total": int(chosen_rows_total or 0),
                "rows_kept_instrument_type": int(chosen_rows_kept or 0),
                "rows_dropped_instrument_type": int(chosen_rows_dropped or 0),
                "symbols_before_symbol_prefilter": int(prefilter_before),
                "symbols_after_symbol_prefilter": int(prefilter_after),
                "symbols_dropped_symbol_prefilter": int(prefilter_dropped),
                "symbols_before_symbol_dedup": int(dedup_before),
                "symbols_after_symbol_dedup": int(dedup_after),
                "symbols_dropped_symbol_dedup": int(dedup_dropped),
                "selection_mode": selection_mode,
                "matches_count": int(len(candidates)),
            }
            try:
                # st_mtime is epoch seconds; store ISO UTC for readability.
                import datetime as dt

                meta["mtime_utc"] = dt.datetime.fromtimestamp(
                    stat.st_mtime, dt.UTC
                ).isoformat()
            except Exception:
                meta["mtime_utc"] = None
        _LAST_SCREENER_META.clear()
        _LAST_SCREENER_META.update(meta)
    except Exception:
        # Never let provenance recording break universe building.
        _LAST_SCREENER_META.clear()

    logger.info(
        "Universe seed (symbols): %d unique (after instrument-type + symbol prefilters + dedup).",
        int(len(uniq)),
    )
    return uniq
