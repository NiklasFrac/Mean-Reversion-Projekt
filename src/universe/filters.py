from __future__ import annotations

from typing import Any

import pandas as pd

from universe.checkpoint import norm_symbol
from universe.coercion import cfg_bool
from universe.numeric_utils import replace_inf_with_nan

__all__ = ["apply_filters_with_reasons", "apply_filters"]


def _apply_filters_with_reasons(
    df: pd.DataFrame,
    filters_cfg: dict[str, Any],
    audit: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        out_empty = df.copy()
        if out_empty.index.name is None:
            out_empty.index.name = "ticker"
        return out_empty, {}

    reasons: dict[str, int] = {}

    def _inc(key: str, n: int) -> None:
        reasons[key] = reasons.get(key, 0) + int(n)

    def _audit_step(
        *,
        step: str,
        code: str | None,
        n_before: int,
        n_after: int,
        removed: int,
        added: int = 0,
    ) -> None:
        if audit is None:
            return
        audit.append(
            {
                "step": str(step),
                "code": str(code) if code is not None else None,
                "n_before": int(n_before),
                "n_after": int(n_after),
                "removed": int(removed),
                "added": int(added),
            }
        )

    def _mark(
        frame: pd.DataFrame, cond: pd.Series, code: str, step: str | None = None
    ) -> pd.DataFrame:
        n_before = int(frame.shape[0])
        dropped = int((~cond).sum())
        if dropped > 0:
            _inc(code, dropped)
        out_frame = frame[cond]
        _audit_step(
            step=step or code,
            code=code,
            n_before=n_before,
            n_after=int(out_frame.shape[0]),
            removed=int(dropped),
            added=0,
        )
        return out_frame

    out = df.copy(deep=True)
    if out.index.name is None:
        out.index.name = "ticker"

    # Prefer historical dollar_adv (computed from price*volume history). Fallback to snapshot if absent.
    if "dollar_adv" not in out.columns:
        if {"price", "volume"}.issubset(out.columns):
            px = pd.to_numeric(out["price"], errors="coerce")
            vol = pd.to_numeric(out["volume"], errors="coerce")
            dadv = replace_inf_with_nan(px * vol)
            out["dollar_adv"] = dadv
        else:
            out["dollar_adv"] = float("nan")

    # Effective price:
    # - Use warmup median whenever that column is present.
    # - Fall back to snapshot only when callers do not provide any warmup column
    #   (compatibility path for minimal frames/tests).
    snapshot_price = pd.to_numeric(out.get("price", float("nan")), errors="coerce")
    if "price_warmup_med" in out.columns:
        warmup_price = pd.to_numeric(out["price_warmup_med"], errors="coerce")
        out["price_eff"] = warmup_price
    else:
        out["price_eff"] = snapshot_price

    # Effective volume for filtering should use the same historical warmup
    # as price/ADV when available.
    if "volume_warmup_avg" in out.columns:
        out["volume_eff"] = pd.to_numeric(out["volume_warmup_avg"], errors="coerce")
    else:
        out["volume_eff"] = pd.to_numeric(
            out.get("volume", float("nan")), errors="coerce"
        )

    drop_na = cfg_bool(filters_cfg, "drop_na", True)
    drop_zero = cfg_bool(filters_cfg, "drop_zero", True)
    # Core validation must follow the same effective basis as filtering:
    # warmup-derived price/volume when present, snapshot fallback otherwise.
    core_cols = [
        c for c in ["price_eff", "market_cap", "volume_eff"] if c in out.columns
    ]

    if drop_na and core_cols:
        cond = out[core_cols].notna().all(axis=1)
        out = _mark(out, cond, "REASON_NA_CORE_FIELD", step="Drop NA core fields")

    if drop_zero:
        c_price = out.get("price_eff", pd.Series(1, index=out.index))
        c_vol = out.get("volume_eff", pd.Series(1, index=out.index))
        cond = (pd.to_numeric(c_price, errors="coerce") > 0) & (
            pd.to_numeric(c_vol, errors="coerce") > 0
        )
        out = _mark(
            out,
            cond,
            "REASON_ZERO_CORE_FIELD",
            step="Drop zero/invalid price or volume",
        )

    if "float_pct" in out.columns and "shares_out" in out.columns:
        ff = pd.to_numeric(out["float_pct"], errors="coerce") * pd.to_numeric(
            out["shares_out"], errors="coerce"
        )
        out["free_float_shares"] = replace_inf_with_nan(ff).astype(float)
    if "float_pct" in out.columns and "market_cap" in out.columns:
        ffm = pd.to_numeric(out["float_pct"], errors="coerce") * pd.to_numeric(
            out["market_cap"], errors="coerce"
        )
        out["free_float_mcap"] = replace_inf_with_nan(ffm).astype(float)

    def _range(
        frame: pd.DataFrame,
        col: str,
        lo: float | None,
        hi: float | None,
        low_code: str,
        high_code: str,
        *,
        label: str,
    ) -> pd.DataFrame:
        f = frame
        if lo is not None and col in f.columns:
            s = pd.to_numeric(f[col], errors="coerce").fillna(-float("inf"))
            cond = s >= float(lo)
            f = _mark(f, cond, low_code, step=f"{label}: min")
        if hi is not None and col in f.columns:
            s = pd.to_numeric(f[col], errors="coerce").fillna(float("inf"))
            cond = s <= float(hi)
            f = _mark(f, cond, high_code, step=f"{label}: max")
        return f

    out = _range(
        out,
        "price_eff",
        filters_cfg.get("min_price"),
        filters_cfg.get("max_price"),
        "REASON_LOW_PRICE",
        "REASON_HIGH_PRICE",
        label="Price",
    )
    out = _range(
        out,
        "market_cap",
        filters_cfg.get("min_market_cap"),
        filters_cfg.get("max_market_cap"),
        "REASON_LOW_MCAP",
        "REASON_HIGH_MCAP",
        label="Market cap",
    )
    out = _range(
        out,
        "volume_eff",
        filters_cfg.get("min_avg_volume"),
        filters_cfg.get("max_avg_volume"),
        "REASON_LOW_VOL",
        "REASON_HIGH_VOL",
        label="Volume",
    )
    # Effective dollar ADV: use historical warmup only when available; do not
    # mix in present-day snapshot values to avoid look-ahead.
    if "dollar_adv_hist" in out.columns:
        eff = pd.to_numeric(out["dollar_adv_hist"], errors="coerce")
        out["dollar_adv_eff"] = eff
    else:
        out["dollar_adv_eff"] = pd.to_numeric(
            out.get("dollar_adv", float("nan")), errors="coerce"
        )

    min_dadv = filters_cfg.get("min_dollar_adv")
    if min_dadv is not None and "dollar_adv_eff" in out.columns:
        n_before = int(out.shape[0])
        dadv_series = pd.to_numeric(out["dollar_adv_eff"], errors="coerce")
        missing_mask = dadv_series.isna()
        if missing_mask.any():
            n_missing = int(missing_mask.sum())
            _inc("REASON_NO_HIST_ADV", n_missing)
            out = out[~missing_mask]
            _audit_step(
                step="ADV warmup missing (no hist dollar_adv)",
                code="REASON_NO_HIST_ADV",
                n_before=n_before,
                n_after=int(out.shape[0]),
                removed=n_missing,
                added=0,
            )
            dadv_series = dadv_series.loc[out.index]
        out = _mark(
            out,
            dadv_series >= float(min_dadv),
            "REASON_LOW_DADV",
            step="Dollar ADV (hist) threshold",
        )

    min_float = filters_cfg.get("min_float_pct")
    treat_missing = cfg_bool(filters_cfg, "treat_missing_float_as_pass", False)
    if min_float is not None and "float_pct" in out.columns:
        fp = pd.to_numeric(out["float_pct"], errors="coerce")
        cond = (fp.fillna(1.0) if treat_missing else fp.fillna(-float("inf"))) >= float(
            min_float
        )
        out = _mark(out, cond, "REASON_FLOAT_PCT", step="Float pct threshold")

    min_ff_shares = filters_cfg.get("min_free_float_shares")
    if min_ff_shares is not None and "free_float_shares" in out.columns:
        s = pd.to_numeric(out["free_float_shares"], errors="coerce").fillna(0.0)
        cond = s >= float(min_ff_shares)
        out = _mark(out, cond, "REASON_FF_SHARES", step="Free-float shares threshold")

    min_ff_mcap = filters_cfg.get("min_free_float_dollar_cap")
    if min_ff_mcap is not None and "free_float_mcap" in out.columns:
        s = pd.to_numeric(out["free_float_mcap"], errors="coerce").fillna(0.0)
        cond = s >= float(min_ff_mcap)
        out = _mark(
            out, cond, "REASON_FF_DOLLAR_CAP", step="Free-float $ cap threshold"
        )

    if "dividend" in out.columns and cfg_bool(filters_cfg, "require_dividend", False):
        cond = out["dividend"].fillna(False).astype(bool)
        out = _mark(out, cond, "REASON_DIVIDEND_REQ", step="Dividend required")

    def _norm_token(value: Any) -> str:
        raw = str(value).strip()
        if not raw:
            return ""
        return norm_symbol(raw)

    def _index_by_norm(frame: pd.DataFrame) -> dict[str, list[Any]]:
        mapping: dict[str, list[Any]] = {}
        for idx_val in frame.index:
            token = _norm_token(idx_val)
            if not token:
                continue
            mapping.setdefault(token, []).append(idx_val)
        return mapping

    whitelist = list(
        dict.fromkeys(
            token
            for token in (
                _norm_token(s) for s in filters_cfg.get("symbol_whitelist", [])
            )
            if token
        )
    )
    blacklist = list(
        dict.fromkeys(
            token
            for token in (
                _norm_token(s) for s in filters_cfg.get("symbol_blacklist", [])
            )
            if token
        )
    )

    if blacklist:
        out_norm_map = _index_by_norm(out)
        drop_idx: list[Any] = []
        for token in blacklist:
            drop_idx.extend(out_norm_map.get(token, []))
        drop_idx = list(dict.fromkeys(drop_idx))
        if drop_idx:
            n_before = int(out.shape[0])
            _inc("REASON_BLACKLISTED", len(drop_idx))
            out = out.drop(index=drop_idx, errors="ignore")
            _audit_step(
                step="Symbol blacklist",
                code="REASON_BLACKLISTED",
                n_before=n_before,
                n_after=int(out.shape[0]),
                removed=int(len(drop_idx)),
                added=0,
            )

    if whitelist:
        out_norm_set = set(_index_by_norm(out).keys())
        df_norm_map = _index_by_norm(df)
        missing_norm = [
            token
            for token in whitelist
            if token not in out_norm_set and token in df_norm_map
        ]
        add_idx: list[Any] = []
        for token in missing_norm:
            add_idx.extend(df_norm_map[token])
        add_idx = list(dict.fromkeys(add_idx))
        if add_idx:
            n_before = int(out.shape[0])
            add_rows = df.loc[add_idx].copy()

            # Re-added whitelist rows should keep the same derived columns as the
            # filtered frame so downstream exports do not see NaN-only artifacts.
            if "price_eff" in out.columns and "price_eff" not in add_rows.columns:
                if "price_warmup_med" in add_rows.columns:
                    warm = pd.to_numeric(add_rows["price_warmup_med"], errors="coerce")
                    snap = pd.to_numeric(
                        add_rows.get("price", float("nan")), errors="coerce"
                    )
                    add_rows["price_eff"] = warm.where(warm.notna(), snap)
                else:
                    add_rows["price_eff"] = pd.to_numeric(
                        add_rows.get("price", float("nan")), errors="coerce"
                    )
            if "dollar_adv" in out.columns and "dollar_adv" not in add_rows.columns:
                if {"price", "volume"}.issubset(add_rows.columns):
                    px = pd.to_numeric(add_rows["price"], errors="coerce")
                    vol = pd.to_numeric(add_rows["volume"], errors="coerce")
                    add_rows["dollar_adv"] = replace_inf_with_nan(px * vol)
                else:
                    add_rows["dollar_adv"] = float("nan")
            if (
                "free_float_shares" in out.columns
                and "free_float_shares" not in add_rows.columns
            ):
                if {"float_pct", "shares_out"}.issubset(add_rows.columns):
                    ff = pd.to_numeric(
                        add_rows["float_pct"], errors="coerce"
                    ) * pd.to_numeric(add_rows["shares_out"], errors="coerce")
                    add_rows["free_float_shares"] = replace_inf_with_nan(ff).astype(
                        float
                    )
                else:
                    add_rows["free_float_shares"] = float("nan")
            if (
                "free_float_mcap" in out.columns
                and "free_float_mcap" not in add_rows.columns
            ):
                if {"float_pct", "market_cap"}.issubset(add_rows.columns):
                    ffm = pd.to_numeric(
                        add_rows["float_pct"], errors="coerce"
                    ) * pd.to_numeric(add_rows["market_cap"], errors="coerce")
                    add_rows["free_float_mcap"] = replace_inf_with_nan(ffm).astype(
                        float
                    )
                else:
                    add_rows["free_float_mcap"] = float("nan")
            if (
                "dollar_adv_eff" in out.columns
                and "dollar_adv_eff" not in add_rows.columns
            ):
                if "dollar_adv_hist" in add_rows.columns:
                    add_rows["dollar_adv_eff"] = pd.to_numeric(
                        add_rows["dollar_adv_hist"], errors="coerce"
                    )
                else:
                    add_rows["dollar_adv_eff"] = pd.to_numeric(
                        add_rows.get("dollar_adv", float("nan")), errors="coerce"
                    )

            out = pd.concat([out, add_rows], axis=0)
            if out.index.has_duplicates:
                out = out[~out.index.duplicated(keep="first")]
            _audit_step(
                step="Symbol whitelist (re-add)",
                code=None,
                n_before=n_before,
                n_after=int(out.shape[0]),
                removed=0,
                added=int(len(add_idx)),
            )

    if isinstance(out, pd.Series):
        out = out.to_frame()
    elif not isinstance(out, pd.DataFrame):
        try:
            tickers = list(out)
            out = df.loc[df.index.intersection(tickers)].copy()
        except Exception:
            out = pd.DataFrame()

    if out.index.name is None:
        out.index.name = "ticker"

    return out, reasons


# Public alias without underscore for external callers/tests
apply_filters_with_reasons = _apply_filters_with_reasons


def apply_filters(df: pd.DataFrame, filters_cfg: dict[str, Any]) -> pd.DataFrame:
    filtered, _ = _apply_filters_with_reasons(df, filters_cfg)
    return filtered
