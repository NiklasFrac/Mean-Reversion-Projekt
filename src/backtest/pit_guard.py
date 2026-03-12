# src/backtest/pit_guard.py
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Tuple, cast

import numpy as np
import pandas as pd

from backtest.windowing.splits import (
    require_backtest_splits,
    require_split_start_end,
)
from backtest.windowing.eval import synthesize_analysis_split
from backtest.utils.tz import align_ts_to_index, to_naive_utc, utc_now

# Type alias for the window runner (e.g. wf_core._process_window)
WindowRunner = Callable[
    [
        int,
        pd.DataFrame,
        pd.DataFrame | None,
        dict[str, Any],
        dict[str, Any],
        Any | None,
        Any | None,
        Any | None,
        str,
    ],
    Tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]],
]

_LOG = logging.getLogger("backtest.pit")
_LOG.propagate = False
if not _LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _LOG.addHandler(h)
_LOG.setLevel(logging.INFO)


@dataclass(frozen=True)
class PitGuardConfig:
    """
    Fine-grained controls for the PIT guard.

    test_len            Test-length limit (bars); None => from YAML
    atol_equity/rtol    Tolerance for equity equality
    compare_trades      Whether trades in the test window are compared
    compare_costs       Check cost/borrow fields (net_pnl, borrow_cost, buyin_penalty_cost, total_costs)
    compare_forced      Check hard_exit/hard_exit_reason
    availability_scope  "all" | "pit" | "window" (as in your pipeline)
    noise_sigma         Standard deviation of multiplicative noise for future prices
    seed                RNG seed (default: cfg['seed'] or 42)

    tol_cost_abs        Absolute tolerance for cost comparisons
    """

    enabled: bool = False
    test_len: int | None = None
    atol_equity: float = 0.0
    rtol_equity: float = 0.0
    compare_trades: bool = True
    compare_costs: bool = True
    compare_forced: bool = True
    availability_scope: str = "all"
    noise_sigma: float = 0.01
    seed: int | None = None
    mode: str = "auto"
    cut_policy: str = "test_end"
    max_windows: int | None = None
    window_indices: list[int] | None = None
    seed_stride: int = 1
    fail_fast: bool = True
    keep_optional_modules: bool = False

    tol_cost_abs: float = 1e-9
    out_dir: str | None = None


# --------------------------- Utils ---------------------------


def _strict_df_copy(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df.copy(deep=True))


def pit_guard_config_from_cfg(cfg: Mapping[str, Any]) -> PitGuardConfig:
    pcfg = cfg.get("pit_guard", {}) if isinstance(cfg.get("pit_guard"), Mapping) else {}
    win_idx = pcfg.get("window_indices")
    if isinstance(win_idx, str):
        try:
            win_idx = [int(x.strip()) for x in win_idx.split(",") if x.strip() != ""]
        except Exception:
            win_idx = None
    elif isinstance(win_idx, (list, tuple)):
        try:
            win_idx = [int(x) for x in win_idx]
        except Exception:
            win_idx = None
    else:
        win_idx = None
    return PitGuardConfig(
        enabled=bool(pcfg.get("enabled", False)),
        test_len=(int(pcfg["test_len"]) if pcfg.get("test_len") is not None else None),
        atol_equity=float(pcfg.get("atol_equity", 0.0)),
        rtol_equity=float(pcfg.get("rtol_equity", 0.0)),
        compare_trades=bool(pcfg.get("compare_trades", True)),
        compare_costs=bool(pcfg.get("compare_costs", True)),
        compare_forced=bool(pcfg.get("compare_forced", True)),
        availability_scope=str(pcfg.get("availability_scope", "all")),
        noise_sigma=float(pcfg.get("noise_sigma", 0.01)),
        seed=(int(pcfg["seed"]) if pcfg.get("seed") is not None else None),
        mode=str(pcfg.get("mode", "auto")),
        cut_policy=str(pcfg.get("cut_policy", "test_end")),
        max_windows=(
            int(pcfg["max_windows"]) if pcfg.get("max_windows") is not None else None
        ),
        window_indices=win_idx,
        seed_stride=int(pcfg.get("seed_stride", 1) or 1),
        fail_fast=bool(pcfg.get("fail_fast", True)),
        keep_optional_modules=bool(pcfg.get("keep_optional_modules", False)),
        tol_cost_abs=float(pcfg.get("tol_cost_abs", 1e-9)),
        out_dir=(str(pcfg.get("out_dir")) if pcfg.get("out_dir") else None),
    )


def sanitize_cfg_for_pit(cfg: Mapping[str, Any], pg: PitGuardConfig) -> dict[str, Any]:
    """
    Disable optional modules for PIT to allow deterministic and fast guards.
    """
    out = dict(cfg)
    if pg.keep_optional_modules:
        return out
    for k in ("bo", "overfit"):
        if isinstance(out.get(k), Mapping):
            d = dict(out.get(k) or {})
            d["enabled"] = False
            out[k] = d
    reporting = (
        dict(out.get("reporting") or {})
        if isinstance(out.get("reporting"), Mapping)
        else {}
    )
    reporting["mode"] = "core"
    test_tearsheet = (
        dict(reporting.get("test_tearsheet") or {})
        if isinstance(reporting.get("test_tearsheet"), Mapping)
        else {}
    )
    test_tearsheet["enabled"] = False
    reporting["test_tearsheet"] = test_tearsheet
    out["reporting"] = reporting
    return out


def _call_window_runner(
    runner: WindowRunner,
    *,
    start_idx: int,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    cfg: Mapping[str, Any],
    adv_map: Any | None,
    borrow_ctx: Any | None,
    availability_long: Any | None,
    availability_scope: str,
) -> Tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Call a window_runner robustly across signature variants.

    New signature (preferred):
      (start_idx, prices, prices_panel, pairs, cfg, adv_map, borrow_ctx, availability_long, availability_scope)

    Legacy signature:
      (start_idx, prices, pairs, cfg, adv_map, borrow_ctx, availability_long, availability_scope)
    """
    try:
        return runner(
            start_idx,
            prices,
            prices_panel,
            pairs,
            dict(cfg),
            adv_map,
            borrow_ctx,
            availability_long,
            availability_scope,
        )
    except TypeError:
        runner_legacy = cast(
            Callable[
                ...,
                Tuple[
                    pd.DataFrame,
                    dict[str, Any],
                    pd.DataFrame,
                    dict[str, Any],
                    dict[str, Any],
                ],
            ],
            runner,
        )
        return runner_legacy(
            start_idx,
            prices,
            pairs,
            dict(cfg),
            adv_map,
            borrow_ctx,
            availability_long,
            availability_scope,
        )


def _mutate_panel_future(
    prices_panel: pd.DataFrame | None,
    *,
    cut: int,
    rng: np.random.Generator,
    noise_sigma: float,
) -> pd.DataFrame | None:
    if (
        prices_panel is None
        or not isinstance(prices_panel, pd.DataFrame)
        or prices_panel.empty
    ):
        return prices_panel
    if cut >= len(prices_panel) or noise_sigma <= 0.0:
        return prices_panel
    mutated = prices_panel.copy(deep=True)
    tail = mutated.iloc[cut:]
    if tail.empty:
        return mutated
    try:
        tail_num = tail.apply(pd.to_numeric, errors="coerce").astype(float)
        noise = pd.DataFrame(
            rng.normal(loc=1.0, scale=float(noise_sigma), size=tail_num.shape),
            index=tail_num.index,
            columns=tail_num.columns,
        )
        mutated.iloc[cut:] = (tail_num * noise).astype(float)
    except Exception:
        pass
    return mutated


def _equity_series(stats_df: pd.DataFrame) -> pd.Series:
    if "equity" not in stats_df or stats_df["equity"].empty:
        return pd.Series(dtype=float)
    s = stats_df["equity"].copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    s = pd.to_numeric(s, errors="coerce")
    s.name = "equity"
    return s


def _allclose(a: pd.Series, b: pd.Series, atol: float, rtol: float) -> bool:
    if len(a) != len(b):
        return False
    if atol == 0.0 and rtol == 0.0:
        return a.equals(b)
    a_vals = a.to_numpy(dtype="float64", na_value=np.nan)
    b_vals = b.to_numpy(dtype="float64", na_value=np.nan)
    return np.allclose(a_vals, b_vals, atol=atol, rtol=rtol, equal_nan=True)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fingerprint_equity(eq: pd.Series) -> str:
    if eq is None or eq.empty:
        return _sha256_bytes(b"empty_equity")
    # Index -> int64[ns] as ndarray (robust across pandas versions)
    idx = pd.to_datetime(eq.index, errors="coerce")
    if hasattr(idx, "asi8"):
        i64 = idx.asi8  # ndarray[int64]
    else:
        i64 = np.asarray(idx, dtype="datetime64[ns]").astype("int64", copy=False)
    v = pd.to_numeric(eq, errors="coerce").astype("float64").to_numpy()
    buf = i64.tobytes() + v.tobytes()
    return _sha256_bytes(buf)


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _normalize_trade_df_for_fp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to deterministic, PIT-relevant fields and normalize types.
    Includes borrow/cost and forced-exit fields.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    cols_pref = [
        "pair",
        "symbol",
        "y_symbol",
        "x_symbol",
        "entry_date",
        "exit_date",
        "net_pnl",
        "gross_pnl",
        "fees",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "buyin_penalty_cost",
        "total_costs",
        "hard_exit",
        "hard_exit_reason",
    ]
    cols = _safe_cols(df, cols_pref)
    x = df[cols].copy()

    # Date fields -> datetime64[ns]
    for c in ("entry_date", "exit_date"):
        if c in x.columns:
            dt = to_naive_utc(pd.to_datetime(x[c], errors="coerce"))
            x[c] = dt

    # Numeric fields -> float64 (NaN -> 0.0 for deterministic bytes)
    num_cols = [
        c
        for c in (
            "net_pnl",
            "gross_pnl",
            "fees",
            "slippage_cost",
            "impact_cost",
            "borrow_cost",
            "buyin_penalty_cost",
            "total_costs",
        )
        if c in x.columns
    ]
    for c in num_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype("float64")

    # Bool/flags
    for c in ("hard_exit",):
        if c in x.columns:
            x[c] = pd.Series(x[c]).fillna(False).astype(bool)

    # Reasons/text
    for c in ("hard_exit_reason",):
        if c in x.columns:
            x[c] = pd.Series(x[c]).fillna("").astype(str)

    # Stable sorting
    sort_keys = [
        c
        for c in ("entry_date", "exit_date", "pair", "symbol", "y_symbol", "x_symbol")
        if c in x.columns
    ]
    if sort_keys:
        x = x.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    return x


def _fingerprint_trades(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return _sha256_bytes(b"empty_trades")
    x = _normalize_trade_df_for_fp(df)

    bufs: list[bytes] = []
    for c in x.columns:
        s = x[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = to_naive_utc(pd.to_datetime(s, errors="coerce"))
            arr = dt.values.astype("datetime64[ns]").view("i8")
            bufs.append(arr.tobytes())
        elif s.dtype == bool:
            bufs.append(s.astype("uint8").to_numpy().tobytes())
        elif pd.api.types.is_numeric_dtype(s):
            bufs.append(
                pd.to_numeric(s, errors="coerce").astype("float64").to_numpy().tobytes()
            )
        else:
            # Strings/objects -> utf-8 with separator + null byte as column delimiter
            bufs.append(
                "|".join("" if pd.isna(v) else str(v) for v in s).encode("utf-8")
            )
            bufs.append(b"\x00")
    return _sha256_bytes(b"".join(bufs))


# --------------------------- PIT Core ---------------------------


def assert_no_future_dependency(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None = None,
    pairs: dict[str, Any],
    cfg: Mapping[str, Any],
    runner: WindowRunner,
    adv_map: Any | None = None,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
    availability_scope: str | None = None,
    pg: PitGuardConfig | None = None,
    # Optional: separate runner for "mutated future"
    runner_mutated: WindowRunner | None = None,
    # Optional: fingerprint label (e.g. segment ID)
    label: str | None = None,
) -> dict[str, Any]:
    """
    **Extended PIT/look-ahead guard**:
    Mutates ONLY the future (> Train+Test) and expects IDENTICAL test outputs
    including borrow/cost/forced-exit fields.
    Fails with a precise AssertionError if a leak is present.

    If `runner_mutated` is provided, it is used in the second run.
    """
    if pg is None:
        pg = PitGuardConfig()

    eff_av_scope = str(availability_scope or pg.availability_scope or "all")

    # --- Input validation & base metric
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise AssertionError("prices.index must be a DatetimeIndex.")
    if "backtest" not in cfg:
        raise AssertionError("cfg['backtest'] is missing.")

    # Conservative mode only: derive cut from backtest.splits.{train,test}.
    splits = require_backtest_splits(
        cfg,
        keys=("train", "test"),
        err_cls=AssertionError,
        err_msg="cfg['backtest']['splits'].{train,test} is missing (conservative mode).",
    )

    idx = prices.index

    def _as_ts(v: Any) -> pd.Timestamp:
        return align_ts_to_index(v, idx)

    tr = require_split_start_end(
        splits,
        "train",
        err_cls=AssertionError,
        err_msg="backtest.splits.{train,test} must contain start/end.",
    )
    te = require_split_start_end(
        splits,
        "test",
        err_cls=AssertionError,
        err_msg="backtest.splits.{train,test} must contain start/end.",
    )

    tr0, tr1 = _as_ts(tr["start"]), _as_ts(tr["end"])
    te0, te1 = _as_ts(te["start"]), _as_ts(te["end"])
    if not (tr0 <= tr1 < te0 <= te1):
        raise AssertionError(
            "backtest.splits must be disjoint and ordered: train < test."
        )

    # Default: cut at test_end. Optional override: shorten test_len in bars.
    cut_policy = str(pg.cut_policy or "test_end").strip().lower()
    if cut_policy not in {"test_end", "exit_end"}:
        cut_policy = "test_end"
    end_raw = te1
    if cut_policy == "exit_end":
        exit_end = te.get("exit_end")
        if exit_end is not None:
            end_raw = _as_ts(exit_end)
        elif te.get("entry_end") is not None:
            end_raw = _as_ts(te.get("entry_end"))

    cut_ts = end_raw
    te_i0 = int(idx.searchsorted(te0, side="left"))
    te_i1 = int(idx.searchsorted(end_raw, side="right"))
    te_n_cfg = max(1, te_i1 - te_i0)
    te_n = (
        max(1, min(te_n_cfg, int(pg.test_len)))
        if (pg.test_len is not None)
        else te_n_cfg
    )
    cut = min(len(idx), te_i0 + te_n)
    if cut <= 0 or cut > len(idx):
        raise AssertionError(
            f"Implausible split lengths: test_len={te_n}, len(prices)={len(idx)}."
        )
    cut_ts = idx[cut - 1]  # last day in the (possibly shortened) test window

    # --- Seed handling
    seed = int(pg.seed) if pg.seed is not None else int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # ==========================
    # Run 1: reference
    # ==========================
    start_idx = 0
    res_ref = _call_window_runner(
        runner,
        start_idx=start_idx,
        prices=prices,
        prices_panel=prices_panel,
        pairs=pairs,
        cfg=cfg,
        adv_map=adv_map,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
        availability_scope=eff_av_scope,
    )
    stats_te_ref: pd.DataFrame = res_ref[2]
    trades_te_ref = cast(pd.DataFrame | None, res_ref[4].get("trades_te"))

    eq_ref = _equity_series(stats_te_ref)
    fp_ref_eq = _fingerprint_equity(eq_ref)
    fp_ref_tr = _fingerprint_trades(
        trades_te_ref if isinstance(trades_te_ref, pd.DataFrame) else pd.DataFrame()
    )
    _LOG.info(
        "PIT[%s] REF  eq=%s trades=%s | n=%s",
        (label or "seg0"),
        fp_ref_eq,
        fp_ref_tr,
        0 if trades_te_ref is None else len(trades_te_ref),
    )

    # ==========================
    # Mutate future (prices)
    # ==========================
    mutated_prices = _strict_df_copy(prices)
    if cut < len(mutated_prices) and pg.noise_sigma > 0.0:
        tail = mutated_prices.iloc[cut:]
        if not tail.empty:
            noise = pd.DataFrame(
                rng.normal(loc=1.0, scale=float(pg.noise_sigma), size=tail.shape),
                index=tail.index,
                columns=tail.columns,
            )
            mutated_prices.iloc[cut:] = (
                tail.astype(float) * noise.astype(float)
            ).astype(float)

    mutated_panel = _mutate_panel_future(
        prices_panel,
        cut=cut,
        rng=rng,
        noise_sigma=float(pg.noise_sigma),
    )

    # ==========================
    # Run 2: mutated future
    # ==========================
    run2 = runner_mutated if (runner_mutated is not None) else runner

    res_new = _call_window_runner(
        run2,
        start_idx=start_idx,
        prices=mutated_prices,
        prices_panel=mutated_panel,
        pairs=pairs,
        cfg=cfg,
        adv_map=adv_map,
        borrow_ctx=borrow_ctx,
        availability_long=availability_long,
        availability_scope=eff_av_scope,
    )
    stats_te_new: pd.DataFrame = res_new[2]
    trades_te_new = cast(pd.DataFrame | None, res_new[4].get("trades_te"))

    eq_new = _equity_series(stats_te_new)
    fp_new_eq = _fingerprint_equity(eq_new)
    fp_new_tr = _fingerprint_trades(
        trades_te_new if isinstance(trades_te_new, pd.DataFrame) else pd.DataFrame()
    )
    _LOG.info(
        "PIT[%s] MUT  eq=%s trades=%s | n=%s",
        (label or "seg0"),
        fp_new_eq,
        fp_new_tr,
        0 if trades_te_new is None else len(trades_te_new),
    )

    # --- Equity comparison
    if not _allclose(eq_ref, eq_new, pg.atol_equity, pg.rtol_equity):
        msg = [
            "PIT guard failed: test equity changed when mutating the future.",
            f"- len(eq_ref)={len(eq_ref)}, len(eq_new)={len(eq_new)}",
            f"- fp_ref_eq={fp_ref_eq}",
            f"- fp_new_eq={fp_new_eq}",
        ]
        if len(eq_ref) == len(eq_new) and len(eq_ref) > 0:
            diff = np.where(
                ~np.isclose(
                    eq_ref.to_numpy(dtype="float64", na_value=np.nan),
                    eq_new.to_numpy(dtype="float64", na_value=np.nan),
                    atol=max(pg.atol_equity, 1e-18),
                    rtol=max(pg.rtol_equity, 1e-18),
                    equal_nan=True,
                )
            )[0]
            if diff.size:
                i0 = int(diff[0])
                msg += [
                    f"- first mismatch @ idx={i0} (ref={eq_ref.iloc[i0]!r}, new={eq_new.iloc[i0]!r})",
                    f"- ref_idx={eq_ref.index[i0]!r}",
                ]
        raise AssertionError("\n".join(msg))

    # --- Trade presence/emptiness hardening (before content comparison)
    if pg.compare_trades and ((trades_te_ref is None) != (trades_te_new is None)):
        raise AssertionError(
            "PIT guard failed: trades_te presence differs between REF and MUT "
            f"(ref_is_none={trades_te_ref is None}, mut_is_none={trades_te_new is None})."
        )

    if (
        pg.compare_trades
        and (trades_te_ref is not None)
        and (trades_te_new is not None)
    ):
        if bool(trades_te_ref.empty) != bool(trades_te_new.empty):
            # Small diagnostic log (fingerprints help with triage)
            _LOG.error(
                "PIT[%s] trades_te emptiness differs (ref_empty=%s, mut_empty=%s) | "
                "fp_ref_trades=%s fp_new_trades=%s",
                (label or "seg0"),
                bool(trades_te_ref.empty),
                bool(trades_te_new.empty),
                _fingerprint_trades(trades_te_ref),
                _fingerprint_trades(trades_te_new),
            )
            raise AssertionError(
                "PIT guard failed: trades_te emptiness differs between REF and MUT."
            )

    # --- Trade comparison
    if (
        pg.compare_trades
        and (trades_te_ref is not None)
        and (trades_te_new is not None)
    ):
        a = _normalize_trade_df_for_fp(trades_te_ref)
        b = _normalize_trade_df_for_fp(trades_te_new)

        # 1) Same structure?
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            raise AssertionError(
                "\n".join(
                    [
                        "PIT guard failed: test trades changed structure when mutating the future.",
                        f"- n_ref={len(a)} n_new={len(b)}",
                        f"- cols_ref={list(a.columns)}",
                        f"- cols_new={list(b.columns)}",
                        f"- fp_ref_trades={fp_ref_tr}",
                        f"- fp_new_trades={fp_new_tr}",
                    ]
                )
            )

        # 2) Same contents?
        if not a.equals(b):
            # Diagnostic: first differing row
            neq = (a != b).any(axis=1)
            try:
                idxs = np.where(neq.to_numpy())[0]
            except Exception:
                idxs = np.array([], dtype=int)
            msg = [
                "PIT guard failed: test trades changed when mutating the future.",
                f"- n_rows={len(a)}",
                f"- fp_ref_trades={fp_ref_tr}",
                f"- fp_new_trades={fp_new_tr}",
            ]
            if len(idxs):
                i0 = int(idxs[0])
                msg += [
                    f"- first mismatch row={i0}",
                    f"- ref_row={a.iloc[i0].to_dict()}",
                    f"- new_row={b.iloc[i0].to_dict()}",
                ]
            raise AssertionError("\n".join(msg))

    # --- Cost/borrow/forced checks (robust fields)
    def _safe_num(s: pd.Series) -> float:
        return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())

    if (trades_te_ref is not None) and (trades_te_new is not None):
        ref = trades_te_ref
        new = trades_te_new

        # Costs
        if pg.compare_costs:
            for col in ("borrow_cost", "buyin_penalty_cost", "total_costs", "net_pnl"):
                if (col in ref.columns) and (col in new.columns):
                    r = _safe_num(pd.Series(ref[col]))
                    n = _safe_num(pd.Series(new[col]))
                    if not np.isclose(r, n, atol=float(pg.tol_cost_abs), rtol=0.0):
                        raise AssertionError(
                            f"PIT guard failed: aggregated {col} changed with future mutation (ref={r:.12g}, new={n:.12g}, tol={pg.tol_cost_abs})."
                        )

        # Forced-Exit
        if (
            pg.compare_forced
            and ("hard_exit" in ref.columns)
            and ("hard_exit" in new.columns)
        ):
            r = int(pd.Series(ref["hard_exit"]).fillna(False).astype(bool).sum())
            n = int(pd.Series(new["hard_exit"]).fillna(False).astype(bool).sum())
            if r != n:
                raise AssertionError(
                    f"PIT guard failed: hard_exit count changed (ref={r}, new={n})."
                )

    payload = {
        "label": label or "seg0",
        "cut": str(pd.Timestamp(cut_ts).date()),
        "seed": int(pg.seed if pg.seed is not None else cfg.get("seed", 42)),
        "test_len": int(te_n),
        "availability_scope": pg.availability_scope,
        "noise_sigma": float(pg.noise_sigma),
        "eq_fp": fp_ref_eq,
        "trades_fp": fp_ref_tr,
        "status": "PASSED",
    }
    try:
        import json
        import os

        out_dir = pg.out_dir or os.path.join("runs", "results", "pit_guard")
        os.makedirs(out_dir, exist_ok=True)
        payload["ts"] = utc_now().isoformat().replace("+00:00", "Z")
        with open(os.path.join(out_dir, f"pit_{(label or 'seg0')}.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass  # Logging is enough; the artifact is optional

    # --- Fingerprint-Log (BestÃ¤tigung)
    _LOG.info(
        "PIT[%s] PASSED  | eq=%s trades=%s | cut=%s",
        (label or "seg0"),
        fp_ref_eq,
        fp_ref_tr,
        pd.Timestamp(cut_ts).date(),
    )
    # If we reach this point, the guard passed
    return payload


def _window_cfg_from_splits(
    cfg: Mapping[str, Any],
    *,
    splits: dict[str, dict[str, str]],
    include_analysis: bool,
    calendar: pd.DatetimeIndex | None,
) -> dict[str, Any]:
    out = dict(cfg)
    bt = dict(out.get("backtest") or {})
    new_splits = dict(splits)

    if include_analysis and "analysis" not in new_splits:
        if calendar is not None and "train" in new_splits:
            try:
                analysis = synthesize_analysis_split(
                    calendar, pd.Timestamp(new_splits["train"]["start"])
                )
                if analysis is not None:
                    new_splits["analysis"] = analysis
            except Exception:
                pass

    bt["splits"] = new_splits
    # Disable WF in the per-window cfg so the runner does not generate windows again.
    if isinstance(bt.get("walkforward"), Mapping):
        wf = dict(bt.get("walkforward") or {})
        wf["enabled"] = False
        bt["walkforward"] = wf
    out["backtest"] = bt
    return out


def assert_no_future_dependency_walkforward(
    *,
    prices: pd.DataFrame,
    prices_panel: pd.DataFrame | None,
    pairs: dict[str, Any],
    cfg: Mapping[str, Any],
    runner: WindowRunner,
    adv_map: Any | None = None,
    borrow_ctx: Any | None = None,
    availability_long: Any | None = None,
    availability_scope: str | None = None,
    pg: PitGuardConfig | None = None,
    runner_mutated: WindowRunner | None = None,
    windows: list[Any] | None = None,
    calendar: pd.DatetimeIndex | None = None,
    include_analysis: bool = False,
) -> list[dict[str, Any]]:
    """
    Walkforward PIT guard: iterates over all WF windows and runs a
    future-mutation check per window.

    Returns a list of result payloads; raises AssertionError on failure
    (`fail_fast`) or in aggregated form at the end.
    """
    import json
    import os
    from backtest.calendars import build_trading_calendar
    from backtest.windowing.walkforward import generate_walkforward_windows_from_cfg

    if pg is None:
        pg = PitGuardConfig()

    wf_cfg = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), Mapping) else {}
    wf = (
        wf_cfg.get("walkforward", {})
        if isinstance(wf_cfg.get("walkforward"), Mapping)
        else {}
    )
    if windows is None:
        if not bool(wf.get("enabled", False)):
            raise AssertionError(
                "pit_guard.walkforward requested but backtest.walkforward.enabled is False."
            )
        cal = calendar
        if cal is None:
            data_cfg = (
                cfg.get("data", {}) if isinstance(cfg.get("data"), Mapping) else {}
            )
            cal = build_trading_calendar(
                {c: prices[c] for c in prices.columns},
                calendar_name=data_cfg.get("calendar_name", "XNYS"),
            )
        windows, meta = generate_walkforward_windows_from_cfg(calendar=cal, cfg=cfg)
    else:
        meta = {"n_windows": len(windows)}
        cal = calendar

    if windows is None:
        windows = []

    def _win_index(win: Any, default: int = -1) -> int:
        if hasattr(win, "i"):
            try:
                return int(getattr(win, "i"))
            except Exception:
                return default
        if isinstance(win, Mapping):
            try:
                return int(win.get("i", default))
            except Exception:
                return default
        return default

    def _win_attr(win: Any, key: str) -> Any:
        if hasattr(win, key):
            try:
                return getattr(win, key)
            except Exception:
                return None
        if isinstance(win, Mapping):
            return win.get(key)
        return None

    # Select window indices
    if pg.window_indices:
        want = {int(i) for i in pg.window_indices}
        windows_sel = [w for w in windows if _win_index(w, -1) in want]
    else:
        windows_sel = list(windows)

    if pg.max_windows is not None and pg.max_windows > 0:
        windows_sel = windows_sel[: int(pg.max_windows)]

    if not windows_sel:
        raise AssertionError("pit_guard: no walkforward windows selected.")

    global_end = None
    try:
        last = windows[-1]
        global_end = _win_attr(last, "test_end")
        if global_end is None and isinstance(last, Mapping):
            global_end = (last.get("test", {}) or {}).get("end")
    except Exception:
        global_end = None

    out_dir = pg.out_dir or os.path.join("runs", "results", "pit_guard")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    base_seed = int(pg.seed if pg.seed is not None else cfg.get("seed", 42))
    stride = int(pg.seed_stride or 1)

    for w in windows_sel:
        try:
            w_i = _win_index(w, 0)
        except Exception:
            w_i = int(len(results))
        label = f"WF-{w_i:03d}"

        # Window-Splits bauen
        splits = None
        if hasattr(w, "as_splits"):
            splits = w.as_splits()
        elif isinstance(w, Mapping):
            tr = (w.get("train") or {}) if isinstance(w.get("train"), Mapping) else {}
            te = (w.get("test") or {}) if isinstance(w.get("test"), Mapping) else {}
            if tr and te:
                splits = {
                    "train": {"start": str(tr.get("start")), "end": str(tr.get("end"))},
                    "test": {"start": str(te.get("start")), "end": str(te.get("end"))},
                }
        if splits is None:
            raise AssertionError(f"pit_guard: invalid window format for {label}")

        # Carry-Over: entry_end/exit_end fÃƒÂ¼r Strategien (falls gewÃƒÂ¼nscht)
        if isinstance(splits.get("test"), Mapping):
            test_split = dict(splits.get("test") or {})
            try:
                test_end = _win_attr(w, "test_end") or test_split.get("end")
            except Exception:
                test_end = test_split.get("end")
            if test_end:
                test_split["entry_end"] = str(pd.Timestamp(test_end).date())
            if global_end is not None:
                test_split["exit_end"] = str(pd.Timestamp(global_end).date())
            splits["test"] = test_split

        cfg_w = _window_cfg_from_splits(
            cfg, splits=splits, include_analysis=include_analysis, calendar=cal
        )
        cfg_w = sanitize_cfg_for_pit(cfg_w, pg)
        pg_w = replace(pg, seed=base_seed + w_i * stride)

        try:
            payload = assert_no_future_dependency(
                prices=prices,
                prices_panel=prices_panel,
                pairs=pairs,
                cfg=cfg_w,
                runner=runner,
                adv_map=adv_map,
                borrow_ctx=borrow_ctx,
                availability_long=availability_long,
                availability_scope=availability_scope,
                pg=pg_w,
                runner_mutated=runner_mutated,
                label=label,
            )
            if isinstance(payload, dict):
                results.append(payload)
        except AssertionError as e:
            fail = {"label": label, "error": str(e)}
            failures.append(fail)
            if pg.fail_fast:
                raise

    summary = {
        "mode": "walkforward",
        "windows_total": int(meta.get("n_windows", len(windows))),
        "windows_checked": int(len(windows_sel)),
        "failures": failures,
        "ts": utc_now().isoformat().replace("+00:00", "Z"),
    }
    try:
        with open(os.path.join(out_dir, "pit_walkforward_summary.json"), "w") as f:
            json.dump({"summary": summary, "windows": results}, f, indent=2)
    except Exception:
        pass

    if failures:
        msg = "PIT guard failed for windows: " + ", ".join(
            [f["label"] for f in failures]
        )
        raise AssertionError(msg)
    return results
