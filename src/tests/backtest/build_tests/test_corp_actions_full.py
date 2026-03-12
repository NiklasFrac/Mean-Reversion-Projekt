# src/backtest/tests/test_corp_actions_full.py
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------- Resolve paths robustly ----------
# This file lives under: src/backtest/tests/test_corp_actions_full.py
# -> BASE = src/backtest/src
BASE = Path(__file__).resolve().parents[1] / "src"
CAB_PATH = BASE / "corp_actions_builder.py"
RUNNER_CANDIDATES = [
    BASE / "runner_corp.py",
    BASE / "runner_corp_actions.py",
]


def _import_by_path(mod_name: str, file_path: Path) -> types.ModuleType:
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


cab = _import_by_path("corp_actions_builder", CAB_PATH)

_runner_mod = None
for cand in RUNNER_CANDIDATES:
    if cand.exists():
        _runner_mod = _import_by_path("runner_corp_mod", cand)
        break
if _runner_mod is None:
    raise FileNotFoundError(
        "No runner found. Expected one of the following paths:\n"
        + "\n".join(str(p) for p in RUNNER_CANDIDATES)
    )


# ---------- kleine Hilfsfunktionen ----------
def _write_universe_meta(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["symbol", "yf_symbol", "exchange", "delist_date"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _read_ca(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(
            columns=["symbol", "date", "type", "factor", "amount", "notes"]
        )


def _run_runner_pipeline(cfg: dict) -> None:
    """
    Tries runner_corp.run_corp_pipeline(cfg) first.
    If unavailable, emulates the necessary parts via corp_actions_builder.
    """
    fn = getattr(_runner_mod, "run_corp_pipeline", None)
    if callable(fn):
        # use the runner directly - return value is not needed
        fn(cfg)
        return

    # Fallback: build only corp_actions via the builder (matches the core step)
    data = cfg.get("data", {}) or {}
    ca_path = data.get("corporate_actions_path")
    price_path = data.get("prices_path")
    if not ca_path or not price_path:
        raise AssertionError(
            "cfg.data.corporate_actions_path / cfg.data.prices_path is missing."
        )

    cab.ensure_and_seed_corporate_actions_file(
        path=ca_path,
        price_path=price_path,
        vendor=(cfg.get("data", {}) or {}).get("corp_actions_vendor") or {},
        universe_meta=data.get("universe_meta"),
        infer_splits=(cfg.get("corp_actions_seed", {}) or {}).get("infer_splits", True),
        infer_delist=(cfg.get("corp_actions_seed", {}) or {}).get("infer_delist", True),
        split_tol=float(
            (cfg.get("corp_actions_seed", {}) or {}).get("split_tol", 0.02)
        ),
        min_price_for_split=float(
            (cfg.get("corp_actions_seed", {}) or {}).get("min_price_for_split", 1.0)
        ),
    )


# ==============================================================================
# 1) OFFLINE / DETERMINISTISCHER END-TO-END TEST (schnell, ohne Internet)
# ==============================================================================
def test_offline_end_to_end_runner_corp(tmp_path: Path):
    """
    Mini universe:
      - SPLT: artificial 3:1 split inside the window (heuristic must trigger).
      - CONST: constant, no events.
      - ZDEL: constant, but delisted via universe_meta.
    Expectation: >=1 split + >=1 delisting in the final CSV.
    """
    ca_path = tmp_path / "corp_actions.csv"
    prices_path = tmp_path / "prices.pkl"
    umeta_path = tmp_path / "universe_meta.csv"

    # price window
    idx = pd.bdate_range("2022-01-03", periods=500)
    base = np.linspace(90.0, 110.0, len(idx))
    splt = pd.Series(base, index=idx)
    cut = idx[250]
    splt.loc[cut:] = splt.loc[cut:] / 3.0  # 3:1 Split simuliert

    const = pd.Series(np.full(len(idx), 50.0), index=idx)
    zdel = pd.Series(np.full(len(idx), 20.0), index=idx)
    prices = pd.DataFrame({"SPLT": splt, "CONST": const, "ZDEL": zdel}, index=idx)
    prices.to_pickle(prices_path)

    # universe_meta with a delist for ZDEL inside the window
    _write_universe_meta(
        umeta_path,
        [
            {
                "symbol": "SPLT",
                "yf_symbol": "SPLT",
                "exchange": "XNYS",
                "delist_date": "",
            },
            {
                "symbol": "CONST",
                "yf_symbol": "CONST",
                "exchange": "XNYS",
                "delist_date": "",
            },
            {
                "symbol": "ZDEL",
                "yf_symbol": "ZDEL",
                "exchange": "XNYS",
                "delist_date": idx[-15].date().isoformat(),
            },
        ],
    )

    # Runner configuration: OFFLINE only (no vendors)
    cfg = {
        "data": {
            "prices_path": str(prices_path),
            "apply_corporate_actions": True,
            "corporate_actions_path": str(ca_path),
            "universe_meta": str(umeta_path),
            "auto_build_universe_meta": False,
            "corp_actions_vendor": {},  # explicitly empty
        },
        "corp_actions_seed": {
            "infer_splits": True,
            "infer_delist": True,
            "split_tol": 0.05,
            "min_price_for_split": 1.0,
        },
    }

    _run_runner_pipeline(cfg)

    ca = _read_ca(ca_path)
    assert not ca.empty, "corp_actions.csv should contain events."
    tcounts = ca["type"].str.lower().value_counts()
    assert tcounts.get("split", 0) >= 1, "Expected >=1 split event (heuristic)."
    assert tcounts.get("delist", 0) >= 1, "Expected >=1 delisting event (universe_meta)."


# ==============================================================================
# 2) OPTIONAL: ONLINE INTEGRATION (yfinance) - skipped on network issues
# ==============================================================================
@pytest.mark.integration
def test_online_yfinance_small_universe(tmp_path: Path):
    """
    Small YF check (AAPL/MSFT/TSLA). Expects at least 1 dividend OR 1 split.
    Skips automatically if yfinance/internet is unavailable.
    """
    try:
        import yfinance as yf  # noqa

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _ = yf.Ticker("AAPL").history(period="5d")
    except Exception:
        pytest.skip("yfinance/internet unavailable - skipping test.")

    # Prices only define the window
    idx = pd.bdate_range("2022-01-03", "2025-06-01")
    prices = pd.DataFrame({"AAPL": 150.0, "MSFT": 300.0, "TSLA": 200.0}, index=idx)
    pkl_path = tmp_path / "prices_online.pkl"
    prices.to_pickle(pkl_path)

    ca_path = tmp_path / "corp_actions.csv"
    cab.ensure_and_seed_corporate_actions_file(
        path=ca_path,
        price_path=pkl_path,
        vendor={"yfinance": True, "yf_throttle_sec": 0.0, "probe_suffixes": []},
        infer_splits=False,
        infer_delist=False,
    )
    ca = _read_ca(ca_path)
    if ca.empty:
        pytest.skip("Vendor currently returned no actions - time/network dependent.")
    tcounts = ca["type"].str.lower().value_counts()
    assert (tcounts.get("dividend", 0) > 0) or (tcounts.get("split", 0) > 0), (
        "Expect at least 1 dividend OR 1 split in the window."
    )


# ==============================================================================
# 3) FAST OFFLINE BUILDER UNIT TEST (without runner)
# ==============================================================================
def test_offline_builder_minimal(tmp_path: Path):
    """
    Pure builder test (without runner):
      - heuristic split for SPLT
      - delisting for ZDEL via universe_meta
    """
    ca_path = tmp_path / "corp_actions.csv"
    prices_path = tmp_path / "prices.pkl"
    umeta_path = tmp_path / "universe_meta.csv"

    idx = pd.bdate_range("2021-01-04", periods=300)
    base = 90 + np.sin(np.linspace(0, 6, len(idx))) * 2
    splt = pd.Series(base, index=idx)
    spl_cut = idx[120]
    splt.loc[spl_cut:] = splt.loc[spl_cut:] / 2.0  # 2:1 Split
    const = pd.Series(np.full(len(idx), 42.0), index=idx)
    zdel = pd.Series(np.full(len(idx), 13.0), index=idx)

    prices = pd.DataFrame({"SPLT": splt, "CONST": const, "ZDEL": zdel}, index=idx)
    prices.to_pickle(prices_path)

    _write_universe_meta(
        umeta_path,
        [
            {
                "symbol": "SPLT",
                "yf_symbol": "SPLT",
                "exchange": "XNYS",
                "delist_date": "",
            },
            {
                "symbol": "CONST",
                "yf_symbol": "CONST",
                "exchange": "XNYS",
                "delist_date": "",
            },
            {
                "symbol": "ZDEL",
                "yf_symbol": "ZDEL",
                "exchange": "XNYS",
                "delist_date": idx[-10].date().isoformat(),
            },
        ],
    )

    cab.ensure_and_seed_corporate_actions_file(
        path=ca_path,
        price_path=prices_path,
        vendor={},  # offline
        universe_meta=str(umeta_path),
        infer_splits=True,
        infer_delist=True,
        split_tol=0.05,
        min_price_for_split=1.0,
    )

    ca = _read_ca(ca_path)
    assert not ca.empty, "corp_actions.csv should contain events."
    types_ = ca["type"].str.lower().value_counts()
    assert types_.get("split", 0) >= 1, "Expected >=1 heuristic split."
    assert types_.get("delist", 0) >= 1, "Expected >=1 delisting from universe_meta."
