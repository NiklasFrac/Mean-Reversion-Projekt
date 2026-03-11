from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest


def test_legacy_adv_cache_snapshot_is_readable() -> None:
    adv_cache = Path("backtest/data/ticker_adv.pkl")
    meta_cache = Path("backtest/data/ticker_metadata.pkl")

    if not adv_cache.exists() or not meta_cache.exists():
        pytest.skip("Legacy backtest/data artifacts are not present in this workspace.")

    with adv_cache.open("rb") as fh:
        adv_map = pickle.load(fh)

    assert isinstance(adv_map, dict)

    adv_series = pd.Series(
        [v.get("adv", 0.0) if isinstance(v, dict) else v for v in adv_map.values()],
        dtype="float64",
    )
    assert adv_series.notna().any()

    meta = pd.read_pickle(meta_cache)
    meta_syms = {s.upper() for s in meta.index.astype(str).tolist()}
    adv_syms = {s.upper() for s in adv_map.keys()}
    if meta_syms and adv_syms:
        # Diagnostic guard: legacy artifacts should not be entirely disjoint.
        assert len(meta_syms & adv_syms) > 0
