from __future__ import annotations

import pandas as pd
import pytest

from backtest.optimize.paper_bo_parts import pipeline as pipeline_mod
from backtest.optimize.paper_bo_parts import sim as sim_mod
from backtest.run import bo_runner as bo_mod


def test_slice_series_tz_alignment() -> None:
    cal = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    s = pd.Series([1, 2, 3, 4, 5], index=idx)

    out = pipeline_mod._slice_series(s, cal[0], cal[-1], cal=cal)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == str(cal.tz)
    assert len(out) == 5


def test_positions_max_hold_days_zero_unbounded() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    z = pd.Series([0.0, -3.0, -3.0, -3.0, -3.0], index=idx)

    pos = sim_mod._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=5.0,
        max_hold_days=0,
        cooldown_days=0,
    )

    assert pos.iloc[0] == 0
    assert (pos.iloc[1:] == 1).all()


def test_prices_frame_merges_symbol_series() -> None:
    idx1 = pd.date_range("2024-01-01", periods=3, freq="D")
    idx2 = pd.date_range("2024-01-04", periods=3, freq="D")

    y1 = pd.Series([1.0, 2.0, 3.0], index=idx1)
    x1 = pd.Series([10.0, 11.0, 12.0], index=idx1)
    y2 = pd.Series([4.0, 5.0, 6.0], index=idx2)
    x2 = pd.Series([20.0, 21.0, 22.0], index=idx2)

    pairs_data = {
        "AAA_BBB": {
            "prices": pd.DataFrame({"y": y1, "x": x1}),
            "meta": {"t1": "AAA", "t2": "BBB"},
        },
        "AAA_CCC": {
            "prices": pd.DataFrame({"y": y2, "x": x2}),
            "meta": {"t1": "AAA", "t2": "CCC"},
        },
    }

    prices_df, pairs_map = bo_mod._prices_frame_from_pairs_data(pairs_data)

    assert "AAA" in prices_df.columns
    assert len(prices_df.index) == 6
    assert prices_df["AAA"].notna().sum() == 6
    assert set(pairs_map.keys()) == {"AAA_BBB", "AAA_CCC"}


def test_build_train_inputs_uses_runtime_pair_prefilter_cfg(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0, 4.0, 5.0],
            "BBB": [1.1, 2.1, 3.1, 4.1, 5.1],
        },
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    cfg = {
        "backtest": {
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            }
        },
        "pair_prefilter": {
            "prefilter_active": True,
            "coint_alpha": 1.0,
            "min_obs": 2,
            "half_life": {
                "min_days": 5,
                "max_days": 60,
                "max_hold_multiple": 2.0,
                "min_derived_days": 5,
            },
        },
    }
    seen: dict[str, float] = {}

    def fake_evaluate_pair_cointegration(
        df: pd.DataFrame, *, coint_alpha: float, min_obs: int, half_life_cfg
    ):
        seen["coint_alpha"] = float(coint_alpha)
        seen["min_obs"] = float(min_obs)
        assert list(df.columns) == ["y", "x"]
        assert isinstance(half_life_cfg, dict)
        return {
            "passed": True,
            "beta": 1.1,
            "z_window": 7,
            "max_hold_days": 14,
            "half_life": 7.0,
        }

    monkeypatch.setattr(
        pipeline_mod, "evaluate_pair_cointegration", fake_evaluate_pair_cointegration
    )
    out, cal = pipeline_mod._build_train_inputs(prices=prices, pairs=pairs, cfg=cfg)

    assert "AAA-BBB" in out
    assert len(cal) == len(idx)
    assert seen == {"coint_alpha": 1.0, "min_obs": 2.0}
    assert out["AAA-BBB"]["z_window"] == 7
    assert out["AAA-BBB"]["max_hold_days"] == 14


def test_build_train_inputs_rejects_non_positive_beta_without_prefilter() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "AAA": [5.0, 4.0, 3.0, 2.0, 1.0],
            "BBB": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}
    cfg = {
        "backtest": {
            "splits": {
                "train": {"start": str(idx[0].date()), "end": str(idx[-1].date())},
            }
        },
        "pair_prefilter": {"prefilter_active": False},
    }

    with pytest.raises(ValueError, match="No valid pairs"):
        pipeline_mod._build_train_inputs(prices=prices, pairs=pairs, cfg=cfg)
