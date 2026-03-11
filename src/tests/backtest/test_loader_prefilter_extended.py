import numpy as np
import pandas as pd
import pytest

from backtest import loader


def test_prefilter_ok_uses_dataframe_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    y = pd.Series(np.linspace(10, 11, len(idx)), index=idx)
    x = pd.Series(np.linspace(20, 19, len(idx)), index=idx)

    def pf_df(df: pd.DataFrame) -> bool:
        assert list(df.columns) == ["y", "x"]
        return False

    monkeypatch.setattr(loader, "pair_prefilter", pf_df)
    assert loader._prefilter_ok(y, x) is False


def test_prepare_pairs_with_prefilter_and_adv(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(
        {"AAA": np.linspace(10, 12, len(idx)), "BBB": np.linspace(20, 18, len(idx))},
        index=idx,
    )
    pairs = {"AAA-BBB": {"t1": "AAA", "t2": "BBB"}}

    monkeypatch.setattr(
        loader,
        "evaluate_pair_cointegration",
        lambda *_args, **_kwargs: {
            "passed": True,
            "z_window": 8,
            "max_hold_days": 16,
            "half_life": 8.0,
        },
    )
    out = loader.prepare_pairs_data(
        prices,
        pairs,
        disable_prefilter=False,
        pair_adv_mode="geometric",
        adv_map={"AAA": 1_000.0, "BBB": 2_000.0},
        prefilter_range=(idx[5], idx[-1]),
    )
    assert "AAA-BBB" in out
    assert out["AAA-BBB"]["meta"]["adv_pair_mode"] == "geometric"
    assert out["AAA-BBB"]["meta"]["cointegration"]["max_hold_days"] == 16
