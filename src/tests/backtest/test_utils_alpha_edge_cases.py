import numpy as np
import pandas as pd
import pytest

from backtest.utils import alpha


def test_pair_prefilter_respects_explicit_runtime_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [1.1, 2.1]}, index=idx)
    seen: dict[str, float] = {}

    def fake_coint(y: pd.Series, x: pd.Series):
        seen["n_obs"] = float(len(x))
        return 0.0, 0.5, {}

    monkeypatch.setattr(alpha, "coint", fake_coint)

    assert alpha.pair_prefilter(df) is False
    assert seen == {}

    def passing_coint(y: pd.Series, x: pd.Series):
        seen["n_obs"] = float(len(x))
        return 0.0, 0.01, {}

    monkeypatch.setattr(alpha, "coint", passing_coint)
    assert alpha.pair_prefilter(df, coint_alpha=1.0, min_obs=2) is True
    assert seen["n_obs"] == pytest.approx(2.0)


def test_alpha_safe_coint_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    assert (
        alpha.safe_coint(pd.Series([], dtype=float), pd.Series([], dtype=float))
        is False
    )
    monkeypatch.setattr(alpha, "coint", boom)
    assert alpha.safe_coint(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) is False


def test_compute_spread_zscore_fallback_and_ols_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = pd.Series([1.0])
    x = pd.Series([1.0])
    spread, z, beta = alpha.compute_spread_zscore(y, x)
    assert beta.eq(1.0).all()
    assert len(spread) == 1 and len(z) == 1

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(alpha, "OLS", boom)
    y2 = pd.Series([1.0, 2.0])
    x2 = pd.Series([1.0, 2.0])
    _spread2, _z2, beta2 = alpha.compute_spread_zscore(y2, x2)
    assert beta2.eq(1.0).all()


def test_pair_prefilter_invalid_inputs() -> None:
    assert alpha.pair_prefilter([1, 2, 3]) is False
    assert alpha.pair_prefilter(pd.DataFrame({"y": [1.0, 2.0]})) is False
    df = pd.DataFrame({"y": [np.nan, np.nan], "x": [np.nan, np.nan]})
    assert alpha.pair_prefilter(df) is False


def test_evaluate_pair_cointegration_derives_half_life_runtime_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resid = 100.0 * np.power(0.93, np.arange(10, dtype=float))
    df = pd.DataFrame({"y": resid, "x": np.zeros_like(resid)})

    class _FakeOLS:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            return type("Fit", (), {"params": np.array([0.0, 0.0])})()

    monkeypatch.setattr(alpha, "coint", lambda *_args, **_kwargs: (0.0, 0.01, {}))
    monkeypatch.setattr(alpha, "OLS", _FakeOLS)

    out = alpha.evaluate_pair_cointegration(
        df,
        coint_alpha=0.05,
        min_obs=5,
        half_life_cfg={
            "min_days": 5,
            "max_days": 60,
            "max_hold_multiple": 2.0,
            "min_derived_days": 5,
        },
    )

    assert out["passed"] is True
    assert out["reject_reason"] is None
    assert out["z_window"] == 10
    assert out["max_hold_days"] == 19
    assert out["half_life"] == pytest.approx(9.551337, rel=1e-5)


def test_evaluate_pair_cointegration_rejects_invalid_half_life_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOLS:
        def __init__(self, *_args, **_kwargs):
            pass

        def fit(self):
            return type("Fit", (), {"params": np.array([0.0, 0.0])})()

    monkeypatch.setattr(alpha, "coint", lambda *_args, **_kwargs: (0.0, 0.01, {}))
    monkeypatch.setattr(alpha, "OLS", _FakeOLS)

    fast = pd.DataFrame(
        {"y": 10.0 * np.power(0.5, np.arange(8, dtype=float)), "x": np.zeros(8)}
    )
    fast_out = alpha.evaluate_pair_cointegration(
        fast,
        min_obs=5,
        half_life_cfg={
            "min_days": 5,
            "max_days": 60,
            "max_hold_multiple": 2.0,
            "min_derived_days": 5,
        },
    )
    assert fast_out["reject_reason"] == "half_life_too_fast"

    non_mr = pd.DataFrame(
        {"y": np.power(2.0, np.arange(8, dtype=float)), "x": np.zeros(8)}
    )
    non_mr_out = alpha.evaluate_pair_cointegration(
        non_mr,
        min_obs=5,
        half_life_cfg={
            "min_days": 5,
            "max_days": 60,
            "max_hold_multiple": 2.0,
            "min_derived_days": 5,
        },
    )
    assert non_mr_out["reject_reason"] == "lambda_non_negative"

    invalid_domain = pd.DataFrame(
        {"y": 10.0 * np.power(-0.2, np.arange(8, dtype=float)), "x": np.zeros(8)}
    )
    invalid_out = alpha.evaluate_pair_cointegration(
        invalid_domain,
        min_obs=5,
        half_life_cfg={
            "min_days": 5,
            "max_days": 60,
            "max_hold_multiple": 2.0,
            "min_derived_days": 5,
        },
    )
    assert invalid_out["reject_reason"] == "lambda_invalid_domain"
