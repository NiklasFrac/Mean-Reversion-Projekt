import numpy as np
import pandas as pd
import pytest

from backtest.strat import baseline
from backtest.utils import strategy as strat_helpers


def test_get_tickers_from_meta_dict_paths() -> None:
    assert baseline._get_tickers_from_meta({"meta": {"t1": "AAA", "t2": "BBB"}}) == (
        "AAA",
        "BBB",
    )
    assert baseline._get_tickers_from_meta({"pair": "CCC/DDD"}) == ("CCC", "DDD")


def test_get_tickers_from_meta_non_dict() -> None:
    assert baseline._get_tickers_from_meta("not-a-dict") is None


def test_estimate_beta_short_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    s_short = pd.Series([1.0])
    assert baseline._estimate_beta_ols_with_intercept(s_short, s_short) == 1.0

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(baseline, "OLS", _boom)
    s = pd.Series([1.0, 2.0, 3.0])
    assert baseline._estimate_beta_ols_with_intercept(s, s) == 1.0


def test_estimate_beta_details_accepts_only_positive_values() -> None:
    x = pd.Series([1.0, 2.0, 3.0])
    y_pos = pd.Series([2.0, 4.0, 6.0])
    beta_pos, reason_pos = strat_helpers.estimate_beta_ols_with_intercept_details(
        y_pos, x
    )
    assert beta_pos == pytest.approx(2.0)
    assert reason_pos is None

    y_neg = pd.Series([3.0, 2.0, 1.0])
    beta_neg, reason_neg = strat_helpers.estimate_beta_ols_with_intercept_details(
        y_neg, x
    )
    assert beta_neg is None
    assert reason_neg == "beta_non_positive"


def test_baseline_positions_exit_branches() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-2.5, 2.5, 0.0, np.nan], index=idx)
    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[2],
        allow_exit_after_end=False,
    )
    assert pos.iloc[0] == 1
    assert pos.iloc[1] == 0


def test_baseline_positions_max_hold() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    z = pd.Series([-2.5, -1.0, -1.0], index=idx)
    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.1,
        stop_z=3.0,
        max_hold_days=1,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
    )
    assert pos.iloc[0] == 1
    assert pos.iloc[1] == 0


def test_baseline_positions_reenter_after_neutral_exit_without_cooldown() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    z = pd.Series([-2.5, 0.0, -2.5], index=idx)
    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
    )
    assert pos.tolist() == [1, 0, 1]


def test_baseline_positions_do_not_reenter_without_fresh_cross_after_stop_exit() -> (
    None
):
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-2.5, -1.9, -3.0, -2.5], index=idx)
    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.5,
        stop_z=3.0,
        max_hold_days=5,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[-1],
        allow_exit_after_end=False,
    )
    assert pos.tolist() == [1, 1, 0, 0]


def test_baseline_positions_allow_exit_after_end() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    z = pd.Series([-2.5, -2.5, -2.5, 0.0], index=idx)
    pos = baseline._positions_from_z(
        z,
        entry_z=2.0,
        exit_z=0.1,
        stop_z=3.0,
        max_hold_days=10,
        cooldown_days=0,
        test_start=idx[0],
        entry_end=idx[1],
        allow_exit_after_end=True,
    )
    assert pos.loc[idx[2]] == 1
    assert pos.loc[idx[3]] == 0


def test_frozen_zscore_uses_train_stats() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    spread = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0], index=idx)
    z, ok = baseline._frozen_zscore(spread, train_index=idx[:3])
    assert ok is True
    assert z.iloc[0] == pytest.approx(-1.224744871, rel=1e-6)
    assert z.iloc[3] == pytest.approx(9.797958971, rel=1e-6)


def test_frozen_zscore_stats_exposes_same_sigma_t() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    spread = pd.Series([1.0, 2.0, 3.0, 10.0, 11.0], index=idx)
    z, mean, sigma, ok = strat_helpers.frozen_zscore_stats(spread, train_index=idx[:3])

    assert ok is True
    assert z.iloc[3] == pytest.approx(9.797958971, rel=1e-6)
    assert mean.iloc[3] == pytest.approx(2.0, rel=1e-6)
    assert sigma.iloc[3] == pytest.approx(np.sqrt(2.0 / 3.0), rel=1e-6)


def test_rolling_zscore_on_allowed_dates_excludes_gap_days() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    train_idx = idx[:3].append(idx[6:8])
    eval_idx = idx[8:10]
    allowed = (
        strat_helpers.prior_train_history(train_idx, eval_index=eval_idx)
        .union(eval_idx)
        .sort_values()
    )

    spread_base = pd.Series(
        [1.0, 2.0, 3.0, 50.0, 60.0, 70.0, 4.0, 5.0, 6.0, 7.0], index=idx
    )
    spread_gap_spike = spread_base.copy()
    spread_gap_spike.loc[idx[3:6]] = [500.0, 600.0, 700.0]

    z_base = strat_helpers.rolling_zscore_on_allowed_dates(
        spread_base,
        allowed_index=allowed,
        window=3,
        min_periods=2,
        full_index=idx,
    )
    z_spike = strat_helpers.rolling_zscore_on_allowed_dates(
        spread_gap_spike,
        allowed_index=allowed,
        window=3,
        min_periods=2,
        full_index=idx,
    )

    assert z_base.loc[eval_idx].tolist() == pytest.approx(
        z_spike.loc[eval_idx].tolist()
    )


def test_prior_train_history_excludes_future_train_blocks() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    train_idx = idx[:2].append(idx[6:8])
    eval_idx = idx[4:6]
    hist = strat_helpers.prior_train_history(train_idx, eval_index=eval_idx)
    assert list(hist) == list(idx[:2])
