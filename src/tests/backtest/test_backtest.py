from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backtest.config import (
    BOConfig,
    BacktestConfig,
    Config,
    CostsConfig,
    DataConfig,
    OutputConfig,
    PairSelectionConfig,
    RiskConfig,
    StrategyConfig,
    WalkforwardConfig,
)
from backtest.data import load_sectors
from backtest.engine import BacktestResult, _summary, run_engine
from backtest.pair_selection import _bh_qvalues, select_pairs
from backtest.run import run_config
from backtest.walkforward import Window, make_windows
from backtest.wf_debug import build_wf_debug


def test_walkforward_rolling_and_expanding_do_not_overlap_tests() -> None:
    idx = pd.date_range("2020-01-01", periods=900, freq="D")
    windows = make_windows(idx, BacktestConfig())
    assert all(
        windows[i].test_end < windows[i + 1].test_start for i in range(len(windows) - 1)
    )
    expanding = make_windows(
        idx, BacktestConfig(walkforward=WalkforwardConfig(train_mode="expanding"))
    )
    assert expanding[0].train_start == expanding[1].train_start


def test_load_sectors_normalizes_symbols_and_drops_missing_sector(tmp_path) -> None:
    path = tmp_path / "screener.csv"
    pd.DataFrame(
        {
            "Symbol": ["brk.b", "a/b", "MISS", "EMPTY"],
            "Sector": ["Finance", "Technology", None, " "],
        }
    ).to_csv(path, index=False)

    sectors = load_sectors(path)

    assert sectors.to_dict() == {"BRK-B": "Finance", "A-B": "Technology"}


def test_pair_selection_uses_train_window_filters_and_ranking() -> None:
    prices = _cointegrated_prices()
    pairs, metrics = select_pairs(
        prices,
        Window(
            0, prices.index[0], prices.index[199], prices.index[200], prices.index[-1]
        ),
        PairSelectionConfig(
            min_corr=0.50,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=60.0,
            max_hurst=0.90,
            max_pairs=1,
        ),
    )
    assert len(pairs) == 1
    assert {"pair", "corr", "eg_pvalue", "alpha", "beta", "half_life", "hurst"}.issubset(metrics)
    assert metrics["eg_pvalue"].iloc[0] <= 0.20
    meta = next(iter(pairs.values()))
    assert meta.y in prices
    assert meta.x in prices
    assert np.isfinite(meta.alpha)
    assert np.isfinite(meta.beta) and meta.beta > 0
    assert np.isfinite(meta.half_life)


def test_pair_selection_filters_correlations_above_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, -0.015, 0.02])
    mid = np.array([0.015, 0.005, -0.01, 0.025, 0.0, 0.005, -0.02, 0.015])
    prices = pd.DataFrame(
        {
            "AAA": _prices_from_returns(base),
            "BBB": _prices_from_returns(base),
            "CCC": _prices_from_returns(mid),
        },
        index=pd.date_range("2024-01-01", periods=len(base) + 1, freq="D"),
    )
    calls: list[tuple[str, str, float]] = []

    def fake_test_pair(
        y: pd.Series, x: pd.Series, y_name: str, x_name: str, corr: float, cfg
    ) -> dict:
        del y, x, cfg
        calls.append((y_name, x_name, corr))
        return {
            "pair": f"{y_name}-{x_name}",
            "y": y_name,
            "x": x_name,
            "corr": corr,
            "eg_pvalue": 0.01,
            "alpha": 0.0,
            "beta": 1.0,
            "half_life": 2.0,
            "hurst": 0.2,
        }

    monkeypatch.setattr("backtest.pair_selection._test_pair", fake_test_pair)
    pairs, metrics = select_pairs(
        prices,
        Window(0, prices.index[0], prices.index[-1], prices.index[-1], prices.index[-1]),
        PairSelectionConfig(
            min_corr=0.50,
            max_corr=0.95,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=60.0,
            max_hurst=0.90,
            max_pairs=10,
        ),
    )

    assert ("AAA", "BBB") not in [(a, b) for a, b, _ in calls]
    assert set(pairs) == {"AAA-CCC", "BBB-CCC"}
    assert metrics["corr"].le(0.95).all()


def test_pair_selection_only_tests_pairs_within_same_sector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = pd.DataFrame(
        {
            "AAA": np.arange(100.0, 112.0),
            "BBB": np.arange(100.0, 112.0) + 1.0,
            "CCC": np.arange(100.0, 112.0) + 2.0,
        },
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    sectors = pd.Series(
        {"AAA": "Technology", "BBB": "Technology", "CCC": "Finance"},
        name="sector",
    )
    calls: list[tuple[str, str]] = []

    def fake_test_pair(
        y: pd.Series, x: pd.Series, y_name: str, x_name: str, corr: float, cfg
    ) -> dict:
        del y, x, corr, cfg
        calls.append((y_name, x_name))
        return {
            "pair": f"{y_name}-{x_name}",
            "y": y_name,
            "x": x_name,
            "corr": 1.0,
            "eg_pvalue": 0.01,
            "alpha": 0.0,
            "beta": 1.0,
            "half_life": 2.0,
            "hurst": 0.2,
        }

    monkeypatch.setattr("backtest.pair_selection._test_pair", fake_test_pair)
    pairs, metrics = select_pairs(
        prices,
        Window(0, prices.index[0], prices.index[-1], prices.index[-1], prices.index[-1]),
        PairSelectionConfig(
            min_corr=-1.0,
            max_corr=1.0,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=60.0,
        ),
        sectors=sectors,
    )

    assert calls == [("AAA", "BBB")]
    assert set(pairs) == {"AAA-BBB"}
    assert metrics["sector"].tolist() == ["Technology"]


def test_pair_selection_can_filter_by_bh_fdr(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = pd.DataFrame(
        {
            "AAA": np.arange(100.0, 110.0),
            "BBB": np.arange(100.0, 110.0) + 1.0,
            "CCC": np.arange(100.0, 110.0) + 2.0,
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    pvalues = {
        ("AAA", "BBB"): 0.001,
        ("AAA", "CCC"): 0.040,
        ("BBB", "CCC"): 0.050,
    }

    def fake_test_pair(
        y: pd.Series, x: pd.Series, y_name: str, x_name: str, corr: float, cfg
    ) -> dict:
        del y, x, cfg
        return {
            "pair": f"{y_name}-{x_name}",
            "y": y_name,
            "x": x_name,
            "corr": corr,
            "eg_pvalue": pvalues[(y_name, x_name)],
            "alpha": 0.0,
            "beta": 1.0,
            "half_life": 2.0,
            "hurst": 0.2,
        }

    monkeypatch.setattr("backtest.pair_selection._test_pair", fake_test_pair)
    pairs, metrics = select_pairs(
        prices,
        Window(0, prices.index[0], prices.index[-1], prices.index[-1], prices.index[-1]),
        PairSelectionConfig(
            min_corr=-1.0,
            max_corr=1.0,
            max_eg_pvalue=0.20,
            fdr_enabled=True,
            fdr_alpha=0.01,
            min_half_life=1.0,
            max_half_life=60.0,
            max_hurst=0.90,
            max_pairs=10,
        ),
    )

    assert set(pairs) == {"AAA-BBB"}
    assert metrics["eg_qvalue"].iloc[0] == pytest.approx(0.003)


def test_bh_qvalues_are_monotonic_after_sorting() -> None:
    q = _bh_qvalues(pd.Series([0.04, 0.001, 0.05]))

    assert q.tolist() == pytest.approx([0.05, 0.003, 0.05])


def test_run_requires_contiguous_test_windows(tmp_path) -> None:
    prices = _cointegrated_prices(periods=160)
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index_label="date")
    cfg = Config(
        data=DataConfig(prices_path=prices_path),
        backtest=BacktestConfig(
            walkforward=WalkforwardConfig(test_months=1, step_months=2)
        ),
        output=OutputConfig(dir=tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="step_months"):
        run_config(cfg)


def test_engine_summary_uses_closed_trade_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {"AAA": [100.0, 110.0, 120.0, 110.0, 100.0], "BBB": [100.0] * 5},
        index=idx,
    )
    positions = pd.DataFrame({"AAA-BBB": [1, 1, 0, -1, 0]}, index=idx)
    zscores = pd.DataFrame({"AAA-BBB": [-2.0, -1.0, 0.0, 2.0, 0.0]}, index=idx)
    betas = pd.DataFrame({"AAA-BBB": [0.0] * len(idx)}, index=idx)

    result = run_engine(
        prices,
        {"AAA-BBB": ("AAA", "BBB")},
        betas,
        positions,
        zscores,
        initial_capital=100_000.0,
        costs=CostsConfig(cost_bps=0.0),
        risk=RiskConfig(max_pair_weight=0.5),
    )

    assert len(result.trades) == 2
    assert result.summary["n_trades"] == 2
    assert result.summary["trade_winrate"] == 1.0
    assert result.summary["avg_holding_days"] == 1.5
    assert result.summary["avg_exposure"] == pytest.approx(0.3)
    assert "trades" not in result.summary
    assert "winrate" not in result.summary


def test_engine_summary_profit_factor_uses_gross_trade_returns() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    daily = pd.DataFrame(
        {"equity": [100_000.0, 101_000.0], "drawdown": [0.0, 0.0]},
        index=idx,
    )
    weights = pd.DataFrame(index=idx)
    trades = pd.DataFrame(
        {
            "return": [0.04, 0.01, -0.02],
            "holding_days": [1, 1, 1],
        }
    )

    summary = _summary(daily, weights, trades)

    assert summary["profit_factor"] == pytest.approx((0.04 + 0.01) / 0.02)


def test_wf_debug_uses_window_attribution_and_trade_quality(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.02, 0.0, 0.03, -0.01], index=idx)
    daily = pd.DataFrame({"return": returns, "turnover": [0.1, 0.2, 0.0, 0.3, 0.1]})
    daily["equity"] = 100_000.0 * (1.0 + daily["return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    positions = pd.DataFrame(
        {
            "AAA-BBB": [1, 0, 0, 1, 0],
            "CCC-DDD": [0, 1, 0, 0, 0],
        },
        index=idx,
    )
    weights = positions.astype(float) * 0.5
    trades = pd.DataFrame(
        {
            "pair": ["AAA-BBB", "CCC-DDD", "EEE-FFF", "AAA-BBB"],
            "entry_date": [idx[0], idx[1], idx[1], idx[3]],
            "exit_date": [idx[1], idx[2], idx[2], idx[4]],
            "return": [0.04, -0.01, -0.03, -0.02],
            "holding_days": [1, 1, 1, 2],
            "exit_reason": ["normal", "stop", "timeout", "forced_window_end"],
        }
    )
    windows = pd.DataFrame(
        [
            {
                "window": 0,
                "train_start": "2023-12-01",
                "train_end": "2023-12-31",
                "test_start": "2024-01-01",
                "test_end": "2024-01-03",
                "train_days": 21,
                "test_days": 3,
                "assets_before_filter": 12,
                "assets_missing_sector": 2,
                "assets_after_sector": 10,
                "eligible_sector_groups": 3,
                "n_pairs": 2,
                "max_pairs": 2,
                "pairs": "AAA-BBB;CCC-DDD",
                "entry_z": 1.2,
                "exit_z": 0.1,
                "stop_z": 3.0,
            },
            {
                "window": 1,
                "train_start": "2024-01-01",
                "train_end": "2024-01-03",
                "test_start": "2024-01-04",
                "test_end": "2024-01-05",
                "train_days": 3,
                "test_days": 2,
                "assets_before_filter": 12,
                "assets_missing_sector": 2,
                "assets_after_sector": 10,
                "eligible_sector_groups": 3,
                "n_pairs": 1,
                "max_pairs": 2,
                "pairs": "AAA-BBB",
                "entry_z": 1.4,
                "exit_z": 0.0,
                "stop_z": 2.5,
            },
        ]
    )
    selected_pairs = pd.DataFrame(
        {
            "window": [0, 0, 1],
            "pair": ["AAA-BBB", "CCC-DDD", "AAA-BBB"],
            "corr": [0.8, 0.9, 0.85],
            "eg_pvalue": [0.01, 0.02, 0.03],
            "half_life": [4.0, 6.0, 5.0],
            "hurst": [np.nan, 0.3, 0.2],
        }
    )
    bo_trials = pd.DataFrame({"window": [0, 0], "score": [1.0, 2.0]})
    cfg = Config(
        data=DataConfig(prices_path=tmp_path / "prices.csv"),
        bo=BOConfig(enabled=True),
        costs=CostsConfig(cost_bps=2.0),
    )

    debug = build_wf_debug(
        BacktestResult(daily, positions, weights, trades, {}),
        windows,
        selected_pairs,
        bo_trials,
        [{"window": 0, "score": np.nan, "params": {"entry_z": 1.2}}],
        cfg,
    )

    json.dumps(debug, allow_nan=False)
    first = debug[0]
    assert first["selection"]["max_pairs_hit"] is True
    assert first["optimization"]["method"] == "bo"
    assert first["optimization"]["trials"] == 2
    assert first["optimization"]["best_score"] is None
    assert first["signals"]["entry_count"] == 2
    assert first["signals"]["exit_count"] == 2
    assert first["performance"]["window_return"] == pytest.approx(1.01 * 0.98 - 1.0)

    trade_quality = first["trade_quality"]
    assert trade_quality["closed_trade_count"] == 3
    assert trade_quality["gross_profit"] == pytest.approx(0.04)
    assert trade_quality["gross_loss"] == pytest.approx(-0.04)
    assert trade_quality["profit_factor"] == pytest.approx(1.0)
    assert trade_quality["largest_win"] == pytest.approx(0.04)
    assert trade_quality["largest_loss"] == pytest.approx(-0.03)
    assert trade_quality["worst_3_losses_share_of_gross_loss"] == pytest.approx(1.0)
    assert trade_quality["normal_exit_count"] == 1
    assert trade_quality["stop_exit_count"] == 1
    assert trade_quality["timeout_exit_count"] == 1
    assert trade_quality["forced_window_end_exit_count"] == 0

    second = debug[1]
    assert second["trade_quality"]["forced_window_end_exit_count"] == 1


def test_smoke_cli_writes_core_outputs(tmp_path) -> None:
    prices = _cointegrated_prices(periods=260)
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index_label="date")

    cfg = Config(
        data=DataConfig(prices_path=prices_path),
        pair_selection=PairSelectionConfig(
            min_corr=0.50,
            max_eg_pvalue=0.20,
            min_half_life=1.0,
            max_half_life=80.0,
            max_hurst=0.90,
            max_pairs=2,
        ),
        backtest=BacktestConfig(
            initial_capital=100_000,
            walkforward=WalkforwardConfig(
                enabled=True,
                train_mode="rolling",
                train_months=3,
                test_months=1,
                step_months=1,
            ),
        ),
        strategy=StrategyConfig(z_window_multiplier=1.0),
        risk=RiskConfig(max_pair_weight=0.5),
        costs=CostsConfig(cost_bps=0.0),
        output=OutputConfig(dir=tmp_path / "out"),
    )
    result = run_config(cfg)
    assert not result.daily.empty
    for name in (
        "summary.json",
        "daily.csv",
        "trades.csv",
        "positions.csv",
        "weights.csv",
        "windows.csv",
        "selected_pairs.csv",
        "wf_debug.json",
    ):
        assert (tmp_path / "out" / name).exists()
    windows = pd.read_csv(tmp_path / "out" / "windows.csv")
    assert (windows["n_pairs"] > 0).all()
    assert windows["pairs"].str.len().gt(0).all()
    assert not pd.read_csv(tmp_path / "out" / "selected_pairs.csv").empty
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    wf_debug = json.loads((tmp_path / "out" / "wf_debug.json").read_text(encoding="utf-8"))
    assert len(wf_debug) == len(windows)
    for section in (
        "window",
        "selection",
        "optimization",
        "signals",
        "performance",
        "trade_quality",
    ):
        assert section in wf_debug[0]


def _cointegrated_prices(periods: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=periods, freq="D")
    x_log = np.log(50) + np.cumsum(rng.normal(0, 0.01, periods))
    spread = np.zeros(periods)
    for i in range(1, periods):
        spread[i] = 0.75 * spread[i - 1] + rng.normal(0, 0.002)
    y_log = x_log + spread
    z_log = np.log(70) + np.cumsum(rng.normal(0, 0.02, periods))
    return pd.DataFrame(
        {"AAA": np.exp(y_log), "BBB": np.exp(x_log), "CCC": np.exp(z_log)},
        index=idx,
    )


def _prices_from_returns(returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
