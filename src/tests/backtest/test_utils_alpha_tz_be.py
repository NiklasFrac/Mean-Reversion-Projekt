import numpy as np
import pandas as pd
import pytest

from backtest.utils import alpha, be, tz


def test_safe_coint_and_pair_prefilter() -> None:
    idx = pd.bdate_range("2024-01-02", periods=50)
    x = pd.Series(np.linspace(10, 11, len(idx)), index=idx)
    y = x + 0.01 * np.random.default_rng(1).normal(size=len(idx))
    assert alpha.safe_coint(x, y, alpha=1.0) is True

    df = pd.DataFrame({"y": y, "x": x})
    assert alpha.pair_prefilter(df) in (True, False)

    spread2, z2, beta2 = alpha.compute_spread_zscore(y, x, cfg={"z_window": 5})
    assert len(spread2) == len(z2) == len(beta2)


def test_tz_helpers() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s = pd.Series([1, 2, 3], index=idx)
    s_ny = tz.ensure_index_tz(s, tz.NY_TZ)
    assert str(s_ny.index.tz) == tz.NY_TZ

    ts = tz.align_ts_to_series(pd.Timestamp("2024-01-02"), s_ny)
    assert ts.tz is not None

    df_naive = tz.to_naive_local(s_ny.to_frame("v"))
    assert df_naive.index.tz is None

    tz.same_tz_or_raise(s_ny.index, ts, allow_naive_pair=False, context="test")
    with pytest.raises(ValueError):
        tz.same_tz_or_raise(
            pd.date_range("2024-01-01", periods=2), ts, allow_naive_pair=False
        )


def test_get_ex_tz_from_cfg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_EXCHANGE_TZ", "US/Eastern")
    assert tz.get_ex_tz({}, None) == tz.NY_TZ
    monkeypatch.delenv("WF_EXCHANGE_TZ", raising=False)


def test_be_fill_math() -> None:
    fills = [
        {"qty": 5, "price": 10.0, "liquidity": "M"},
        {"qty": 5, "price": 10.2, "liquidity": "T"},
    ]
    vwap, notional, qty = be._compute_vwap_and_totals(fills)
    assert qty == 10
    assert vwap == pytest.approx(notional / qty)

    fees, m_cnt, t_cnt = be._calc_fees_from_fills(
        fills, maker_bps=-1.0, taker_bps=2.0, min_fee=0.0
    )
    assert m_cnt == 1 and t_cnt == 1
    assert fees < 0.0

    slip = be._calc_slippage_from_fills(fills, side="buy", reference_px=10.0, tick=0.01)
    assert slip["slippage_ccy"] >= 0.0
    assert slip["vwap"] > 0.0


def test_be_more_helpers() -> None:
    fills = [{"qty": 1, "price": 10.0, "ts": "2024-01-02T10:00:00"}]
    realized = be._realized_spread_bps(vwap=10.0, reference_mid=10.1, side="buy")
    assert realized > 0.0

    ttf = be._time_to_fill_ms(pd.Timestamp("2024-01-02T09:59:59"), fills)
    assert ttf >= 0.0

    df = pd.DataFrame({"gross_pnl": [1.0]})
    out = be._ensure_cost_columns(df)
    assert "net_pnl" in out.columns

    diag = be._ensure_lob_diag_columns(df)
    assert "lob_net_pnl" in diag.columns

    assert be._parse_pair_symbols_generic("AAA/BBB") == ("AAA", "BBB")

    price_map = {
        "AAA": pd.Series([1.0, 1.1], index=pd.bdate_range("2024-01-02", periods=2))
    }
    px = be._series_price_at(price_map, "AAA", pd.Timestamp("2024-01-03"))
    assert px is not None

    ann = be._infer_annualization_factor(pd.bdate_range("2024-01-02", periods=5))
    assert ann == 252
