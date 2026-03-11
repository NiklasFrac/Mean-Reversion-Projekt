from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from universe import fundamentals as funda
from universe import outputs


class _FakeTicker:
    def __init__(self, outer: "_FakeYF", ticker: str):
        self.outer = outer
        self.fast_info = outer._fast_info
        self._info = outer._info
        self.ticker = ticker

    def get_info(self) -> dict[str, Any]:
        return dict(self._info)

    @property
    def info(self) -> dict[str, Any]:
        return dict(self._info)

    def get_major_holders(self):
        return self.outer._holders


class _FakeYF:
    def __init__(
        self,
        fast_info: dict[str, Any] | None,
        info: dict[str, Any],
        holders=None,
        fallback_price=12.5,
        fallback_vol=1_100_000,
    ):
        self._fast_info = fast_info
        self._info = info
        self._holders = holders
        self._fallback_price = fallback_price
        self._fallback_vol = fallback_vol

    def Ticker(
        self, ticker: str
    ) -> _FakeTicker:  # pragma: no cover - exercised indirectly
        return _FakeTicker(self, ticker)

    def download(self, sym: str, *_, **__):  # fallback path
        return pd.DataFrame(
            {
                "Adj Close": [self._fallback_price],
                "Volume": [self._fallback_vol],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )


def test_fetch_fundamentals_one_prefers_fast_info(monkeypatch):
    holders = pd.DataFrame(
        {"v": [55.0, 44.0], "label": ["% Held by Insiders", "% Held by Institutions"]}
    )
    fake = _FakeYF(
        fast_info={
            "lastPrice": 10.0,
            "marketCap": 1_000_000_000,
            "tenDayAverageVolume": 1_200_000,
            "sharesOutstanding": 100_000_000,
            "quoteType": "ETF",
            "currency": "USD",
            "exchange": "NYSE",
            "market": "us",
        },
        info={
            "longName": "Sample Fund",
            "sector": "Tech",
            "industry": "Software",
            "country": "US",
            "dividendYield": 0.02,
            "heldPercentInstitutions": 0.4,
            "quoteType": "ETF",
        },
        holders=holders,
    )
    monkeypatch.setattr(funda, "yf", fake)

    record = funda._fetch_fundamentals_one("AAA")

    assert record.price == 10.0
    assert record.market_cap == 1_000_000_000
    assert record.volume == 1_200_000
    assert record.float_pct is not None
    assert record.is_etf is True
    assert record.dividend is True
    assert record.country == "US"


def test_fetch_fundamentals_one_uses_fallback_when_missing(monkeypatch):
    fake = _FakeYF(
        fast_info=None, info={}, holders=None, fallback_price=9.5, fallback_vol=500_000
    )
    monkeypatch.setattr(funda, "yf", fake)

    rec = funda._fetch_fundamentals_one("BBB")

    assert rec.price == pytest.approx(9.5)
    assert rec.volume == pytest.approx(500_000)
    assert rec.market_cap is None  # not derivable without shares_out


def test_circuit_breaker_cycles_state():
    cb = funda.CircuitBreaker(max_consec_fail=2)
    assert cb.open() is False
    cb.err()
    assert cb.open() is False
    cb.err()
    assert cb.open() is True
    cb.ok()
    assert cb.open() is False


def test_write_universe_ext_csv_emits_expected_columns(tmp_path):
    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [10.0],
            "market_cap": [1_000_000_000],
            "volume": [1_000_000],
            "float_pct": [0.5],
            "free_float_shares": [500_000_000],
            "free_float_mcap": [500_000_000],
            "dividend": [True],
            "is_etf": [False],
            "sector": ["Tech"],
            "industry": ["Software"],
            "country": ["US"],
            "shares_out": [1_000_000_000],
        },
        index=idx,
    )
    df_funda = pd.DataFrame(
        {
            "long_name": ["Acme Corp"],
            "country": ["US"],
            "currency": ["USD"],
            "exchange_code": ["XNYS"],
            "market": ["us"],
            "quote_type": ["EQUITY"],
        },
        index=idx,
    )
    out_csv = tmp_path / "ext.csv"

    outputs.write_universe_ext_csv(df_uni, df_funda, out_csv, adv_window=42)

    result = pd.read_csv(out_csv)
    expected_cols = {
        "ticker",
        "stable_id",
        "currency",
        "issuer_id",
        "price",
        "market_cap",
        "volume",
        "float_pct",
        "free_float_shares",
        "free_float_mcap",
        "dividend",
        "is_etf",
        "sector",
        "industry",
        "country",
        "shares_out",
        "exchange_code",
        "market",
        "quote_type",
        "adv_window_used",
    }
    assert expected_cols.issubset(set(result.columns))
    assert result.loc[0, "issuer_id"].startswith("acme")
    assert result.loc[0, "adv_window_used"] == 42


def test_write_universe_ext_csv_uses_price_eff_for_screened_price(tmp_path):
    idx = pd.Index(["AAA"], name="ticker")
    df_uni = pd.DataFrame(
        {
            "price": [2.0],
            "price_eff": [10.0],
            "price_warmup_med": [10.0],
            "market_cap": [1_000_000_000],
            "volume": [1_000_000],
            "float_pct": [0.5],
            "dollar_adv": [2_000_000.0],
            "dollar_adv_eff": [10_000_000.0],
            "dollar_adv_hist": [10_000_000.0],
            "dividend": [True],
            "is_etf": [False],
            "shares_out": [1_000_000_000],
        },
        index=idx,
    )
    df_funda = pd.DataFrame(index=idx)
    out_csv = tmp_path / "ext_basis.csv"

    outputs.write_universe_ext_csv(df_uni, df_funda, out_csv, adv_window=30)

    result = pd.read_csv(out_csv)
    assert float(result.loc[0, "price"]) == pytest.approx(10.0)
    assert float(result.loc[0, "price_snapshot"]) == pytest.approx(2.0)
    assert result.loc[0, "price_filter_basis"] == "warmup_median"
    assert float(result.loc[0, "dollar_adv_filter_value"]) == pytest.approx(
        10_000_000.0
    )
    assert float(result.loc[0, "dollar_adv_snapshot"]) == pytest.approx(2_000_000.0)
    assert result.loc[0, "dollar_adv_filter_basis"] == "warmup_hist"
