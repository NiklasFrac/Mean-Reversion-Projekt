from __future__ import annotations

import types

import pandas as pd

from universe import fundamentals as uf


def test_fetch_fundamentals_handles_slash_symbols(monkeypatch):
    calls: dict[str, list[str]] = {"ticker": [], "download": []}

    class FakeTicker:
        def __init__(self, sym: str) -> None:
            calls["ticker"].append(sym)
            self.fast_info = {
                # Force fallback path so we also exercise yf.download with normalized symbol.
                "lastPrice": None,
                "marketCap": None,
                "tenDayAverageVolume": None,
            }

        def get_info(self) -> dict[str, object]:
            return {"currency": "USD"}

    def fake_download(sym: str, **kwargs) -> pd.DataFrame:
        calls["download"].append(sym)
        idx = pd.date_range("2024-01-01", periods=3)
        return pd.DataFrame(
            {
                "Adj Close": [123, 124, 125],
                "Close": [123, 124, 125],
                "Volume": [1_000_000, 1_000_000, 1_000_000],
            },
            index=idx,
        )

    fake_yf = types.SimpleNamespace(Ticker=FakeTicker, download=fake_download)
    monkeypatch.setattr(uf, "yf", fake_yf)

    df, monitoring = uf.fetch_fundamentals_parallel(
        tickers=["BRK/A"],
        workers=1,
        show_progress=False,
        rate_limit_per_sec=100.0,
    )

    assert "BRK-A" in df.index
    assert df.loc["BRK-A", "price"] == 125.0
    assert calls["ticker"] == ["BRK-A"]
    assert calls["download"] == ["BRK-A"]
    assert monitoring["failed"] == []
