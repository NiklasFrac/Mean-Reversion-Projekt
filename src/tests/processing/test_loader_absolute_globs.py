from __future__ import annotations

from pathlib import Path

import pandas as pd

from processing.raw_loader import load_raw_prices_from_universe


def test_load_raw_prices_from_universe_supports_absolute_globs(tmp_path: Path):
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    prices = pd.DataFrame({"AAA": [1.0, 2.0, 3.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0, 12.0]}, index=idx)

    p_prices = (tmp_path / "raw_prices.pkl").resolve()
    p_volume = (tmp_path / "raw_volume.pkl").resolve()
    prices.to_pickle(p_prices)
    volume.to_pickle(p_volume)

    prices_out, volume_out, used = load_raw_prices_from_universe(
        tmp_path,
        price_globs=[str(p_prices)],
        volume_globs=[str(p_volume)],
    )

    pd.testing.assert_frame_equal(prices_out, prices)
    assert volume_out is not None
    pd.testing.assert_frame_equal(volume_out, volume)
    assert Path(str(used["prices"])).resolve() == p_prices
    assert Path(str(used["volume"])).resolve() == p_volume
