import pandas as pd

from processing.pipeline import _reconcile_ohlc


def test_reconcile_clamps_close_and_open():
    idx = pd.date_range("2023-01-01", periods=2, tz="America/New_York")
    cols = pd.MultiIndex.from_tuples(
        [
            ("AAA", "close"),
            ("AAA", "high"),
            ("AAA", "low"),
            ("AAA", "open"),
            ("AAA", "volume"),
            ("BBB", "close"),
            ("BBB", "high"),
            ("BBB", "low"),
        ],
        names=["symbol", "field"],
    )
    data = [
        [
            110.0,
            100.0,
            90.0,
            150.0,
            10.0,
            5.0,
            7.0,
            9.0,
        ],  # AAA: close>high, open>high; BBB: close<low
        [95.0, 105.0, 97.0, 80.0, 20.0, 10.0, 11.0, 9.0],  # AAA: open<low
    ]
    df = pd.DataFrame(data, index=idx, columns=cols)

    out = _reconcile_ohlc(df)

    close_aaa = out.xs(("AAA", "close"), axis=1)
    high_aaa = out.xs(("AAA", "high"), axis=1)
    low_aaa = out.xs(("AAA", "low"), axis=1)
    open_aaa = out.xs(("AAA", "open"), axis=1)

    # AAA: high should be raised to close, low unchanged, open clipped into band
    assert float(high_aaa.iloc[0]) == float(close_aaa.iloc[0])
    assert float(low_aaa.iloc[0]) == 90.0
    assert float(open_aaa.iloc[0]) == float(close_aaa.iloc[0])  # clipped to new high
    assert float(open_aaa.iloc[1]) == float(low_aaa.iloc[1])  # clipped up to low

    # BBB: close below low -> low lowered to close
    close_bbb = out.xs(("BBB", "close"), axis=1)
    low_bbb = out.xs(("BBB", "low"), axis=1)
    high_bbb = out.xs(("BBB", "high"), axis=1)
    assert float(low_bbb.iloc[0]) == float(close_bbb.iloc[0])
    assert float(high_bbb.iloc[0]) == 7.0  # unchanged

    # Volume untouched
    vol_aaa = out.xs(("AAA", "volume"), axis=1)
    assert float(vol_aaa.iloc[0]) == 10.0


def test_reconcile_repairs_crossed_bounds_when_close_missing():
    idx = pd.date_range("2023-01-01", periods=1, tz="America/New_York")
    cols = pd.MultiIndex.from_tuples(
        [
            ("AAA", "close"),
            ("AAA", "high"),
            ("AAA", "low"),
            ("BBB", "close"),
            ("BBB", "high"),
            ("BBB", "low"),
        ],
        names=["symbol", "field"],
    )
    # AAA has high<low and close missing. BBB is present to ensure multi-symbol path.
    df = pd.DataFrame(
        [[float("nan"), 10.0, 20.0, 1000.0, 1100.0, 900.0]], index=idx, columns=cols
    )

    out = _reconcile_ohlc(df)

    high_aaa = out.xs(("AAA", "high"), axis=1)
    low_aaa = out.xs(("AAA", "low"), axis=1)

    assert float(high_aaa.iloc[0]) == 20.0
    assert float(low_aaa.iloc[0]) == 10.0
