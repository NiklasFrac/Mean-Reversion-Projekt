import pandas as pd

from backtest.loader import load_adv_map


def test_load_adv_map_accepts_semicolon_delimited_csv(tmp_path) -> None:
    p = tmp_path / "adv.csv"
    p.write_text("ticker;dollar_adv_hist\nAAA;1000\nBBB;2000\n", encoding="utf-8")

    out = load_adv_map(p)
    assert out["AAA"] == 1000.0
    assert out["BBB"] == 2000.0


def test_load_adv_map_accepts_pickle_dict_symbol_to_float(tmp_path) -> None:
    p = tmp_path / "adv.pkl"
    pd.to_pickle({"AAA": 123.0, "BBB": 456.0}, p)
    out = load_adv_map(p)
    assert out == {"AAA": 123.0, "BBB": 456.0}


def test_load_adv_map_accepts_pickle_dict_symbol_to_meta(tmp_path) -> None:
    p = tmp_path / "adv.pkl"
    pd.to_pickle(
        {
            "AAA": {"adv": 123.0, "last_price": 10.0},
            "BBB": {"adv": 456.0, "last_price": 20.0},
        },
        p,
    )
    out = load_adv_map(p)
    assert out == {"AAA": 123.0, "BBB": 456.0}
