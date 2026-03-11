from __future__ import annotations

import pandas as pd


from processing.processing_primitives import process_and_fill_prices


def test_process_and_fill_prices_empty_input():
    out, removed, diag = process_and_fill_prices(pd.DataFrame())
    assert out.empty and removed == [] and diag == {}
