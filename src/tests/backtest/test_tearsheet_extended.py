from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from backtest.reporting import tearsheet


def test_write_tearsheet_with_tca_and_appendices(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    eq = pd.Series(
        100.0 + np.cumsum(np.linspace(1.0, -0.5, len(idx))), index=idx, name="equity"
    )
    stats_df = pd.DataFrame(
        {
            "date": idx,
            "equity": eq.values,
        }
    )
    out_dir = tmp_path / "tearsheet"
    tearsheet.write_tearsheet(eq, stats_df, out_dir, include_pdf=False, save_svg=False)
    assert (out_dir / "stats.csv").exists()
    assert (out_dir / "stats.json").exists()
    assert (out_dir / "equity_curve.png").exists()
    assert (out_dir / "drawdown_underwater.png").exists()


def test_tearsheet_validate_equity_and_returns_errors() -> None:
    with pytest.raises(TypeError):
        tearsheet._validate_equity("not-a-series")

    bad_idx = pd.Index(["bad-date", "also-bad"])
    bad_series = pd.Series([1.0, 2.0], index=bad_idx)
    with pytest.raises(TypeError):
        tearsheet._validate_equity(bad_series)

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    eq = pd.Series([100.0, 101.0, 102.0], index=idx)
    with pytest.raises(ValueError):
        tearsheet.compute_returns(eq, kind="bad")


def test_tearsheet_json_default_helpers() -> None:
    assert tearsheet._json_default(np.float64(1.25)) == 1.25
    ts = pd.Timestamp("2024-01-01")
    assert "2024-01-01" in tearsheet._json_default(ts)
    assert tearsheet._json_default(ts.date()) == "2024-01-01"
    assert tearsheet._json_default(float("nan")) is None
    assert "object" in tearsheet._json_default(object())
