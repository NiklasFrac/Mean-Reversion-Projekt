from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from backtest.run.train_only import _read_train_only_equity


def test_read_train_only_equity_handles_mixed_tz_without_warning(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train_only"
    train_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "date": [
                "2024-01-01T00:00:00+00:00",
                "2024-01-01T16:00:00-05:00",
                "2024-01-03",
            ],
            "equity": [100.0, 101.0, 102.0],
        }
    )
    df.to_csv(train_dir / "equity_curve_train.csv", index=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = _read_train_only_equity(train_dir)

    assert out is not None and not out.empty
    assert not rec
