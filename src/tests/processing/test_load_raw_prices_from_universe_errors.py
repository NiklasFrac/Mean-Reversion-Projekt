from __future__ import annotations

from pathlib import Path

import pytest


from processing.raw_loader import load_raw_prices_from_universe


def test_load_raw_prices_from_universe_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_raw_prices_from_universe(tmp_path)
