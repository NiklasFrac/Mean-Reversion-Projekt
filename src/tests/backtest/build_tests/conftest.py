"""
Legacy "build_tests" suite.

This folder contains older integration/build-time checks that depended on an
earlier module layout (e.g. a top-level `bo.py`, `borrow_manager.py`, and a
`src/backtest/src/` package).

They are kept for reference, but are NOT part of the default unit test suite.
To opt in, run with:
  RUN_BUILD_TESTS=1 python -m pytest -q src/tests/backtest/build_tests
"""

import os
import sys
from pathlib import Path
from typing import Any


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:  # pytest hook
    # Avoid import-time failures from legacy tests unless explicitly enabled.
    if str(os.getenv("RUN_BUILD_TESTS", "")).strip() in {"1", "true", "yes", "on"}:
        return False
    p = Path(str(collection_path))
    return p.suffix == ".py" and p.name.startswith("test_")


HERE = Path(__file__).resolve()
CANDIDATES = [
    HERE.parents[1] / "src" / "backtest" / "src",  # <repo>/src/backtest/src
    HERE.parents[2] / "src" / "backtest" / "src",
    HERE.parents[3] / "src" / "backtest" / "src",
]
for p in CANDIDATES:
    if (p / "data_loader.py").exists():
        sys.path.insert(0, str(p))
        break
