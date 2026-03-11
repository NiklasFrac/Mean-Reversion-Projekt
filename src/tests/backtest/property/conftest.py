"""
Legacy property-based tests (Hypothesis).

These tests target an older schema snapshot kept under the test tree. Keep them
opt-in so normal test runs do not pull in legacy-only dependencies.

Enable with:
  RUN_PROPERTY_TESTS=1 python -m pytest -q src/tests/backtest/property
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _opt_in_enabled() -> bool:
    return str(os.getenv("RUN_PROPERTY_TESTS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:  # pytest hook
    if _opt_in_enabled():
        return False
    p = Path(str(collection_path))
    return p.suffix == ".py" and p.name.startswith("test_")


if _opt_in_enabled():
    HERE = Path(__file__).resolve()

    # Minimal path injection for the legacy tests (only when opted in).
    for parent in [HERE] + list(HERE.parents):
        if (parent / "src").is_dir():
            REPO_ROOT = parent
            break
    else:
        REPO_ROOT = HERE.parents[3]

    SRC_ROOT = REPO_ROOT / "src"
    for p in (REPO_ROOT, SRC_ROOT):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


@pytest.fixture(autouse=True)
def _fix_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
