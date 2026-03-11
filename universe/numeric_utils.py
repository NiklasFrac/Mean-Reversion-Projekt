from __future__ import annotations

from typing import TypeVar

import pandas as pd

T = TypeVar("T", pd.Series, pd.DataFrame)


def replace_inf_with_nan(obj: T) -> T:
    return obj.replace([float("inf"), float("-inf")], float("nan"))
