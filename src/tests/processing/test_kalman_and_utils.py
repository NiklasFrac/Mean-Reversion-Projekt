from __future__ import annotations

import numpy as np


from processing.fill import kalman_smoother_level


def test_kalman_smoother_level_empty_and_all_nan():
    assert kalman_smoother_level(np.array([])).size == 0
    y = np.array([np.nan, np.nan, np.nan], dtype=float)
    out = kalman_smoother_level(y)
    assert np.isnan(out).all() and out.shape == y.shape


def test__to_float_ndarray_basic():
    arr = np.asarray([1, 2, 3], dtype=float)
    assert arr.dtype == float and arr.tolist() == [1.0, 2.0, 3.0]
