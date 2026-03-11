from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _compute_block_boundaries(n_samples: int, n_blocks: int) -> np.ndarray:
    base = n_samples // n_blocks
    rem = n_samples % n_blocks
    sizes = np.full(n_blocks, base, dtype=np.int64)
    if rem:
        sizes[:rem] += 1
    boundaries = np.empty(n_blocks + 1, dtype=np.int64)
    boundaries[0] = 0
    np.cumsum(sizes, out=boundaries[1:])
    return boundaries


def _embargo_len(embargo: float | int, block_len: int) -> int:
    if embargo <= 0:
        return 0
    if isinstance(embargo, float) and float(embargo).is_integer():
        return int(embargo)
    if isinstance(embargo, float):
        return int(math.ceil(float(embargo) * float(block_len)))
    return int(embargo)


def _train_indices_with_purge_embargo(
    boundaries: np.ndarray,
    test_blocks: Iterable[int],
    purge: int,
    embargo: float | int,
) -> np.ndarray:
    n_blocks = len(boundaries) - 1
    test_set = set(int(b) for b in test_blocks)
    parts: list[np.ndarray] = []

    for b in range(n_blocks):
        if b in test_set:
            continue

        left = int(boundaries[b])
        right = int(boundaries[b + 1])
        blen = right - left
        if blen <= 0:
            continue

        emb = _embargo_len(embargo, blen)
        if (b - 1) in test_set:
            left = max(left, int(boundaries[b]) + int(purge) + int(emb))
        if (b + 1) in test_set:
            right = min(right, int(boundaries[b + 1]) - int(purge) - int(emb))

        if right > left:
            parts.append(np.arange(left, right, dtype=np.int64))

    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts, dtype=np.int64)


def _trim_range_indices(
    left: int,
    right: int,
    *,
    purge: int,
    embargo: float | int,
) -> np.ndarray:
    blen = right - left
    if blen <= 0:
        return np.empty(0, dtype=np.int64)
    emb = _embargo_len(embargo, blen)
    l2 = min(right, left + int(max(0, purge)))
    r2 = max(l2, right - int(max(0, purge)) - int(max(0, emb)))
    if r2 <= l2:
        return np.empty(0, dtype=np.int64)
    return np.arange(l2, r2, dtype=np.int64)


def _trim_block_indices(
    boundaries: np.ndarray,
    block_id: int,
    *,
    purge: int,
    embargo: float | int,
) -> np.ndarray:
    left = int(boundaries[block_id])
    right = int(boundaries[block_id + 1])
    return _trim_range_indices(left, right, purge=purge, embargo=embargo)
