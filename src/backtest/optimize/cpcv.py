# src/backtest/cpcv.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, islice
from math import comb
from typing import Iterable, Iterator, List, Sequence, Tuple, overload

import numpy as np
from numpy.typing import NDArray

from backtest.optimize import cv_blocks as _cvb

__all__ = [
    "CPCVSplits",
    "CPCV",
    "cpcv_splits",
    "cpcv_splits_from_boundaries",
    "num_cpcv_splits",
]

Array1D = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CPCVSplits:
    """A single CPCV split (ascending, disjoint indices)."""

    train_idx: Array1D
    test_idx: Array1D


def _validate_params(
    n_blocks: int,
    k_test_blocks: int,
    purge: int,
    embargo: int | float,
) -> None:
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2.")
    if not (1 <= k_test_blocks < n_blocks):
        raise ValueError("k_test_blocks must be in [1, n_blocks-1].")
    if purge < 0:
        raise ValueError("purge must be >= 0.")
    if isinstance(embargo, float):
        if not (0.0 <= embargo < 1.0):
            raise ValueError("embargo as float must be in [0.0, 1.0).")
    elif embargo < 0:
        raise ValueError("embargo must be >= 0.")


def _compute_block_boundaries(n_samples: int, n_blocks: int) -> Array1D:
    """
    Return a 'boundaries' array of length n_blocks+1 with cumulative edges.
    Blocks are consecutive and as evenly sized as possible; any remainder is
    assigned to the first blocks.
    """
    return _cvb._compute_block_boundaries(n_samples, n_blocks)


def _embargo_len(embargo: int | float, block_len: int) -> int:
    """Compute embargo length in samples for this side of a block."""
    return _cvb._embargo_len(embargo, block_len)


def _concat_blocks(boundaries: Array1D, block_ids: Sequence[int]) -> Array1D:
    """Concatenate indices of multiple blocks in ascending order."""
    if not block_ids:
        return np.empty(0, dtype=np.int64)
    parts: List[Array1D] = []
    for b in block_ids:
        left = int(boundaries[b])
        right = int(boundaries[b + 1])
        if right > left:
            parts.append(np.arange(left, right, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts, dtype=np.int64)


def _train_indices_with_purge_embargo(
    boundaries: Array1D,
    test_blocks: Sequence[int],
    purge: int,
    embargo: int | float,
) -> Array1D:
    """
    Build train indices as the complement of test blocks and trim side-selectively.
    If the left neighbor is a test block, trim left by purge + embargo.
    If the right neighbor is a test block, trim right by purge + embargo.
    """
    return _cvb._train_indices_with_purge_embargo(
        boundaries, test_blocks, purge, embargo
    )


def num_cpcv_splits(n_blocks: int, k_test_blocks: int) -> int:
    """Number of CPCV splits = C(n_blocks, k_test_blocks)."""
    if not (1 <= k_test_blocks < n_blocks):
        raise ValueError("k_test_blocks must be in [1, n_blocks-1].")
    return comb(n_blocks, k_test_blocks)


def _reservoir_sample_combinations(
    n_blocks: int,
    k_test_blocks: int,
    max_splits: int,
    rng: np.random.Generator,
) -> List[Tuple[int, ...]]:
    """
    Reservoir sampling over all combinations C(n,k) to draw max_splits
    representative combinations without full materialization.
    """
    if max_splits <= 0:
        return []
    reservoir: List[Tuple[int, ...]] = []
    for t, combo in enumerate(combinations(range(n_blocks), k_test_blocks), start=1):
        if t <= max_splits:
            reservoir.append(combo)
        else:
            j = rng.integers(1, t + 1)  # inkl. t
            if j <= max_splits:
                idx = int(rng.integers(0, max_splits))
                reservoir[idx] = combo
    # Sort for deterministic output (lexicographic).
    reservoir.sort()
    return reservoir


def _iter_combinations_ordered(
    n_blocks: int,
    k_test_blocks: int,
    *,
    max_splits: int | None,
    shuffle: bool,
    random_state: int | None,
) -> Iterable[Tuple[int, ...]]:
    """Generate (optionally capped) combinations in deterministic order."""
    total = num_cpcv_splits(n_blocks, k_test_blocks)

    if shuffle:
        # Shuffle only makes sense with max_splits (reservoir sampling).
        if max_splits is None:
            raise ValueError("shuffle=True requires max_splits to be set.")
        rng = np.random.default_rng(random_state if random_state is not None else 0)
        sample = _reservoir_sample_combinations(
            n_blocks, k_test_blocks, max_splits, rng
        )
        return sample

    # Deterministic lexicographic order (optionally capped via islice).
    combo_iter: Iterable[Tuple[int, ...]] = combinations(range(n_blocks), k_test_blocks)
    if (max_splits is not None) and (max_splits < total):
        combo_iter = islice(combo_iter, max_splits)
    return combo_iter


@overload
def cpcv_splits(
    n_samples: int,
    n_blocks: int,
    k_test_blocks: int,
    *,
    purge: int = ...,
    embargo: int | float = ...,
    max_splits: int | None = ...,
    shuffle: bool = ...,
    random_state: int | None = ...,
) -> Iterator[CPCVSplits]: ...
@overload
def cpcv_splits(
    n_samples: int,  # kept for overload symmetry; not used if 'boundaries' provided
    n_blocks: int,  # kept for overload symmetry; not used if 'boundaries' provided
    k_test_blocks: int,
    *,
    boundaries: Array1D,
    purge: int = ...,
    embargo: int | float = ...,
    max_splits: int | None = ...,
    shuffle: bool = ...,
    random_state: int | None = ...,
) -> Iterator[CPCVSplits]: ...


def cpcv_splits(
    n_samples: int,
    n_blocks: int,
    k_test_blocks: int,
    *,
    purge: int = 0,
    embargo: int | float = 0,
    max_splits: int | None = None,
    shuffle: bool = False,
    random_state: int | None = None,
    boundaries: Array1D | None = None,
) -> Iterator[CPCVSplits]:
    """
    CPCV splits (Combinatorial Purged Cross-Validation) for time series.

    Parameters
    ----------
    n_samples:
        Total length of the sample (0..n_samples-1). Ignored when 'boundaries' is provided.
    n_blocks:
        Number of consecutive time blocks.
    k_test_blocks:
        Number of test blocks per split (combinations).
    purge:
        Additional safety gap (samples) at block borders toward test blocks.
    embargo:
        Embargo at test borders (int -> samples per side, float -> fraction of block).
    max_splits:
        Cap the number of splits, or use reservoir sampling when shuffle=True.
    shuffle:
        Only allowed with max_splits; random but reproducible via random_state.
    random_state:
        Seed for sampling (only relevant when shuffle=True).
    boundaries:
        Optional pre-defined block boundaries (array length n_blocks+1).

    Notes
    -----
    Output indices are strictly increasing and train/test disjoint.
    """
    _validate_params(n_blocks, k_test_blocks, purge, embargo)

    if boundaries is None:
        boundaries = _compute_block_boundaries(n_samples, n_blocks)
    else:
        if boundaries.ndim != 1 or len(boundaries) != (n_blocks + 1):
            raise ValueError("boundaries must be 1D with length n_blocks+1.")
        if not np.all(boundaries[:-1] <= boundaries[1:]):
            raise ValueError("boundaries must be non-decreasing.")
        # n_samples is derivable from boundaries but not used here.

    combo_iter = _iter_combinations_ordered(
        n_blocks=n_blocks,
        k_test_blocks=k_test_blocks,
        max_splits=max_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for test_blocks in combo_iter:
        test_idx = _concat_blocks(boundaries, test_blocks)
        train_idx = _train_indices_with_purge_embargo(
            boundaries, test_blocks, purge, embargo
        )
        # Sicherheit: Disjunktheit (optional, aber O(n) – daher nicht per assert erzwungen)
        # If needed, locally test that intersection is empty:
        # if test_idx.size and train_idx.size:
        #     assert np.intersect1d(test_idx, train_idx).size == 0
        yield CPCVSplits(train_idx=train_idx, test_idx=test_idx)


def cpcv_splits_from_boundaries(
    boundaries: Array1D,
    k_test_blocks: int,
    *,
    purge: int = 0,
    embargo: int | float = 0,
    max_splits: int | None = None,
    shuffle: bool = False,
    random_state: int | None = None,
) -> Iterator[CPCVSplits]:
    """Variant with explicit block boundaries (e.g., shared multi-asset index)."""
    n_blocks = len(boundaries) - 1
    # n_samples not required; pass a dummy value so the overload stays consistent.
    yield from cpcv_splits(
        n_samples=int(boundaries[-1]),
        n_blocks=n_blocks,
        k_test_blocks=k_test_blocks,
        purge=int(purge),
        embargo=embargo,
        max_splits=max_splits,
        shuffle=shuffle,
        random_state=random_state,
        boundaries=boundaries,
    )


class CPCV:
    """
    Convenience class API similar to sklearn splitters (simplified).

    Example
    -------
    >>> splitter = CPCV(n_blocks=10, k_test_blocks=2, purge=5, embargo=0.05, max_splits=32)
    >>> for split in splitter.split(n_samples=len(X)):
    ...     train, test = split.train_idx, split.test_idx
    """

    __slots__ = (
        "_n_blocks",
        "_k_test_blocks",
        "_purge",
        "_embargo",
        "_max_splits",
        "_shuffle",
        "_random_state",
        "_boundaries",
    )

    def __init__(
        self,
        *,
        n_blocks: int,
        k_test_blocks: int,
        purge: int = 0,
        embargo: int | float = 0,
        max_splits: int | None = None,
        shuffle: bool = False,
        random_state: int | None = None,
        boundaries: Array1D | None = None,
    ) -> None:
        _validate_params(n_blocks, k_test_blocks, purge, embargo)
        if boundaries is not None:
            if boundaries.ndim != 1 or len(boundaries) != (n_blocks + 1):
                raise ValueError("boundaries must be 1D with length n_blocks+1.")
            if not np.all(boundaries[:-1] <= boundaries[1:]):
                raise ValueError("boundaries must be non-decreasing.")

        self._n_blocks = int(n_blocks)
        self._k_test_blocks = int(k_test_blocks)
        self._purge = int(purge)
        self._embargo = embargo
        self._max_splits = max_splits
        self._shuffle = bool(shuffle)
        self._random_state = random_state
        self._boundaries = boundaries

    def get_n_splits(self) -> int:
        """Theoretical number of splits without capping."""
        return num_cpcv_splits(self._n_blocks, self._k_test_blocks)

    def with_boundaries(self, boundaries: Array1D) -> "CPCV":
        """Return a copy with explicit boundaries (useful for multi-asset indices)."""
        return CPCV(
            n_blocks=self._n_blocks,
            k_test_blocks=self._k_test_blocks,
            purge=self._purge,
            embargo=self._embargo,
            max_splits=self._max_splits,
            shuffle=self._shuffle,
            random_state=self._random_state,
            boundaries=boundaries,
        )

    def split(self, *, n_samples: int | None = None) -> Iterator[CPCVSplits]:
        """Generate splits using explicit boundaries or an n_samples length."""
        boundaries = self._boundaries
        if boundaries is None:
            if n_samples is None:
                raise ValueError(
                    "Either pass 'n_samples' here or provide 'boundaries' in the constructor."
                )
            boundaries = _compute_block_boundaries(int(n_samples), self._n_blocks)

        yield from cpcv_splits(
            n_samples=int(boundaries[-1]),
            n_blocks=self._n_blocks,
            k_test_blocks=self._k_test_blocks,
            purge=self._purge,
            embargo=self._embargo,
            max_splits=self._max_splits,
            shuffle=self._shuffle,
            random_state=self._random_state,
            boundaries=boundaries,
        )
