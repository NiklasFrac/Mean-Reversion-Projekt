import numpy as np
import pandas as pd
import pytest

from analysis.pairs import (
    hierarchical_clustering_from_corr,
    list_high_pairs_vectorized,
    pairs_df_from_corr,
)
from analysis.utils import parse_pair


@pytest.mark.unit
def test_pairs_df_and_vectorized_consistency():
    cols = list("ABCD")
    C = pd.DataFrame(np.eye(4), index=cols, columns=cols)
    C.loc["A", "B"] = C.loc["B", "A"] = 0.85
    C.loc["C", "D"] = C.loc["D", "C"] = 0.75
    df_pairs = pairs_df_from_corr(C, threshold=0.7)
    vec_pairs = list_high_pairs_vectorized(C, threshold=0.7)
    names_df = {
        f"{left}-{right}"
        for left, right in df_pairs[["left", "right"]].itertuples(index=False)
    }
    names_vec = {f"{left}-{right}" for left, right, _ in vec_pairs}
    assert names_df == names_vec
    assert "A-B" in names_df and "C-D" in names_df


@pytest.mark.unit
def test_hierarchical_clustering_returns_linkage():
    cols = list("ABC")
    C = pd.DataFrame(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.3], [0.2, 0.3, 1.0]], index=cols, columns=cols
    )
    Z = hierarchical_clustering_from_corr(C, method="average")
    # linkage hat (n-1, 4)
    assert Z.shape == (len(cols) - 1, 4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "inp, exp",
    [
        ("A-B", ("A", "B")),
        ("A/B", ("A", "B")),
        ("A, B", ("A", "B")),
        ("A B", ("A", "B")),
        (["A", "B"], ("A", "B")),
        (("A", "B"), ("A", "B")),
    ],
)
def test_parse_pair_variants(inp, exp):
    assert parse_pair(inp) == exp
