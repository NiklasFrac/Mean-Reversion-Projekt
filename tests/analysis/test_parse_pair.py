# src/tests/analysis/test_parse_pair.py
import pytest

from analysis.utils import parse_pair


@pytest.mark.parametrize(
    "inp,exp",
    [
        (("A", "B"), ("A", "B")),
        ("A-B", ("A", "B")),
        ("A/B", ("A", "B")),
        ("A, B", ("A", "B")),
        ("A B", ("A", "B")),
    ],
)
def test_parse_pair_variants(inp, exp):
    assert parse_pair(inp) == exp


def test_parse_pair_invalid():
    with pytest.raises(ValueError):
        parse_pair("A_only")
