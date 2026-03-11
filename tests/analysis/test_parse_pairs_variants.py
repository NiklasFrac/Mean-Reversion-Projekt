import pytest

import analysis.data_analysis as da


def test_parse_pair_variants():
    assert da.parse_pair("A-B") == ("A", "B")
    assert da.parse_pair("A/B") == ("A", "B")
    assert da.parse_pair("A,B") == ("A", "B")
    assert da.parse_pair("A B") == ("A", "B")
    assert da.parse_pair(["A", "B"]) == ("A", "B")


def test_parse_pair_errors():
    with pytest.raises(ValueError):
        da.parse_pair("single")
