from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from universe.fundamentals import _best_float_pct as _best_float_pct_with_quality


@settings(
    max_examples=60,
    phases=[Phase.generate],
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    derandomize=True,
    database=None,
)
@given(
    price=st.floats(min_value=0.01, max_value=1e6),
    shares_out=st.floats(min_value=1e3, max_value=1e12),
    float_shares=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e12)),
    short_pct_float=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    shares_short=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e12)),
    held_inst=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    insiders=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    institutions=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    inst_of_float=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
def test_best_float_pct_range(
    price,
    shares_out,
    float_shares,
    short_pct_float,
    shares_short,
    held_inst,
    insiders,
    institutions,
    inst_of_float,
):
    info = {
        "floatShares": float_shares,
        "shortPercentOfFloat": short_pct_float,
        "sharesShort": shares_short,
        "heldPercentInstitutions": held_inst,
        "heldPercentInsiders": insiders,
    }
    mh = (insiders, institutions, inst_of_float)

    v, _quality = _best_float_pct_with_quality(info, shares_out, mh, held_inst)
    assert v is None or (0.0 <= v <= 1.0)
