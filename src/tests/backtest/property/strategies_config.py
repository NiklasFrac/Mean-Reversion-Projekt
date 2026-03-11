from __future__ import annotations

from hypothesis import strategies as st

from .legacy_config_schema import (
    BacktestConfig,
    BorrowCfg,
    CalendarCfg,
    ExecLOB,
    ExecLight,
    ExecLightCfg,
    LightFees,
    RiskCaps,
    RiskCfg,
)


def _calendar_strat():
    return st.builds(
        CalendarCfg,
        exchange=st.sampled_from(["XNYS", "XNAS", "XETR"]),
        mapping=st.sampled_from(["strict", "prior"]),
        entry=st.sampled_from(["next_open", "prev_close"]),
        exit=st.sampled_from(["next_close", "same_close"]),
        lag_bars=st.integers(min_value=0, max_value=5),
        tz=st.just("America/New_York"),
    )


def _light_fees_strat():
    return st.builds(
        LightFees,
        per_trade=st.floats(min_value=0.0, max_value=100.0),
        bps=st.floats(min_value=0.0, max_value=5_000.0),
        per_share=st.floats(min_value=0.0, max_value=10.0),
        min_fee=st.floats(min_value=0.0, max_value=10.0),
        max_fee=st.floats(min_value=0.0, max_value=100.0),
    )


def _exec_light_strat():
    return st.builds(
        ExecLight,
        mode=st.just("light"),
        light=st.builds(
            ExecLightCfg,
            enabled=st.booleans(),
            reject_on_missing_price=st.booleans(),
            fees=_light_fees_strat(),
        ),
    )


def _exec_lob_strat():
    return st.builds(
        ExecLOB,
        mode=st.just("lob"),
        tick=st.floats(
            min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        levels=st.integers(min_value=1, max_value=10),
        size_per_level=st.integers(min_value=1, max_value=10_000),
        min_spread_ticks=st.integers(min_value=1, max_value=10),
        lam=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        max_add=st.integers(min_value=0, max_value=1000),
        bias_top=st.floats(
            min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        cancel_prob=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        max_cancel=st.integers(min_value=0, max_value=1000),
        steps_per_day=st.integers(min_value=1, max_value=1000),
        post_costs=st.fixed_dictionaries(
            {
                "per_trade": st.floats(min_value=0.0, max_value=100.0),
                "maker_bps": st.floats(min_value=-100.0, max_value=100.0),
                "taker_bps": st.floats(min_value=0.0, max_value=10_000.0),
            }
        ),
    )


def _exec_any_strat():
    return st.one_of(_exec_light_strat(), _exec_lob_strat())


def _borrow_strat():
    @st.composite
    def _build(draw):
        enabled = draw(st.booleans())
        csv_path = (
            None if not enabled else draw(st.sampled_from(["backtest/data/borrow.csv"]))
        )
        day_basis = draw(st.sampled_from([252, 360, 365]))
        return BorrowCfg(enabled=enabled, day_basis=day_basis, csv_path=csv_path)

    return _build()


def _risk_strat():
    return st.builds(
        RiskCfg,
        caps=st.builds(
            RiskCaps,
            max_gross=st.floats(min_value=0.0, max_value=5.0),
            max_net=st.floats(min_value=0.0, max_value=5.0),
            per_trade=st.floats(min_value=0.0, max_value=2.0),
            per_name=st.floats(min_value=0.0, max_value=2.0),
            clusters=st.none()
            | st.dictionaries(
                st.text(min_size=1, max_size=6),
                st.floats(min_value=0.0, max_value=2.0),
                max_size=5,
            ),
        ),
    )


def valid_cfgs():
    return st.builds(
        BacktestConfig,
        run_mode=st.sampled_from(["aktuell", "konservativ"]),
        version=st.none() | st.text(min_size=0, max_size=6),
        global_seed=st.none() | st.integers(min_value=0, max_value=2**31 - 1),
        calendar=_calendar_strat(),
        execution=_exec_any_strat(),
        borrow=_borrow_strat(),
        risk=_risk_strat(),
    )
