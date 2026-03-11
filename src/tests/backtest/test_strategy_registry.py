import pytest

from backtest.runner_backtest import _build_strategy


def test_strategy_registry_unknown_name_raises() -> None:
    cfg = {"strategy": {"name": "does_not_exist"}}
    with pytest.raises(KeyError):
        _build_strategy(cfg, borrow_ctx=None)
