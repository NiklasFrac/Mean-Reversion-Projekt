from __future__ import annotations

import processing.io_atomic as io_mod
from processing.io_atomic import collect_runtime_context


def test_collect_runtime_context_shapes(monkeypatch, tmp_path):
    # Freeze pip_freeze deterministisch
    def fake_freeze():
        meta = tmp_path / "backtest" / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        lock = meta / "requirements.lock"
        lock.write_text("pandas==0\nnumpy==0\n", encoding="utf-8")
        return str(lock), "c0ffee"

    monkeypatch.setattr(io_mod, "_pip_freeze", fake_freeze, raising=True)
    monkeypatch.setattr(io_mod, "_git_commit", lambda: "deadbeef", raising=True)
    monkeypatch.setattr(io_mod, "_git_dirty", lambda: False, raising=True)

    ctx = collect_runtime_context()
    assert "timestamp" in ctx and "python" in ctx and "platform" in ctx
    assert (
        isinstance(ctx["libs"], dict)
        and "pandas" in ctx["libs"]
        and "numpy" in ctx["libs"]
    )
    assert ctx["git_commit"] == "deadbeef"
    assert ctx["git_dirty"] is False
    assert isinstance(ctx.get("pip_lock_path"), (str, type(None)))
    assert isinstance(ctx.get("pip_lock_sha1"), (str, type(None)))
