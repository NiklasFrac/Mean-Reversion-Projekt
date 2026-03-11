from __future__ import annotations

import json
import pickle
from pathlib import Path


from processing.io_atomic import atomic_write_json, atomic_write_pickle


def test_atomic_write_json_and_pickle_are_atomic(tmp_path: Path):
    obj = {"a": 1, "b": [1, 2, 3]}
    jp = tmp_path / "out" / "x.json"
    pp = tmp_path / "out" / "x.pkl"

    # JSON
    atomic_write_json(obj, jp)
    assert jp.exists() and not (jp.with_suffix(jp.suffix + ".tmp")).exists()
    got = json.loads(jp.read_text(encoding="utf-8"))
    assert got == obj

    # Pickle
    atomic_write_pickle(obj, pp)
    assert pp.exists() and not (pp.with_suffix(pp.suffix + ".tmp")).exists()
    with pp.open("rb") as f:
        back = pickle.load(f)
    assert back == obj

    # Überschreiben ohne Race/Artefakt
    obj2 = {"a": 2}
    atomic_write_json(obj2, jp)
    assert json.loads(jp.read_text(encoding="utf-8")) == obj2
