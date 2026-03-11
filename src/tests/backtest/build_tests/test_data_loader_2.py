from pathlib import Path

import pandas as pd

cfg_pairs_path = Path("backtest/data/filtered_pairs.pkl")
print("pairs path exists?", cfg_pairs_path.exists(), cfg_pairs_path.resolve())
if cfg_pairs_path.exists():
    # inspect file type
    try:
        # try open as pickle
        obj = pd.read_pickle(cfg_pairs_path)
        print("Pickle loaded. Type:", type(obj))
        # if dict-like, show sample
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print("dict keys sample:", keys[:20])
        elif isinstance(obj, (list, tuple)):
            print("list sample:", obj[:20])
        elif isinstance(obj, pd.DataFrame):
            print("DataFrame head:\n", obj.head(10))
        else:
            print("Object repr:", repr(obj)[:500])
    except Exception as e:
        print("Pickle load failed:", e)
        # try raw open inspect
        try:
            with open(cfg_pairs_path, "rb") as f:
                raw = f.read(1024)
            print("Raw bytes head:", raw[:200])
        except Exception as e2:
            print("Cannot read file:", e2)
else:
    print("Pairs file does not exist at that path. Check config.")
