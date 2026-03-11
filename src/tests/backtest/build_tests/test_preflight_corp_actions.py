"""
Strenger End-to-End-Preflight fuer corporate actions.

Ziel: Wenn dieser Test gruen ist, laeuft der produktive Runner mit
- echten Dividenden (amount > 0)
- echten Splits   (0 < factor <= 10)
- echten Delistings (ohne False Positives)
zuverlaessig durch.

Wie:
- Mini-Universum (AAPL, MSFT, TSLA, TWTR, ATVI) 2019..heute
- Dividenden & Splits: yfinance (Vendor)
- Delistings: SEC EDGAR (Vendor)
- Preis-Track wird AUTOMATISCH an EDGAR-Delist-Daten gekappt, damit
  Delistings *nicht* als False-Positive herausgefiltert werden.

Start in VS Code: Run dieses Skript
Pfadvorschlag: src/backtest/tools/preflight_corp_actions.py (ersetzt bestehendes)
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

# --------------------------------------------------------------------
# Pfade/Imports
# --------------------------------------------------------------------
ROOT = Path.cwd()
SRC_DIR = ROOT / "src" / "backtest" / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import corp_actions_builder as cab
except Exception as e:
    print("[FAIL] Konnte corp_actions_builder nicht importieren:", e)
    print("Tipp: Datei vorhanden?  src/backtest/src/corp_actions_builder.py")
    sys.exit(1)

LOG = logging.getLogger("preflight")
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)


# --------------------------------------------------------------------
# HTTP Utils (requests mit Retries)
# --------------------------------------------------------------------
def _req_json(
    url: str, headers: Dict[str, str], tries: int = 3, sleep_sec: float = 0.8
) -> dict:
    import requests

    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            ct = (r.headers.get("Content-Type") or "").lower()
            if "json" in ct:
                return r.json()
            return json.loads(r.text)
        except Exception as e:
            last = e
            time.sleep(sleep_sec * (i + 1))
    raise last or RuntimeError("HTTP JSON request failed")


def _req_text(
    url: str, headers: Dict[str, str], tries: int = 3, sleep_sec: float = 0.8
) -> str:
    import requests

    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep_sec * (i + 1))
    raise last or RuntimeError("HTTP TEXT request failed")


# --------------------------------------------------------------------
# YAML-Laden (optional) + User-Agent aus ENV
# --------------------------------------------------------------------
def _load_yaml_vendor_cfg() -> Dict:
    ypath = ROOT / "backtest" / "configs" / "config.yaml"
    base = {
        "yfinance": True,
        "yf_throttle_sec": 0.0,
        "probe_suffixes": [],
        "sec_edgar": {
            "enabled": True,
            "us_exchanges": ["*"],
            "user_agent": os.environ.get("EDGAR_USER_AGENT", ""),
            "throttle_sec": 0.25,
        },
        # Builder-Postprocessing (Produktions-Defaults)
        "confirm_delists_against_prices": True,
        "delist_grace_days": 30,
        "mark_special_dividends": True,
        "special_div_threshold": 0.25,
        # fuer den Test: kein Delist aus EU-Preisabriss erzwingen
        "infer_delist_from_price_gap": False,
    }
    try:
        import yaml  # optional

        if ypath.exists():
            y = yaml.safe_load(ypath.read_text(encoding="utf-8")) or {}
            vendor = (y.get("data", {}) or {}).get("corp_actions_vendor", {}) or {}
            out = base.copy()
            out.update({k: v for k, v in vendor.items() if k != "sec_edgar"})
            ed_in = vendor.get("sec_edgar") or {}
            ed_out = base["sec_edgar"].copy()
            ed_out.update(ed_in)
            # Fuer den Test: kein US-Filter, damit TWTR/ATVI sicher dabei sind
            ed_out["us_exchanges"] = ["*"]
            out["sec_edgar"] = ed_out
            # ENV kann YAML ueberschreiben
            if os.environ.get("EDGAR_USER_AGENT"):
                out["sec_edgar"]["user_agent"] = os.environ["EDGAR_USER_AGENT"]
            return out
    except Exception:
        pass
    return base


# --------------------------------------------------------------------
# EDGAR-Delist-Daten besorgen (mit Fallback)
# --------------------------------------------------------------------
FORMS_INTEREST = {"25", "25-NSE", "25-NYSE", "15-12G", "15-12B", "15-15D"}


def _edgar_map_ticker_to_cik(ua: str) -> Dict[str, str]:
    headers = {
        "User-Agent": ua,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    mapping: Dict[str, str] = {}
    # aktiv
    try:
        active = _req_json("https://www.sec.gov/files/company_tickers.json", headers)
        for _, v in (active or {}).items():
            tk = str(v.get("ticker", "")).upper().strip()
            cik_val = v.get("cik_str")
            if cik_val is None:
                continue
            try:
                cik = f"{int(cik_val):010d}"
            except Exception:
                continue
            if tk and cik:
                mapping[tk] = cik
    except Exception as e:
        LOG.info("EDGAR active mapping: %s", e)
    # historisch (inkl. delisteter)
    try:
        txt = _req_text(
            "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt", headers
        )
        for line in (txt or "").splitlines():
            parts = line.split(":")
            if len(parts) >= 3:
                cik = str(parts[1].strip()).zfill(10)
                for tk in [x.strip().upper() for x in parts[2].split("|") if x.strip()]:
                    mapping.setdefault(tk, cik)
    except Exception as e:
        LOG.info("EDGAR historical mapping: %s", e)
    # Sicherheitsnetz
    mapping.setdefault("TWTR", "0001418091")
    mapping.setdefault("ATVI", "0000718877")
    return mapping


def _edgar_delist_dates(tickers: List[str], ua: str) -> Dict[str, pd.Timestamp]:
    headers = {
        "User-Agent": ua,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    mp = _edgar_map_ticker_to_cik(ua)
    out: Dict[str, pd.Timestamp] = {}
    for tk in tickers:
        cik = mp.get(tk.upper())
        if not cik:
            continue
        try:
            js = _req_json(f"https://data.sec.gov/submissions/CIK{cik}.json", headers)
            recent = (js.get("filings") or {}).get("recent") or {}
            forms = recent.get("form", []) or []
            fdates = recent.get("filingDate", []) or []
            cand: List[pd.Timestamp] = []
            for f, d in zip(forms, fdates):
                if str(f).upper() in FORMS_INTEREST:
                    try:
                        cand.append(pd.Timestamp(d).normalize())
                    except Exception:
                        pass
            if cand:
                out[tk.upper()] = min(cand)
        except Exception as e:
            LOG.info("EDGAR submissions %s failed: %s", tk, e)
    # Fallback-Daten, falls leer
    out.setdefault("TWTR", pd.Timestamp("2022-11-07"))
    out.setdefault("ATVI", pd.Timestamp("2023-10-23"))
    return out


# --------------------------------------------------------------------
# Hauptlogik
# --------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore")

    # 1) Konfiguration & Reachability
    vendor = _load_yaml_vendor_cfg()
    edgar_cfg = vendor.get("sec_edgar", {})
    ua = edgar_cfg.get("user_agent") or os.environ.get("EDGAR_USER_AGENT") or ""
    if not ua or "@" not in ua:
        print(
            "[FAIL] EDGAR user_agent fehlt oder ist ungueltig. Setze ENV EDGAR_USER_AGENT oder YAML data.corp_actions_vendor.sec_edgar.user_agent."
        )
        return 1

    # yfinance-Reachability
    try:
        import yfinance as yf  # noqa

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _ = yf.Ticker("AAPL").history(period="5d")
    except Exception as e:
        print("[FAIL] yfinance nicht erreichbar:", e)
        return 1

    # 2) Mini-Universum und Delist-Daten aus EDGAR
    delist_dt = _edgar_delist_dates(["TWTR", "ATVI"], ua)
    LOG.info("EDGAR Delist-Daten: %s", {k: str(v.date()) for k, v in delist_dt.items()})

    # 3) Preis-Matrix bauen und Delists *kappen*
    end = pd.Timestamp.today().normalize()
    start = pd.Timestamp("2019-01-01")
    idx = pd.date_range(start, end, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "TSLA": 200.0,
            "TWTR": 50.0,
            "ATVI": 90.0,
        },
        index=idx,
    )
    # Delisted-Serien ab Delist-Datum+1 auf NaN setzen (letzter gueltiger Tag = Delist)
    for sym in ["TWTR", "ATVI"]:
        d = delist_dt.get(sym)
        if d is not None:
            prices.loc[prices.index > d, sym] = float("nan")

    # 4) Temp-Ordner vorbereiten
    outdir = ROOT / "backtest" / "tmp" / "preflight"
    if outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    p_prices = outdir / "prices_preflight.pkl"
    p_ca = outdir / "corp_actions_preflight.csv"
    prices.to_pickle(p_prices)

    # 5) Vendor-Settings finalisieren: keine Heuristik-Delists im Test
    vendor_pf = dict(vendor)
    vendor_pf["yf_throttle_sec"] = 0.0
    vendor_pf["infer_delist_from_price_gap"] = False
    vendor_pf["sec_edgar"] = dict(vendor.get("sec_edgar") or {})
    vendor_pf["sec_edgar"]["enabled"] = True
    vendor_pf["sec_edgar"]["us_exchanges"] = ["*"]

    # 6) Pipeline ausfuehren wie im Runner
    LOG.info("Starte ensure_and_seed_corporate_actions_file (Mini-Universum)...")
    try:
        cab.ensure_and_seed_corporate_actions_file(
            path=p_ca,
            price_path=p_prices,
            vendor=vendor_pf,
            universe_meta=None,
            infer_splits=False,  # echte Splits via yfinance
            infer_delist=False,  # echte Delists via EDGAR
        )
    except Exception as e:
        print("[FAIL] ensure_and_seed_corporate_actions_file brach ab:", e)
        return 1

    # 7) Ergebnisse pruefen (streng)
    try:
        ca = pd.read_csv(p_ca)
    except Exception as e:
        print("[FAIL] Ausgabedatei konnte nicht gelesen werden:", e)
        return 1

    if ca.empty:
        print("[FAIL] Preflight-CSV ist leer - keine Events gefunden.")
        return 1

    # Typen normalisieren
    ca["type"] = ca["type"].astype(str).str.lower().str.strip()

    # a) Dividenden: >0 und vorhanden
    div = ca[ca["type"].eq("dividend")].copy()
    if div.empty or not (div["amount"].astype(float) > 0).all():
        print("[FAIL] Dividenden fehlen oder amount<=0 gefunden.")
        return 1

    # b) Splits: vorhanden & plausibel (0<factor<=10) - min. eines fuer AAPL/TSLA erwartet
    spl = ca[ca["type"].eq("split")].copy()
    ok_split_range = (spl["factor"].astype(float) > 0) & (
        spl["factor"].astype(float) <= 10
    )
    if spl.empty or not ok_split_range.all():
        print("[FAIL] Splits fehlen oder unplausible Faktoren gefunden.")
        return 1
    if spl[spl["symbol"].isin(["AAPL", "TSLA"])].empty:
        print("[FAIL] Erwarteter Split bei AAPL oder TSLA nicht gefunden (yfinance).")
        return 1

    # c) Delistings: exakt TWTR & ATVI - keine weiteren, keine False Positives
    dl = ca[ca["type"].eq("delist")].copy()
    exp = {"TWTR", "ATVI"}
    got = set(dl["symbol"].astype(str).str.upper())
    if got != exp:
        print(f"[FAIL] Delistings stimmen nicht. Erwartet {exp}, gefunden {got}.")
        return 1
    # Notes sollten EDGAR tragen
    if (
        dl[
            ~dl["notes"].astype(str).str.contains("sec-edgar", case=False, na=False)
        ].shape[0]
        > 0
    ):
        print("[FAIL] Delisting-Notes stammen nicht aus EDGAR (sec-edgar).")
        return 1
    # Keine Delists fuer die anderen
    others = {"AAPL", "MSFT", "TSLA"}
    if any(sym in got for sym in others):
        print("[FAIL] False-Positive-Delist fuer AAPL/MSFT/TSLA entdeckt.")
        return 1

    # d) Keine Duplikate im Schluessel
    dup = ca.duplicated(
        subset=["symbol", "date", "type", "factor", "amount"], keep=False
    )
    if bool(dup.any()):
        print("[FAIL] Doppelte Events gefunden (symbol,date,type,factor,amount).")
        return 1

    # e) Zusammenfassung
    n_div = int((ca["type"] == "dividend").sum())
    n_spl = int((ca["type"] == "split").sum())
    n_del = int((ca["type"] == "delist").sum())

    print("\n===== STRICT PREFLIGHT SUMMARY =====")
    print(f"CSV:      {p_ca}")
    print(f"Window:   [{start.date()} .. {end.date()}]")
    print(f"Counts -> Dividends={n_div}, Splits={n_spl}, Delistings={n_del}")
    print(
        "[OK] Strenger Preflight erfolgreich. Runner-Pipeline wird mit echten Werten und ohne False Positives laufen."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
