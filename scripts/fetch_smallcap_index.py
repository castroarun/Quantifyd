"""Fetch REAL NIFTY Smallcap 250 (and Midcap 150 / Nifty 500) index history so the regime study can be
redone on a true signal instead of the 68%-accurate traded-value proxy.

Tries, in order:
  1. niftyindices.com historical-data endpoint (same host the live book already uses for constituents)
  2. Kite historical API via the project's kite_service (index instrument tokens)
Writes into market_data_unified as symbols NIFTYSMLCAP250 / NIFTYMIDCAP150 / NIFTY500.
Read-only on everything else; prints a clear report of what worked.
"""
import json, sys, time
import urllib.request
import sqlite3
from datetime import date

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TARGETS = [("NIFTY SMALLCAP 250", "NIFTYSMLCAP250"),
           ("NIFTY MIDCAP 150", "NIFTYMIDCAP150"),
           ("NIFTY 500", "NIFTY500")]
START, END = "01-Jan-2011", date.today().strftime("%d-%b-%Y")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


def try_niftyindices(index_name):
    url = "https://niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"
    payload = {"cinfo": json.dumps({"name": index_name, "startDate": START, "endDate": END,
                                    "indexName": index_name})}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json; charset=UTF-8", "User-Agent": UA,
                 "Referer": "https://niftyindices.com/reports/historical-data",
                 "Accept": "application/json, text/javascript, */*; q=0.01",
                 "X-Requested-With": "XMLHttpRequest"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.loads(r.read().decode())
    rows = json.loads(raw["d"]) if isinstance(raw.get("d"), str) else raw.get("d")
    out = []
    for x in rows or []:
        d = x.get("HistoricalDate") or x.get("Date")
        cl = x.get("CLOSE") or x.get("Close")
        op = x.get("OPEN") or x.get("Open") or cl
        hi = x.get("HIGH") or x.get("High") or cl
        lo = x.get("LOW") or x.get("Low") or cl
        if not d or not cl:
            continue
        try:
            from datetime import datetime as dt
            iso = dt.strptime(d.strip(), "%d %b %Y").date().isoformat()
        except Exception:
            try:
                iso = dt.strptime(d.strip(), "%d-%b-%Y").date().isoformat()
            except Exception:
                continue
        f = lambda v: float(str(v).replace(",", ""))
        out.append((iso, f(op), f(hi), f(lo), f(cl)))
    return sorted(set(out))


def try_kite(index_name):
    sys.path.insert(0, "/home/arun/quantifyd")
    from services.kite_service import get_kite_with_refresh
    k = get_kite_with_refresh()
    inst = k.instruments("NSE")
    cands = [i for i in inst if index_name.replace(" ", "") in i["tradingsymbol"].replace(" ", "").upper()
             or i["tradingsymbol"].upper() == index_name.replace(" ", "")]
    if not cands:
        cands = [i for i in inst if i.get("segment") == "INDICES"
                 and index_name.split()[-1] in i["tradingsymbol"].upper()
                 and ("SMALL" in i["tradingsymbol"].upper() if "SMALLCAP" in index_name else True)]
    if not cands:
        return []
    tok = cands[0]["instrument_token"]
    print(f"    kite token {tok} ({cands[0]['tradingsymbol']})", flush=True)
    out = []
    for y in range(2011, date.today().year + 1):
        try:
            d = k.historical_data(tok, f"{y}-01-01", f"{y}-12-31", "day")
            out += [(x["date"].date().isoformat(), x["open"], x["high"], x["low"], x["close"]) for x in d]
            time.sleep(0.35)
        except Exception as e:
            print(f"    {y}: {e}", flush=True)
    return sorted(set(out))


con = sqlite3.connect(DB)
print(f"Fetching real index history ({START} -> {END})\n")
for name, sym in TARGETS:
    have = con.execute("SELECT COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='day'",
                       (sym,)).fetchone()[0]
    if have > 500:
        print(f"{name}: already have {have} rows as {sym} — skipping"); continue
    rows = []
    for label, fn in (("niftyindices", try_niftyindices), ("kite", try_kite)):
        try:
            print(f"  {name} via {label} ...", flush=True)
            rows = fn(name)
            if rows:
                print(f"    OK {len(rows)} rows {rows[0][0]} .. {rows[-1][0]}", flush=True)
                break
            print("    empty", flush=True)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
    if not rows:
        print(f"  {name}: NO DATA obtained\n"); continue
    con.executemany(
        "INSERT OR IGNORE INTO market_data_unified(symbol,timeframe,date,open,high,low,close,volume) "
        "VALUES(?,'day',?,?,?,?,?,0)", [(sym, d, o, h, l, c) for d, o, h, l, c in rows])
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='day'",
                    (sym,)).fetchone()[0]
    print(f"  -> stored as {sym}: {n} rows\n", flush=True)
print("done")
