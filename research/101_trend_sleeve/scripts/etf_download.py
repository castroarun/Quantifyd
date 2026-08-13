# -*- coding: utf-8 -*-
"""Download diversifying NSE ETFs (global equity + commodity) daily history -> market_data_unified,
for the diversified managed-futures trend search. Resumable (skips existing dates)."""
import sqlite3, sys, datetime as dt, time
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
from services.kite_auth import get_access_token, KITE_API_KEY
kite = KiteConnect(api_key=KITE_API_KEY); kite.set_access_token(get_access_token())
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TOKENS = {  # symbol -> instrument_token
    "MON100": 5821185, "MOM100": 5484289, "MAFANG": 897793, "MASPTOP50": 1480193,
    "MAHKTECH": 1810945, "HNGSNGBEES": 4680705, "ITBEES": 4885505, "PSUBNKBEES": 3848193,
    "CPSEETF": 595969, "HDFCSML250": 3643649,
}
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=60000")
for sym, tok in TOKENS.items():
    have = {r[0] for r in c.execute("SELECT date FROM market_data_unified WHERE symbol=? AND timeframe='day'", (sym,))}
    frm = dt.date(2015, 1, 1); end = dt.date.today(); ins = 0
    d = frm
    while d < end:                                   # chunk ~700 days (daily API limit)
        d2 = min(d + dt.timedelta(days=700), end)
        try:
            h = kite.historical_data(tok, d.isoformat(), d2.isoformat(), "day")
        except Exception as e:
            print(f"{sym} {d}: ERR {e}"); d = d2 + dt.timedelta(days=1); time.sleep(0.5); continue
        for row in h:
            ds = row["date"].strftime("%Y-%m-%d")
            if ds in have: continue
            c.execute("INSERT INTO market_data_unified(symbol,timeframe,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?,?)",
                      (sym, "day", ds, row["open"], row["high"], row["low"], row["close"], row.get("volume", 0) or 0))
            have.add(ds); ins += 1
        d = d2 + dt.timedelta(days=1); time.sleep(0.35)
    c.commit()
    r = c.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM market_data_unified WHERE symbol=? AND timeframe='day'", (sym,)).fetchone()
    print(f"{sym}: +{ins} new | total {r[0]} rows {r[1]}..{r[2]}")
c.close()
print("DONE")
