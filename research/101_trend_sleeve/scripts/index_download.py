# -*- coding: utf-8 -*-
"""Download NIFTY 50 + BANKNIFTY index daily 2011-2015 (for ATM strike selection on the extended
monthly straddle). Into market_data_unified. Resumable."""
import sqlite3, sys, datetime as dt, time
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
from services.kite_auth import get_access_token, KITE_API_KEY
kite = KiteConnect(api_key=KITE_API_KEY); kite.set_access_token(get_access_token())
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TOKENS = {"NIFTY50": 256265, "BANKNIFTY": 260105}
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=60000")
for sym, tok in TOKENS.items():
    have = {r[0] for r in c.execute("SELECT date FROM market_data_unified WHERE symbol=? AND timeframe='day'", (sym,))}
    d = dt.date(2011, 1, 1); end = dt.date(2015, 12, 31); ins = 0
    while d < end:
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
    print(f"{sym}: +{ins} new | total {r[0]} {r[1]}..{r[2]}")
c.close(); print("DONE")
