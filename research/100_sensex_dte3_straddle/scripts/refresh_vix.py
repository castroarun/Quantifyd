# -*- coding: utf-8 -*-
"""Refresh INDIAVIX daily in market_data.db from its last stored date -> today, via Kite."""
import sqlite3, datetime as dt, sys
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
from services.kite_auth import get_access_token, KITE_API_KEY
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TOKEN = 264969  # INDIA VIX instrument token
kite = KiteConnect(api_key=KITE_API_KEY); kite.set_access_token(get_access_token())
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=60000")
last = c.execute("SELECT MAX(date) FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'").fetchone()[0]
frm = dt.date.fromisoformat(last[:10]) if last else dt.date(2015, 1, 1)
print("last stored INDIAVIX:", last, "-> fetching from", frm, "to today")
data = kite.historical_data(TOKEN, frm.isoformat(), dt.date.today().isoformat(), "day")
have = {r[0] for r in c.execute("SELECT date FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'")}
ins = 0
for row in data:
    d = row["date"].strftime("%Y-%m-%d")
    if d in have: continue
    c.execute("INSERT INTO market_data_unified(symbol,timeframe,date,open,high,low,close,volume) VALUES('INDIAVIX','day',?,?,?,?,?,?)",
              (d, row["open"], row["high"], row["low"], row["close"], row.get("volume", 0) or 0))
    ins += 1
c.commit()
mx = c.execute("SELECT MAX(date), COUNT(*) FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='day'").fetchone()
print(f"inserted {ins} new rows; INDIAVIX now through {mx[0]} ({mx[1]} rows)")
c.close()
