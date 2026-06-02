#!/usr/bin/env python3
"""
Download India VIX history (5minute + day) into the central market_data.db,
mirroring download_index_5min.py. Additive: writes a new symbol 'INDIAVIX'.
Gap-aware (safe to re-run). Run with the project venv python.
"""
import sys, json, time, sqlite3
sys.path.insert(0, "/home/arun/quantifyd")
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from config import KITE_API_KEY

QROOT="/home/arun/quantifyd"
DB=QROOT+"/backtest_data/market_data.db"
TOKEN=QROOT+"/backtest_data/access_token.json"
SYMBOL="INDIAVIX"
START=datetime(2024,3,1); END=datetime(2026,3,27)
RATE=0.35

def kite():
    k=KiteConnect(api_key=KITE_API_KEY)
    k.set_access_token(json.load(open(TOKEN))["access_token"]); return k

def existing(tf):
    c=sqlite3.connect(DB);r=c.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM market_data_unified "
      "WHERE symbol=? AND timeframe=?",(SYMBOL,tf)).fetchone();c.close();return r

def save(tf,candles):
    if not candles: return 0
    c=sqlite3.connect(DB);cur=c.cursor();ins=0
    for x in candles:
        dt=x["date"]; ds=dt.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dt,"strftime") else str(dt)
        cur.execute("INSERT INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume) "
                    "VALUES (?,?,?,?,?,?,?,0)",(SYMBOL,tf,ds,x["open"],x["high"],x["low"],x["close"]))
        ins+=1
    c.commit();c.close();return ins

def fetch(k,itok,tf,chunk_days):
    lo,hi,n=existing(tf)
    print(f"[{tf}] existing rows={n} {lo}..{hi}")
    cur=START; total=0
    while cur<END:
        nxt=min(cur+timedelta(days=chunk_days),END)
        try:
            cs=k.historical_data(itok,cur,nxt,tf)
            total+=save(tf,cs)
        except Exception as e:
            print("  chunk err",cur.date(),nxt.date(),str(e)[:120])
        cur=nxt+timedelta(days=1); time.sleep(RATE)
    lo,hi,n=existing(tf)
    print(f"[{tf}] DONE inserted~{total}; now rows={n} {lo}..{hi}")

def main():
    k=kite()
    itok=k.ltp(["NSE:INDIA VIX"])["NSE:INDIA VIX"]["instrument_token"]
    print("INDIA VIX token",itok)
    fetch(k,itok,"day",1800)
    fetch(k,itok,"5minute",7)

if __name__=="__main__": main()
