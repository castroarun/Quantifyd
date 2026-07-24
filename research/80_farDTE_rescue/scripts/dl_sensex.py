#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download SENSEX index 5-min + daily history into market_data.db (symbol='SENSEX').
Chunked ~90-day windows via Kite historical. Runs on the VPS (canonical host)."""
import sqlite3, time, sys
from datetime import datetime, timedelta
import sys, os; sys.path.insert(0, "/home/arun/quantifyd")
from services.kite_service import get_kite

DB="/home/arun/quantifyd/backtest_data/market_data.db"
TOK=265  # SENSEX index
k=get_kite()
con=sqlite3.connect(DB); cur=con.cursor()

def upsert(rows, tf):
    n=0
    for r in rows:
        d=r['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(r['date'],'strftime') else str(r['date'])
        cur.execute("INSERT INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume,created_at) "
                    "VALUES ('SENSEX',?,?,?,?,?,?,?,datetime('now')) "
                    "ON CONFLICT DO NOTHING", (tf,d,r['open'],r['high'],r['low'],r['close'],r.get('volume',0) or 0))
        n+=cur.rowcount if cur.rowcount>0 else 0
    con.commit(); return n

# check unique index exists to make ON CONFLICT work; if not, dedupe manually
cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND sql LIKE '%symbol%timeframe%date%'")
has_uidx = cur.fetchone()[0] > 0

def upsert_safe(rows, tf):
    n=0
    for r in rows:
        d=r['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(r['date'],'strftime') else str(r['date'])
        ex=cur.execute("SELECT 1 FROM market_data_unified WHERE symbol='SENSEX' AND timeframe=? AND date=?",(tf,d)).fetchone()
        if ex: continue
        cur.execute("INSERT INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume,created_at) "
                    "VALUES ('SENSEX',?,?,?,?,?,?,?,datetime('now'))",
                    (tf,d,r['open'],r['high'],r['low'],r['close'],r.get('volume',0) or 0))
        n+=1
    con.commit(); return n

START=datetime(2021,1,1)
END=datetime.now()

for tf,interval,step in [('5minute','5minute',90),('day','day',1500)]:
    total=0; cur_dt=START
    while cur_dt<END:
        to=min(cur_dt+timedelta(days=step), END)
        try:
            d=k.historical_data(TOK, cur_dt, to, interval)
            got=upsert_safe(d, tf)
            total+=got
            print('  %s %s..%s  +%d (running %d)' % (tf, cur_dt.date(), to.date(), got, total)); sys.stdout.flush()
        except Exception as e:
            print('  %s %s ERR %s' % (tf, cur_dt.date(), str(e)[:70])); sys.stdout.flush()
        cur_dt=to+timedelta(days=1)
        time.sleep(0.4)
    print('=== %s DONE: %d rows ===' % (tf, total))

r=cur.execute("SELECT timeframe, COUNT(*), MIN(date), MAX(date) FROM market_data_unified WHERE symbol='SENSEX' GROUP BY timeframe").fetchall()
for x in r: print('FINAL', x)
con.close()
print('ALL DONE')
