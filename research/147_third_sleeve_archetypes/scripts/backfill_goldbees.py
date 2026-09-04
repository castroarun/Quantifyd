"""Backfill GOLDBEES daily history 2007->2015 from Kite (VPS-only, additive:
INSERT OR IGNORE on dates we don't have). Then report the extended range.
"""
import json
import os
import sqlite3
import time
from datetime import date, timedelta

from kiteconnect import KiteConnect

ROOT = '/home/arun/quantifyd'
api_key = None
for line in open(f'{ROOT}/.env'):
    if line.startswith('KITE_API_KEY'):
        api_key = line.split('=', 1)[1].strip()
tok = json.load(open(f'{ROOT}/backtest_data/access_token.json'))
access = tok.get('access_token') or tok.get('token')
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access)

ins = [i for i in kite.instruments('NSE') if i['tradingsymbol'] == 'GOLDBEES']
assert ins, 'GOLDBEES not found on NSE'
token = ins[0]['instrument_token']
print('GOLDBEES token', token)

con = sqlite3.connect(f'{ROOT}/backtest_data/market_data.db')
have = {r[0][:10] for r in con.execute(
    "SELECT date FROM market_data_unified WHERE symbol='GOLDBEES' AND timeframe='day'")}
print(f'have {len(have)} rows, earliest {min(have) if have else None}')

start = date(2007, 1, 1)
end = date(2015, 1, 10)
rows = []
cur = start
while cur < end:
    chunk_end = min(cur + timedelta(days=1800), end)
    candles = kite.historical_data(token, cur.isoformat(), chunk_end.isoformat(), 'day')
    for cd in candles:
        d = cd['date'].date().isoformat()
        if d not in have:
            rows.append((f'GOLDBEES', 'day', d + ' 00:00:00', cd['open'], cd['high'],
                         cd['low'], cd['close'], cd['volume']))
    print(f'{cur} -> {chunk_end}: {len(candles)} candles', flush=True)
    cur = chunk_end + timedelta(days=1)
    time.sleep(0.4)

con.executemany(
    "INSERT INTO market_data_unified(symbol,timeframe,date,open,high,low,close,volume,created_at) "
    "VALUES (?,?,?,?,?,?,?,?,datetime('now'))", rows)
con.commit()
r = con.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM market_data_unified "
                "WHERE symbol='GOLDBEES' AND timeframe='day'").fetchone()
print(f'inserted {len(rows)} rows; GOLDBEES now {r[0]} rows {r[1][:10]} -> {r[2][:10]}')
