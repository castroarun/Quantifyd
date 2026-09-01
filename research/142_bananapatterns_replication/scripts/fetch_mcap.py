"""Fetch a one-time market-cap snapshot via yfinance for every daily symbol in the DB.

Output: results/mcap_snapshot.json  {symbol: {mcap: float, px: float, asof: date}}
Used by bluesky_replay.py --mcap-floor to build point-in-time mcap as
shares_adj (= mcap/px_today, split-safe on adjusted prices) x close(then).
Resumable: skips symbols already in the output file. Log to stdout.
"""
import json
import sqlite3
import time
from datetime import date
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
OUT = STUDY / 'results' / 'mcap_snapshot.json'

conn = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
syms = [r[0] for r in conn.execute(
    "select symbol from (select symbol, count(*) n from market_data_unified "
    "where timeframe='day' group by symbol) where n >= 260")]
conn.close()

snap = json.load(open(OUT)) if OUT.exists() else {}
print(f'{len(syms)} symbols; {len(snap)} already fetched', flush=True)

for i, s in enumerate(syms):
    if s in snap:
        continue
    ys = s.replace('-BE', '').replace('-B', '') + '.NS'
    try:
        fi = yf.Ticker(ys).fast_info
        mc = fi.get('market_cap') or fi.get('marketCap')
        px = fi.get('last_price') or fi.get('lastPrice')
        snap[s] = dict(mcap=float(mc) if mc else None,
                       px=float(px) if px else None, asof=str(date.today()))
    except Exception as e:
        snap[s] = dict(mcap=None, px=None, asof=str(date.today()), err=str(e)[:80])
    if (i + 1) % 50 == 0:
        json.dump(snap, open(OUT, 'w'))
        ok = sum(1 for v in snap.values() if v.get('mcap'))
        print(f'[{i+1}/{len(syms)}] saved; {ok} with mcap', flush=True)
    time.sleep(0.25)

json.dump(snap, open(OUT, 'w'))
ok = sum(1 for v in snap.values() if v.get('mcap'))
print(f'DONE: {ok}/{len(snap)} symbols with mcap', flush=True)
