#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fill the NIFTY50 30-minute gap (2026-05-05 -> present) by aggregating the
5-minute series. Complete days only (day must reach the 15:25 bar). Idempotent:
deletes previously-derived rows in the window before inserting."""
import sqlite3
from collections import defaultdict

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
CUTOFF = '2026-05-05 15:59'          # existing native 30-min data ends 2026-05-05 15:15

con = sqlite3.connect(DB)
cur = con.cursor()

rows = cur.execute(
    "SELECT date, open, high, low, close, volume FROM market_data_unified "
    "WHERE symbol='NIFTY50' AND timeframe='5minute' AND date > ? ORDER BY date",
    (CUTOFF,)).fetchall()
print(f"5-min rows after cutoff: {len(rows)}")

by_day = defaultdict(list)
for r in rows:
    by_day[r[0][:10]].append(r)

out = []
for d in sorted(by_day):
    bars = by_day[d]
    if bars[-1][0][11:16] < '15:25':
        print(f"  skip partial day {d} (last bar {bars[-1][0][11:16]})")
        continue
    buckets = defaultdict(list)
    for b in bars:
        hh, mm = int(b[0][11:13]), int(b[0][14:16])
        idx = ((hh * 60 + mm) - 555) // 30          # 555 = 09:15
        if idx < 0:
            continue
        buckets[idx].append(b)
    for idx in sorted(buckets):
        bs = buckets[idx]
        t = 555 + idx * 30
        ts = f"{d} {t//60:02d}:{t%60:02d}:00"
        out.append((ts, bs[0][1], max(x[2] for x in bs), min(x[3] for x in bs),
                    bs[-1][4], sum(x[5] or 0 for x in bs)))

cur.execute("DELETE FROM market_data_unified WHERE symbol='NIFTY50' "
            "AND timeframe='30minute' AND date > ?", (CUTOFF,))
print(f"deleted {cur.rowcount} previously-derived rows")
cur.executemany(
    "INSERT INTO market_data_unified (symbol, timeframe, date, open, high, low, close, volume) "
    "VALUES ('NIFTY50', '30minute', ?, ?, ?, ?, ?, ?)", out)
con.commit()
print(f"inserted {len(out)} derived 30-min bars "
      f"({out[0][0] if out else '-'} .. {out[-1][0] if out else '-'})")
mx = cur.execute("SELECT MAX(date) FROM market_data_unified WHERE symbol='NIFTY50' "
                 "AND timeframe='30minute'").fetchone()[0]
print(f"NIFTY50 30-min now ends: {mx}")
