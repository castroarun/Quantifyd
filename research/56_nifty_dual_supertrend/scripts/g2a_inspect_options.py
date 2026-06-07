#!/usr/bin/env python3
"""Inspect options_data.db: schema, coverage, and feasibility of reconstructing
NIFTY 30-min bars from per-minute underlying_spot for the G2 real-chain replay."""
import sqlite3, os
from datetime import datetime
from collections import defaultdict

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
con = sqlite3.connect(DB)

print("=== TABLES ===")
tabs = [r[0] for r in con.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(tabs)

for t in tabs:
    try:
        cols = con.execute(f"PRAGMA table_info({t})").fetchall()
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"\n--- {t}  ({cnt:,} rows) ---")
        print("  cols:", [c[1] for c in cols])
    except Exception as e:
        print(f"  {t}: {e}")

# Find the main chain table (most rows)
main = max(tabs, key=lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
print(f"\n=== MAIN TABLE = {main} ===")
cols = [c[1] for c in con.execute(f"PRAGMA table_info({main})").fetchall()]
print("cols:", cols)

# date range + underlyings
dcol = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else cols[0])
ucol = next((c for c in cols if c in ("underlying","name","symbol","index_name")), None)
print(f"date col guessed: {dcol}; underlying col: {ucol}")
try:
    rng = con.execute(f"SELECT MIN({dcol}),MAX({dcol}) FROM {main}").fetchone()
    print("date range:", rng)
except Exception as e:
    print("range err", e)
if ucol:
    us = con.execute(f"SELECT {ucol},COUNT(*) FROM {main} GROUP BY {ucol}").fetchall()
    print("underlyings:", us)

print("\n=== SAMPLE ROW ===")
for r in con.execute(f"SELECT * FROM {main} LIMIT 3"):
    print(dict(zip(cols, r)))

# Feasibility: reconstruct NIFTY 30-min from underlying_spot
if ucol and "underlying_spot" in cols:
    print("\n=== NIFTY underlying_spot coverage ===")
    q = (f"SELECT {dcol}, underlying_spot FROM {main} "
         f"WHERE {ucol} LIKE '%NIFTY%' AND {ucol} NOT LIKE '%BANK%' "
         f"AND underlying_spot IS NOT NULL ORDER BY {dcol}")
    rows = con.execute(q).fetchall()
    print(f"  rows with NIFTY spot: {len(rows):,}")
    if rows:
        days = defaultdict(int)
        for d, s in rows:
            day = str(d)[:10]
            days[day] += 1
        print(f"  distinct days: {len(days)}  [{min(days)} .. {max(days)}]")
        print(f"  spot sample: {rows[0]} ... {rows[-1]}")
con.close()
