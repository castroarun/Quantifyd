"""research/159 — data coverage probe. Read-only. Run on VPS."""
import sqlite3, os
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
print('DB:', DB, 'exists=', DB.exists())
con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
cur = con.cursor()

print('--- daily coverage ---')
for row in cur.execute("SELECT COUNT(DISTINCT symbol), COUNT(*), MIN(date), MAX(date) FROM market_data_unified WHERE timeframe='day'"):
    print('symbols=%s rows=%s min=%s max=%s' % row)

print('--- KMEW-like ---')
for row in cur.execute("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM market_data_unified WHERE timeframe='day' AND symbol LIKE '%KMEW%' GROUP BY symbol"):
    print(row)

print('--- ACCENT-like ---')
for row in cur.execute("SELECT symbol, COUNT(*), MIN(date), MAX(date) FROM market_data_unified WHERE timeframe='day' AND symbol LIKE '%ACCENT%' GROUP BY symbol"):
    print(row)

print('--- symbols with >=250 daily rows ---')
for row in cur.execute("SELECT COUNT(*) FROM (SELECT symbol FROM market_data_unified WHERE timeframe='day' GROUP BY symbol HAVING COUNT(*)>=250)"):
    print('count=', row[0])

print('--- last 6 distinct dates ---')
for row in cur.execute("SELECT date, COUNT(*) FROM market_data_unified WHERE timeframe='day' AND date>='2026-09-01' GROUP BY date ORDER BY date"):
    print(row)

print('--- KMEW tail ---')
for row in cur.execute("SELECT date,open,high,low,close,volume FROM market_data_unified WHERE timeframe='day' AND symbol='KMEW' ORDER BY date DESC LIMIT 5"):
    print(row)

print('--- KMEW head ---')
for row in cur.execute("SELECT date,open,high,low,close,volume FROM market_data_unified WHERE timeframe='day' AND symbol='KMEW' ORDER BY date ASC LIMIT 5"):
    print(row)

print('--- zero-volume day rows by year (phantom check) ---')
for row in cur.execute("SELECT substr(date,1,4) y, COUNT(*) FROM market_data_unified WHERE timeframe='day' AND (volume IS NULL OR volume=0) GROUP BY y ORDER BY y DESC LIMIT 8"):
    print(row)
con.close()
