"""Quick state check for market_data.db catchup verification."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "backtest_data" / "market_data.db"
con = sqlite3.connect(str(DB))

# Per-timeframe latest date + symbol count
print("=== Per-timeframe state ===")
for r in con.execute(
    "SELECT timeframe, COUNT(DISTINCT symbol), MAX(date) "
    "FROM market_data_unified GROUP BY timeframe ORDER BY timeframe"
).fetchall():
    print(f"  {r[0]:10s} symbols={r[1]:>4}  max_date={r[2]}")

# 5-min: how many stocks are caught up to today vs stale
print("\n=== 5-min freshness ===")
total = con.execute(
    "SELECT COUNT(DISTINCT symbol) FROM market_data_unified WHERE timeframe='5minute'"
).fetchone()[0]
fresh = con.execute(
    "SELECT COUNT(DISTINCT symbol) FROM market_data_unified "
    "WHERE timeframe='5minute' AND date > '2026-05-06'"
).fetchone()[0]
mid = con.execute(
    "SELECT COUNT(DISTINCT symbol) FROM market_data_unified "
    "WHERE timeframe='5minute' AND date > '2026-04-01' AND date <= '2026-05-06'"
).fetchone()[0]
stale = con.execute(
    "SELECT COUNT(DISTINCT symbol) FROM market_data_unified "
    "WHERE timeframe='5minute' "
    "GROUP BY symbol HAVING MAX(date) <= '2026-04-01'"
).fetchall()
print(f"  Total 5-min stocks:               {total}")
print(f"  Have data > 2026-05-06 (today):   {fresh}")
print(f"  Have data Apr-1 to May-6:         {mid}")
print(f"  Still stale (≤ Apr-1):            {len(stale)}")

# Sample stale ones
if stale:
    print(f"\nSample stale (still need backfill): {[r[0] for r in stale[:10]]}")
con.close()
