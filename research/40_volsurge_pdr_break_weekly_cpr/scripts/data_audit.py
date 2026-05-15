"""Data-availability audit for the VOLSURGE/PDR/CPR system backtest.

Checks, for the 86 F&O stocks:
  - which intraday timeframes exist (5/10/15/30/60min) + daily
  - date ranges & row counts
  - whether ANY historical OPTIONS premium/IV data exists anywhere
Run on laptop (frozen snapshot) AND VPS (canonical) for a true picture.
"""
import sqlite3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.data_manager import FNO_LOT_SIZES  # noqa: E402

FNO = sorted(FNO_LOT_SIZES.keys())
DB = ROOT / 'backtest_data' / 'market_data.db'

print(f"DB: {DB}  exists={DB.exists()}  size={DB.stat().st_size/1e9:.2f} GB" if DB.exists() else f"DB MISSING: {DB}")
print(f"F&O universe size: {len(FNO)} stocks\n")

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

# 1. What timeframes exist at all?
cur.execute("SELECT DISTINCT timeframe FROM market_data_unified")
tfs = sorted(r[0] for r in cur.fetchall())
print(f"Stored timeframes in market_data_unified: {tfs}\n")

# 2. Per stored timeframe: how many F&O stocks covered + date range
qmarks = ",".join("?" * len(FNO))
for tf in tfs:
    cur.execute(
        f"""SELECT symbol, MIN(date), MAX(date), COUNT(*)
            FROM market_data_unified
            WHERE timeframe=? AND symbol IN ({qmarks})
            GROUP BY symbol""",
        [tf, *FNO],
    )
    rows = cur.fetchall()
    covered = {r[0] for r in rows}
    if not rows:
        print(f"[{tf:>9}] 0 / {len(FNO)} F&O stocks\n")
        continue
    gmin = min(r[1] for r in rows)
    gmax = max(r[2] for r in rows)
    # cohort split by earliest date
    long_hist = [r[0] for r in rows if r[1] <= '2019-01-01']
    print(f"[{tf:>9}] {len(covered)} / {len(FNO)} F&O stocks | dates {gmin[:10]} -> {gmax[:10]}")
    print(f"            long-history (<=2019) stocks: {len(long_hist)} -> {sorted(long_hist)}")
    missing = sorted(set(FNO) - covered)
    print(f"            F&O stocks with NO {tf} data ({len(missing)}): {missing}\n")

# 3. Daily coverage specifically (needed for weekly CPR + daily-trend filter)
cur.execute(
    f"""SELECT COUNT(DISTINCT symbol) FROM market_data_unified
        WHERE timeframe='day' AND symbol IN ({qmarks})""",
    FNO,
)
print(f"Daily-candle coverage (for weekly CPR + daily trend): {cur.fetchone()[0]} / {len(FNO)} F&O stocks")

# 4. Options data hunt — any table/db with option premium or IV history?
print("\n--- OPTIONS DATA HUNT ---")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(f"Tables in market_data.db: {[r[0] for r in cur.fetchall()]}")
conn.close()

for name in ['iv_history.db', 'options_data.db', 'option_chain.db']:
    for d in [ROOT / 'backtest_data', ROOT / 'data']:
        p = d / name
        if p.exists():
            c2 = sqlite3.connect(str(p))
            t = [r[0] for r in c2.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            print(f"{p} | size={p.stat().st_size} | tables={t}")
            for tbl in t:
                n = c2.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"    {tbl}: {n} rows")
            c2.close()
print("--- END ---")
