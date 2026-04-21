"""Find which NAS system had the 10 closed trades at 24550 strike today."""
import sqlite3, glob, os
for db in sorted(glob.glob('/home/arun/quantifyd/backtest_data/nas_*.db')):
    try:
        c = sqlite3.connect(db)
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        pos_tbl = next((t for t in tables if 'position' in t.lower()), None)
        if not pos_tbl: continue
        try:
            rows = c.execute(f"SELECT COUNT(*), SUM(CASE WHEN strike=24550 THEN 1 ELSE 0 END) FROM {pos_tbl} WHERE DATE(entry_time)='2026-04-21'").fetchone()
            print(f'{os.path.basename(db):<35}  {pos_tbl}:  total_today={rows[0]}  @24550={rows[1]}')
        except Exception as e:
            print(f'{os.path.basename(db)}: query err {e}')
    except Exception as e:
        print(f'{db}: {e}')
