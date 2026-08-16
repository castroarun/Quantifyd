#!/usr/bin/env python3
"""Journal cleanup + NAS-only sync.

1. Purge the ORB-INDEX paper rows (strangle_trading.db projections) — the
   user wants the journal to carry live trades only, NAS for now.
2. Re-sync every NAS book (11 databases) into journal_trades.

Safe to re-run: the journal upsert is keyed on (source_db, source_table,
source_id).
"""
import sqlite3
import sys

sys.path.insert(0, '/home/arun/quantifyd')

from config import DATA_DIR  # noqa: E402
from services.journal.journal_db import get_journal_db  # noqa: E402
from services.journal.sources import nas_source  # noqa: E402

JDB = DATA_DIR / 'journal.db'

con = sqlite3.connect(str(JDB))
before = con.execute('SELECT COUNT(*) FROM journal_trades').fetchone()[0]

# 1. purge ORB-INDEX paper projections
cur = con.execute(
    "DELETE FROM journal_trades WHERE source_db='strangle_trading'")
purged = cur.rowcount
con.commit()
after_purge = con.execute('SELECT COUNT(*) FROM journal_trades').fetchone()[0]
print(f'purged ORB-INDEX rows: {purged}  ({before} -> {after_purge})')

# orphaned child rows
for tbl, key in (('journal_trade_context', 'trade_id'),
                 ('journal_trade_notes', 'trade_id'),
                 ('journal_trade_tags', 'trade_id'),
                 ('journal_trade_screenshots', 'trade_id')):
    try:
        con.execute(
            f'DELETE FROM {tbl} WHERE {key} NOT IN '
            f'(SELECT id FROM journal_trades)')
    except sqlite3.OperationalError:
        pass
con.commit()
con.close()

# 2. sync all NAS books
db = get_journal_db()
rows = nas_source.fetch_closed_trades()
n = 0
for r in rows:
    try:
        db.upsert_trade(r)
        n += 1
    except Exception as e:
        print('upsert failed', r.get('strategy'), r.get('source_id'), e)
print(f'NAS rows upserted: {n}')

con = sqlite3.connect(str(JDB))
print('\njournal now holds:')
for row in con.execute(
        'SELECT strategy, mode, COUNT(*), MAX(exit_time) FROM journal_trades '
        'GROUP BY strategy, mode ORDER BY 3 DESC'):
    print(f'  {row[0]:16s} {row[1]:6s} {row[2]:4d}  last={str(row[3])[:16]}')
con.close()
print('DONE')
