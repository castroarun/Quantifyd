#!/usr/bin/env python3
"""Remove 2026-08-20's phantom ORB rows.

Every position ORB opened today was closed at exactly its stop within 13-42
seconds by the paper-mode SL reconciliation bug (research/113). No order reached
the exchange — all ids are PAPER-* — so these fills never existed and their
-Rs 25k is fictitious. Leaving them would poison the book's record, the journal
and the liveness reader.

Backs the DB up first, prints every row it removes, and leaves the PAPER order
rows in place as an audit trail.
"""
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
DB = ROOT / 'backtest_data' / 'orb_trading.db'
DAY = '2026-08-20'
STATUS = ROOT / 'research' / '113_orb_paper_entry_forensic' / 'ORB_PAPER_ENTRY_FORENSIC_STATUS.md'

bak = DB.with_suffix(f'.db.bak_{DAY.replace("-", "")}_phantom_purge')
if not bak.exists():
    shutil.copy2(DB, bak)
print('backup:', bak.name)

c = sqlite3.connect(str(DB))
c.row_factory = sqlite3.Row
rows = list(c.execute(
    "SELECT id, instrument, direction, qty, entry_price, exit_price, entry_time, exit_time, "
    "exit_reason, pnl_inr FROM orb_positions WHERE trade_date=? AND exit_reason='SL_HIT_EXCHANGE'",
    (DAY,)))

if not rows:
    print('nothing to purge')
    sys.exit(0)

print(f'purging {len(rows)} phantom rows:')
total = 0.0
lines = []
for r in rows:
    d = dict(r)
    total += d['pnl_inr'] or 0
    held = ''
    try:
        held = f"{(datetime.fromisoformat(d['exit_time']) - datetime.fromisoformat(d['entry_time'])).seconds}s"
    except Exception:
        pass
    line = (f"  id={d['id']:4d} {d['instrument']:11s} {d['direction']:5s} qty {d['qty']:4d} "
            f"entry {d['entry_price']:>9} exit {d['exit_price']:>9} held {held:>5s} "
            f"pnl {d['pnl_inr']:>9}")
    print(line)
    lines.append(line)

c.execute("DELETE FROM orb_positions WHERE trade_date=? AND exit_reason='SL_HIT_EXCHANGE'", (DAY,))
c.commit()
left = c.execute('SELECT COUNT(*) FROM orb_positions WHERE trade_date=?', (DAY,)).fetchone()[0]
c.close()
print(f'removed {len(rows)} rows worth a fictitious Rs {total:,.0f}; {left} rows remain for {DAY}')

if STATUS.exists():
    with STATUS.open('a') as fh:
        fh.write(f"\n## Phantom-row purge — {datetime.now():%Y-%m-%d %H:%M} IST\n\n"
                 f"Removed {len(rows)} rows from `orb_positions` for {DAY} "
                 f"(exit_reason=SL_HIT_EXCHANGE), fictitious P&L Rs {total:,.0f}. "
                 f"DB backed up to `{bak.name}`. PAPER order rows kept as an audit trail.\n\n"
                 "```\n" + "\n".join(lines) + "\n```\n")
    print('status doc updated')
