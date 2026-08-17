#!/usr/bin/env python3
"""Retire the ORB-index strangle paper book (2026-08-17).

Evidence: 342 trades, net -Rs 1,97,181, all 10 variants negative, GROSS
already -Rs 80,403 (so not a cost problem), win rate 43.9% vs the 70-96%
its original backtest claimed, payoff 0.37x needing a 73% win rate to break
even. Retired rather than tuned.

Actions:
  1. disable every variant in strangle_trading.db (stops new entries)
  2. neutralise the three APScheduler jobs in app.py
  3. leave the trade history intact for the record
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, '/home/arun/quantifyd')

DB = Path('/home/arun/quantifyd/backtest_data/strangle_trading.db')
APP = Path('/home/arun/quantifyd/app.py')

# ---- 1. disable variants ----
con = sqlite3.connect(str(DB))
open_pos = con.execute(
    "SELECT COUNT(*) FROM strangle_positions WHERE status='OPEN'").fetchone()[0]
cur = con.execute('UPDATE strangle_variants SET enabled=0 WHERE enabled=1')
con.commit()
print(f'variants disabled: {cur.rowcount}; open positions right now: {open_pos}')
for r in con.execute('SELECT variant_id, enabled FROM strangle_variants ORDER BY variant_id'):
    print(f'   {r[0]:20s} enabled={r[1]}')
con.close()

# ---- 2. stop the scheduler jobs ----
s = APP.read_text()
if 'RETIRED 2026-08-17' in s:
    print('app.py already patched')
else:
    old = """try:
    # Master tick: every 60s during 9:15-15:30 IST Mon-Fri
    scheduler.add_job(
        _strangle_master_tick,
        'cron', day_of_week='mon-fri',
        hour='9-15', minute='*', second='5',
        id='strangle_master_tick', replace_existing=True,
    )
    # Belt-and-suspenders EOD square-off
    scheduler.add_job(
        _strangle_eod_squareoff,
        'cron', day_of_week='mon-fri', hour=15, minute=25,
        id='strangle_eod_squareoff', replace_existing=True,
    )
    scheduler.add_job(
        _strangle_daily_summary,
        'cron', day_of_week='mon-fri', hour=16, minute=0,
        id='strangle_daily_summary', replace_existing=True,
    )
    logger.info(
        "Strangle scheduled jobs registered: master_tick(every 60s 9-15 Mon-Fri), "
        "eod_squareoff(15:25), daily_summary(16:00)"
    )"""
    new = """# ORB-index strangle book RETIRED 2026-08-17 — 342 trades, net -Rs 1.97L,
# all 10 variants negative, gross negative before costs, live win rate 43.9%
# vs the 70-96% its backtest claimed. Jobs disabled; history kept; the API
# endpoints still serve the record read-only.
try:
    if False:  # retired — scheduler registration intentionally skipped
        scheduler.add_job(
            _strangle_master_tick,
            'cron', day_of_week='mon-fri',
            hour='9-15', minute='*', second='5',
            id='strangle_master_tick', replace_existing=True,
        )
        scheduler.add_job(
            _strangle_eod_squareoff,
            'cron', day_of_week='mon-fri', hour=15, minute=25,
            id='strangle_eod_squareoff', replace_existing=True,
        )
        scheduler.add_job(
            _strangle_daily_summary,
            'cron', day_of_week='mon-fri', hour=16, minute=0,
            id='strangle_daily_summary', replace_existing=True,
        )
    for _jid in ('strangle_master_tick', 'strangle_eod_squareoff',
                 'strangle_daily_summary'):
        try:
            scheduler.remove_job(_jid)
        except Exception:
            pass
    logger.info("Strangle (ORB-index) RETIRED 2026-08-17 — no scheduled jobs")"""
    assert old in s, 'strangle job block not found'
    APP.write_text(s.replace(old, new, 1))
    print('app.py PATCHED — strangle jobs retired')
