"""Backfill ORB daily backtest for the last N trading days.

Runs on VPS via _vps_query.py. Skips weekends. Re-stores each day via
services.orb_daily_backtest.run_backtest(run_date=D).
"""
import os, sys, time
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv('.env')

from datetime import date, timedelta
from services.orb_daily_backtest import run_backtest, get_backtest_run

DAYS_BACK = 60  # ~3 months of trading days

today = date.today()
start = today - timedelta(days=DAYS_BACK * 2)  # generous, we'll filter weekends

dates = []
d = today
while d >= start:
    if d.weekday() < 5:  # Mon-Fri
        dates.append(d)
    d -= timedelta(days=1)
dates = dates[:DAYS_BACK]
dates.reverse()  # oldest first

print(f'Backfilling {len(dates)} trading days from {dates[0]} to {dates[-1]}')

done = 0
skipped = 0
errors = 0
t0 = time.time()

for i, dt in enumerate(dates):
    # Skip if already stored
    existing = get_backtest_run(run_date=dt.isoformat())
    if existing and existing.get('trades_taken') is not None:
        skipped += 1
        continue
    try:
        out = run_backtest(run_date=dt)
        done += 1
        elapsed = time.time() - t0
        print(f'[{i+1}/{len(dates)}] {dt} taken={out["trades_taken"]} '
              f'blocked={out["signals_blocked"]} net=Rs{out["net_pnl_inr"]:+.0f} '
              f'(elapsed {elapsed:.0f}s)')
        sys.stdout.flush()
    except Exception as e:
        errors += 1
        print(f'[{i+1}/{len(dates)}] {dt} ERR: {e}')

print(f'\nDone. stored={done} skipped={skipped} errors={errors}')
