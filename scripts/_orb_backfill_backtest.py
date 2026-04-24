"""
Backfill ORB backtest runs for a range of historical dates.

The daily backtest cron (`_orb_daily_backtest`) runs once at 15:45 each
session and stores results in `backtest_data/orb_backtest.db`. That DB
only covers days the cron has run on. This script populates it for
arbitrary historical dates so the retrofit analyzer has a wider sample.

Usage
-----
    venv/bin/python3 scripts/_orb_backfill_backtest.py [N_DAYS]

Default N_DAYS = 250. Runs the backtest from (today - N_DAYS) up to
(yesterday), skipping weekends and any date already in the DB.

Must be run on the VPS (or any host with a valid Kite access token
at `backtest_data/access_token.json`). Each day's backtest makes ~15
historical_data calls, so expect ~30 minutes for 250 days at Kite's
3 req/s rate limit.
"""

import os
import sys
import sqlite3
from datetime import date, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)


def _existing_run_dates(db_path):
    if not os.path.exists(db_path):
        return set()
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT run_date FROM orb_backtest_runs")}
    finally:
        conn.close()


def run(n_days: int = 250):
    from services.orb_daily_backtest import run_backtest, DB_PATH_DEFAULT

    today = date.today()
    start = today - timedelta(days=n_days)
    existing = _existing_run_dates(DB_PATH_DEFAULT)
    print(f"Target range: {start} → {today - timedelta(days=1)}")
    print(f"Already in DB: {len(existing)} days")

    todo = []
    d = start
    while d < today:
        if d.weekday() < 5 and d.isoformat() not in existing:
            todo.append(d)
        d += timedelta(days=1)
    print(f"To backfill:   {len(todo)} days (weekends + existing skipped)\n")

    ok = 0
    fail = 0
    for i, d in enumerate(todo, 1):
        try:
            out = run_backtest(run_date=d)
            ok += 1
            print(f"[{i:>3}/{len(todo)}] {d}: taken={out.get('trades_taken', 0)} "
                  f"pnl=Rs{out.get('net_pnl_inr', 0):+.0f}")
        except Exception as e:
            fail += 1
            print(f"[{i:>3}/{len(todo)}] {d}: FAILED — {e}")

    print(f"\nDone. ok={ok} fail={fail}")
    print(f"Now run:  venv/bin/python3 scripts/_orb_retrofit_backtest.py")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    run(n)
