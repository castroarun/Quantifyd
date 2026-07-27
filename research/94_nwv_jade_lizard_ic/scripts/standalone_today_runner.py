#!/usr/bin/env python3
"""One-day standalone runner for the NWV-JL paper book (research/94).

Used only when the book must be active BEFORE the app's next pre-open restart
registers the APScheduler jobs (no market-hours gunicorn restart allowed).
Runs monitor_job every 60s + the 15:25 pivot-exit check, then exits at 15:31.
Safe alongside nothing else: the in-app jobs don't exist until the next restart.
"""
import sys
import time
from datetime import datetime

sys.path.insert(0, '/home/arun/quantifyd')
from services.nwv_trade import monitor_job, pivot_exit_job  # noqa: E402

pivot_done = False
print(f"[standalone] NWV-JL runner up {datetime.now()}", flush=True)
while True:
    now = datetime.now()
    hm = now.strftime('%H:%M')
    if hm >= '15:31':
        break
    if '09:15' <= hm <= '15:30' and now.weekday() < 5:
        try:
            monitor_job()
        except Exception as e:
            print(f"[standalone] monitor err {e}", flush=True)
        if now.minute in (15, 45):
            try:
                pivot_exit_job()          # idempotent; guards its own time window
            except Exception as e:
                print(f"[standalone] pivot err {e}", flush=True)
    time.sleep(60)
print(f"[standalone] done for today {datetime.now()}", flush=True)
