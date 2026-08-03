# -*- coding: utf-8 -*-
"""Pre-market NAS kill-flag safety check (cron 09:05 IST Mon-Fri, after the 09:00 preopen restart,
before the 09:16 entry). Prevents the 2026-08-03 incident where a kill flag armed on a prior session
silently blocked a whole trading day.

Rule:
  - kill flag armed on a PRIOR calendar day (STALE)  -> AUTO-CLEAR + log CRITICAL. A forgotten flag
    from a previous session must never silently eat a trading day; a genuine multi-day halt is rare
    and the user would re-arm.
  - kill flag armed TODAY (deliberate same-day halt) -> LEAVE it, log WARNING. Respect an intraday panic.
  - not armed -> OK, nothing to do.
"""
import sys, datetime
sys.path.insert(0, "/home/arun/quantifyd")
from services.nas_kill_switch import is_killed, get_kill_info, clear_killed

now = datetime.datetime.now()
stamp = now.strftime("%Y-%m-%d %H:%M:%S")
if not is_killed():
    print("[%s] kill flag NOT armed — OK, trading can proceed." % stamp)
    sys.exit(0)

info = get_kill_info() or {}
armed_at = info.get("killed_at", "") or ""
armed_date = armed_at[:10]
today = now.strftime("%Y-%m-%d")
reason = info.get("reason", "")

if armed_date and armed_date < today:
    clear_killed()
    print("[%s] CRITICAL: STALE kill flag AUTO-CLEARED. Armed %s (reason=%r). It was carried over from "
          "a prior session and would have blocked today's trading — cleared so paper/live can run." %
          (stamp, armed_at, reason))
else:
    print("[%s] WARNING: kill flag armed TODAY (%s, reason=%r) — LEFT in place (deliberate same-day halt). "
          "No trading until it is cleared manually." % (stamp, armed_at, reason))
