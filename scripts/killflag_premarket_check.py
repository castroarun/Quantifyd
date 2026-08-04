# -*- coding: utf-8 -*-
"""Pre-market NAS panic-flag safety check (cron 09:05 IST Mon-Fri, after the 09:00 preopen restart,
before the 09:16 entry). Prevents a stale flag from a prior session silently eating a trading day
(the 2026-08-03 kill-flag incident, and the freeze-flag equivalent).

For BOTH the kill flag and the freeze flag:
  - armed on a PRIOR calendar day (STALE)  -> AUTO-CLEAR + log CRITICAL.
  - armed TODAY (deliberate same-day halt) -> LEAVE it, log WARNING.
  - not armed -> OK.
"""
import sys, json, datetime
sys.path.insert(0, "/home/arun/quantifyd")
from services.nas_kill_switch import (is_killed, get_kill_info, clear_killed,
                                       is_frozen, clear_frozen, FREEZE_FLAG_PATH)

now = datetime.datetime.now()
stamp = now.strftime("%Y-%m-%d %H:%M:%S")
today = now.strftime("%Y-%m-%d")


def handle(name, armed, armed_date, reason, clear_fn):
    if not armed:
        print("[%s] %s flag NOT armed — OK." % (stamp, name))
        return
    if armed_date and armed_date < today:
        clear_fn()
        print("[%s] CRITICAL: STALE %s flag AUTO-CLEARED. Armed %s (reason=%r) — carried from a prior "
              "session; cleared so today can trade." % (stamp, name, armed_date, reason))
    else:
        print("[%s] WARNING: %s flag armed TODAY (%s, reason=%r) — LEFT in place (deliberate same-day "
              "halt). No trading until cleared manually." % (stamp, name, armed_date, reason))


# --- kill flag ---
kinfo = get_kill_info() or {}
handle("KILL", is_killed(), (kinfo.get("killed_at", "") or "")[:10], kinfo.get("reason", ""), clear_killed)

# --- freeze flag ---
finfo = {}
try:
    if FREEZE_FLAG_PATH.exists():
        finfo = json.loads(FREEZE_FLAG_PATH.read_text())
except Exception:
    finfo = {}
handle("FREEZE", is_frozen(), (finfo.get("frozen_at", "") or "")[:10], finfo.get("reason", ""), clear_frozen)
