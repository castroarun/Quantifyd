#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SENSEX intraday P&L curve for today -> static/app/sensex_mtm.json.

The NIFTY 'Overall NAS · intraday P&L' chart (nas_mtm.json) covers only the 8 NIFTY systems; SENSEX
has no live SSE stream (standalone writer), so this reconstructs the SENSEX book's 1-minute intraday
P&L from the real option_chain premiums + the sensex_atm/atm2/atm4 position records:

    at minute t, book P&L = sum over legs of
        (entry - exit_price)*qty        if the leg is already closed at t   (realized)
        (entry - premium_t)*qty         if the leg is open at t             (unrealized, short)
        0                               if not yet entered

Runs every minute via cron (out-of-process, no gunicorn restart). option_chain updates ~1/min so the
curve stays current.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
IST = timezone(timedelta(hours=5, minutes=30))
OUT = "static/app/sensex_mtm.json"
DBS = ["backtest_data/sensex_atm_trading.db",
       "backtest_data/sensex_atm2_trading.db",
       "backtest_data/sensex_atm4_trading.db"]
OCDB = "backtest_data/options_data.db"


def _write(data):
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, OUT)


def main():
    today = datetime.now(IST).strftime("%Y-%m-%d")
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    legs = []
    for db in DBS:
        if not os.path.exists(db):
            continue
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
        c.row_factory = sqlite3.Row
        for r in c.execute(
            "SELECT tradingsymbol,entry_price,qty,entry_time,exit_time,exit_price,status "
            "FROM nas_atm_positions WHERE substr(entry_time,1,10)=? "
            "AND status IN ('ACTIVE','CLOSED') AND (exit_reason IS NULL OR exit_reason!='TEST_CLEANUP')",
            (today,)):
            legs.append(dict(
                sym=r["tradingsymbol"], entry=r["entry_price"] or 0, qty=r["qty"] or 0,
                entry_hm=(r["entry_time"] or "")[11:16],
                exit_hm=((r["exit_time"] or "")[11:16] if (r["status"] == "CLOSED" and r["exit_time"]) else None),
                exit_price=r["exit_price"] or 0))
        c.close()

    if not legs:
        _write({"generated_at": stamp, "points": [], "now": 0})
        print("no SENSEX legs today")
        return

    syms = list({l["sym"] for l in legs if l["sym"]})
    prem = {}   # {sym: {hh:mm: ltp}}
    c = sqlite3.connect("file:%s?mode=ro" % OCDB, uri=True)
    qs = ",".join("?" * len(syms))
    for sym, hm, ltp in c.execute(
        "SELECT tradingsymbol, substr(snapshot_time,12,5), ltp FROM option_chain "
        "WHERE tradingsymbol IN (%s) AND snapshot_time LIKE ?" % qs, syms + [today + "%"]):
        if ltp is not None:
            prem.setdefault(sym, {})[hm] = ltp
    c.close()

    ent0 = min((l["entry_hm"] for l in legs if l["entry_hm"]), default="09:16")
    grid = sorted({hm for m in prem.values() for hm in m if hm >= ent0})

    def prem_at(sym, hm):
        d = prem.get(sym, {})
        if hm in d:
            return d[hm]
        ks = [k for k in d if k <= hm]     # carry forward last known
        return d[max(ks)] if ks else None

    points = []
    for hm in grid:
        pnl = 0.0
        for l in legs:
            if not l["entry_hm"] or hm < l["entry_hm"]:
                continue
            if l["exit_hm"] and hm >= l["exit_hm"]:
                pnl += (l["entry"] - l["exit_price"]) * l["qty"]
            else:
                p = prem_at(l["sym"], hm)
                if p is None:
                    p = l["entry"]
                pnl += (l["entry"] - p) * l["qty"]
        points.append([hm, round(pnl)])

    now = points[-1][1] if points else 0
    lo = min((p[1] for p in points), default=0)
    hi = max((p[1] for p in points), default=0)
    _write({"generated_at": stamp, "points": points, "now": now, "lo": lo, "hi": hi})
    print("wrote %s (%d pts, now %s, lo %s hi %s)" % (OUT, len(points), now, lo, hi))


if __name__ == "__main__":
    main()
