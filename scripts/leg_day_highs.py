#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-leg DAY HIGH (max traded premium) for the active NIFTY NAS legs -> static/app/leg_highs.json.

Runs every minute during market hours via cron, OUT-OF-PROCESS (no gunicorn restart), the same
house pattern as sensex_live_writer / dump_nas_mtm. The NIFTY SystemPanel looks these up by
tradingsymbol to show a MAX column next to the live LTP. quote()'s ohlc.high is today's high of the
option's traded price. SENSEX carries its own `max` inside sensex_live.json (its writer).
"""
import json
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, "/home/arun/quantifyd")
from services.kite_service import get_kite

ROOT = "/home/arun/quantifyd"
OUT = os.path.join(ROOT, "static", "app", "leg_highs.json")

# NIFTY NAS system DBs (NFO). (db_file, positions_table)
DBS = [
    ("nas_trading.db", "nas_positions"),
    ("nas_atm_trading.db", "nas_atm_positions"),
    ("nas_atm2_trading.db", "nas_atm_positions"),
    ("nas_atm4_trading.db", "nas_atm_positions"),
    ("nas_916_otm_trading.db", "nas_positions"),
    ("nas_916_atm_trading.db", "nas_atm_positions"),
    ("nas_916_atm2_trading.db", "nas_atm_positions"),
    ("nas_916_atm4_trading.db", "nas_atm_positions"),
]


def main():
    syms = set()
    for db, tab in DBS:
        p = os.path.join(ROOT, "backtest_data", db)
        if not os.path.exists(p):
            continue
        try:
            c = sqlite3.connect("file:%s?mode=ro" % p, uri=True)
            for (t,) in c.execute(
                "SELECT tradingsymbol FROM %s WHERE status='ACTIVE' AND tradingsymbol IS NOT NULL" % tab
            ):
                if t:
                    syms.add(t)
            c.close()
        except Exception as e:
            print("db %s: %s" % (db, e))

    highs = {}
    if syms:
        try:
            q = get_kite().quote(["NFO:" + s for s in syms]) or {}
            for s in syms:
                v = q.get("NFO:" + s)
                if v:
                    hi = (v.get("ohlc") or {}).get("high")
                    if hi is not None:
                        highs[s] = round(hi, 1)
        except Exception as e:
            print("quote err: %s" % e)

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "highs": highs}, f)
    os.replace(tmp, OUT)
    print("wrote %s (%d legs)" % (OUT, len(highs)))


if __name__ == "__main__":
    main()
