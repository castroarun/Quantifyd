# -*- coding: utf-8 -*-
"""Standalone live-position writer for the 3 SENSEX 9:16 executors.

Why standalone (not a Flask job): the live SENSEX book must be visible on the NAS page at the TOP
with a LIVE marking, but the app cannot be restarted while real-money SENSEX positions are open
(market-hours rule). A separate process reads the sensex_atm/atm2/atm4 DBs + live BFO LTPs every
~12s and writes static/app/sensex_live.json, which a new top card renders. No Flask restart.

After 15:30 this folds into app.py as a proper 10s job (alongside the NAS-RECONCILE-SENSEX fix) and
this standalone loop is retired.

Reads DB truth only (mode column decides live vs paper); never places orders.
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from config import SENSEX_ATM_DEFAULTS, SENSEX_ATM2_DEFAULTS, SENSEX_ATM4_DEFAULTS

IST = timezone(timedelta(hours=5, minutes=30))
OUT = "static/app/sensex_live.json"
SYS = [
    ("sensex_atm",  "SENSEX ATM",  "backtest_data/sensex_atm_trading.db",  SENSEX_ATM_DEFAULTS),
    ("sensex_atm2", "SENSEX ATM2", "backtest_data/sensex_atm2_trading.db", SENSEX_ATM2_DEFAULTS),
    ("sensex_atm4", "SENSEX ATM4", "backtest_data/sensex_atm4_trading.db", SENSEX_ATM4_DEFAULTS),
]


def _kite():
    from services.kite_service import get_kite
    return get_kite()


def _arm(sys_cfg, leg, spot):
    """Human-readable arm state for a leg, matching what the guardian checks."""
    sl = leg["sl_price"] or 0
    ms = sys_cfg.get("move_stop_pct") or 0
    notes = (leg.get("notes") or "")
    if ms:
        es = leg.get("entry_spot") or 0
        if not es:
            return "move-stop (no entry_spot!)"
        lo, hi = es * (1 - ms), es * (1 + ms)
        d = min(abs(hi - spot), abs(spot - lo)) if spot else 0
        return "move-stop %d–%d (%.0f away)" % (lo, hi, d)
    if sl >= 900000:
        return "NAKED — UNARMED"
    tag = " · BE-protect" if "PROTECTIVE" in notes else ""
    return "SL %.0f%s" % (sl, tag)


def _oc_day_highs(symbols):
    """{tradingsymbol: max 1-min LTP today} from options_data.db option_chain — robust vs opening/
    bad ticks (they don't survive the 1-min sampling) and covers CLOSED legs too (full-day history
    by tradingsymbol). 2026-08-12: kite.quote ohlc.high showed a phantom 461.9 vs a real 367 high."""
    out = {}
    syms = [x for x in symbols if x]
    if not syms:
        return out
    try:
        dbp = os.path.join(_ROOT, "backtest_data", "options_data.db")
        c = sqlite3.connect("file:%s?mode=ro" % dbp, uri=True)
        today = datetime.now(IST).strftime("%Y-%m-%d")
        qs = ",".join("?" * len(syms))
        for ts, mx in c.execute(
            "SELECT tradingsymbol, MAX(ltp) FROM option_chain "
            "WHERE tradingsymbol IN (%s) AND snapshot_time LIKE ? GROUP BY tradingsymbol" % qs,
            syms + [today + "%"]):
            if mx is not None:
                out[ts] = mx
        c.close()
    except Exception:
        pass
    return out


def build():
    kite = _kite()
    spot = None
    try:
        spot = kite.ltp(["BSE:SENSEX"])["BSE:SENSEX"]["last_price"]
    except Exception:
        pass

    today = datetime.now(IST).strftime("%Y-%m-%d")

    # Collect TODAY's legs per system — both the still-open legs AND the ones that already exited
    # today (e.g. an ATM strangle whose PE stopped out but CE survives). Showing the exited leg is
    # what makes the card honest: the system P&L must net the realized loss on the closed leg
    # against the unrealized gain on the survivor. Exclude test-cleanup rows. Only ACTIVE legs need
    # a live LTP round-trip.
    active_by_sys = {}
    all_syms = set()
    all_leg_syms = set()   # every leg today (active + closed) for the day-high lookup
    for key, _lab, db, _cfg in SYS:
        if not os.path.exists(db):
            active_by_sys[key] = []
            continue
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        # status IN ('ACTIVE','CLOSED') only: an open position or a leg that genuinely filled and
        # exited today. Excludes order-rejected / failed / pending — those never filled, so they are
        # NOT positions and must never show as a live short or count toward P&L (a rejected leg with
        # exit_price=0 would otherwise fake a full-credit profit).
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM nas_atm_positions "
            "WHERE substr(entry_time,1,10)=? "
            "AND status IN ('ACTIVE','CLOSED') "
            "AND (exit_reason IS NULL OR exit_reason!='TEST_CLEANUP') "
            "ORDER BY strangle_id, leg", (today,))]
        c.close()
        active_by_sys[key] = rows
        for r in rows:
            if r.get("tradingsymbol"):
                all_leg_syms.add(r["tradingsymbol"])
            if r.get("status") == "ACTIVE" and r.get("tradingsymbol"):
                all_syms.add(r["tradingsymbol"])

    ltp_map = {}
    if all_syms:
        try:
            q = kite.ltp(["BFO:" + s for s in all_syms]) or {}
            for s in all_syms:
                v = q.get("BFO:" + s)
                if v and v.get("last_price") is not None:
                    ltp_map[s] = v["last_price"]
        except Exception:
            pass
    # robust day-high (max traded premium) per leg from the 1-min option_chain — active AND closed
    high_map = _oc_day_highs(all_leg_syms)

    systems = []
    day_pnl = 0.0
    for key, lab, _db, cfg in SYS:
        rows = active_by_sys.get(key, [])
        legs = []
        sys_pnl = 0.0
        live_any = False
        for r in rows:
            entry = r.get("entry_price") or 0
            qty = r.get("qty") or 0
            mode = (r.get("mode") or "paper")
            if mode == "live":
                live_any = True
            _h = high_map.get(r.get("tradingsymbol"))
            mx = round(_h, 1) if _h is not None else None
            if r.get("status") == "ACTIVE":
                ltp = ltp_map.get(r.get("tradingsymbol"))
                pnl = (entry - ltp) * qty if ltp is not None else 0
                disp = round(ltp, 1) if ltp is not None else None
                arm = _arm(cfg, r, spot)
            else:
                # exited today — realized P&L, show the exit price and reason in place of live/arm
                xpx = r.get("exit_price") or 0
                pnl = (entry - xpx) * qty
                disp = round(xpx, 1)
                arm = "%s @%.0f" % (r.get("exit_reason") or "CLOSED", xpx)
            sys_pnl += pnl
            legs.append({
                "cp": r.get("instrument_type"),
                "strike": r.get("strike"),
                "qty": qty,
                "mode": mode,
                "entry": round(entry, 1),
                "ltp": disp,
                "max": mx,
                "arm": arm,
                "status": r.get("status"),
                "pnl": round(pnl),
            })
        # Header sums ALL systems (paper + live). Was live-only, which showed today=0 whenever
        # the whole book runs paper (e.g. master-mode=paper).
        day_pnl += sys_pnl
        systems.append({
            "key": key,
            "label": lab,
            "mode": "live" if live_any else ("paper" if legs else "flat"),
            "pnl": round(sys_pnl),
            "n_legs": len(legs),
            "legs": legs,
        })

    return {
        "generated_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "spot": round(spot, 1) if spot else None,
        "expiry": (datetime.now(IST).date() + timedelta(days=(3 - datetime.now(IST).date().weekday()) % 7)).strftime("%d %b current-week (Thu)"),
        "day_pnl": round(day_pnl),
        "systems": systems,
    }


def main():
    """One-shot: build the snapshot and write it, then exit. Re-run each minute by cron
    (the house pattern used by every other *_paper / publish_* writer here). This makes it
    immune to SSH-backgrounding quirks and self-healing — a crash just retries next minute."""
    tmp = OUT + ".tmp"
    data = build()
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, OUT)
    print("wrote %s @ %s (day %s)" % (OUT, data["generated_at"], data["day_pnl"]))


if __name__ == "__main__":
    main()
