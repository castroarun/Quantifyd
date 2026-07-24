"""SL RE-ANCHOR PAPER-SHADOW (research/77) — zero-risk counterfactual tracker. [FIXED]

Takes the NAS 916 systems' ACTUAL traded legs and replays each leg's REAL intraday premium path
(options_data.db, matched by tradingsymbol) to compute what a "at 12:00, re-anchor the SL of any LIVE leg
to 30% above its then-current premium" rule WOULD have done vs what actually happened.
Places NO orders, touches NO live state.

CORRECTNESS (v2 — fixes v1 bugs):
  * each leg's path is restricted to [its own entry_time, its own actual exit_time] — never before entry,
    never after the leg really closed.
  * the re-anchor only applies to legs that are LIVE at 12:00 (entry < 12:00 < exit). Legs opened after
    noon (e.g. ATM2 move-stop re-centres) keep the plain 30% stop — they were never re-anchored.
  * if a leg shadow-stops, it books at that premium; otherwise it books its REAL exit.

Usage:  python3 scripts/sl_reanchor_shadow.py [YYYY-MM-DD]   |   --backfill
"""
import sqlite3, sys, os, datetime as dt
import numpy as np

ROOT = "/home/arun/quantifyd"
ODB = os.path.join(ROOT, "backtest_data", "options_data.db")
OUTCSV = os.path.join(ROOT, "research", "77_sl_tightening", "results", "shadow_log.csv")
SYSTEMS = [("nas_916_atm", "916-ATM"), ("nas_916_atm2", "916-ATM2"), ("nas_916_atm4", "916-ATM4")]
NOON = 12 * 3600


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def isosec(s):
    """entry_time/exit_time are ISO 'YYYY-MM-DDTHH:MM:SS...' or 'YYYY-MM-DD HH:MM:SS'."""
    try:
        return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])
    except Exception:
        return None


def leg_path(oc, tsym, day, t0, t1):
    rows = oc.execute("SELECT snapshot_time, ltp FROM option_chain WHERE tradingsymbol=? "
                      "AND date(snapshot_time)=? AND ltp>0 ORDER BY snapshot_time", (tsym, day)).fetchall()
    if not rows:
        return None
    t = np.array([tsec(r[0]) for r in rows]); p = np.array([float(r[1]) for r in rows])
    m = (t >= t0) & (t <= t1)                      # <-- only the leg's own live window
    if m.sum() < 3:
        return None
    return t[m], p[m]


def shadow_leg(times, prems, entry, t0, t1, real_exit):
    """Return (shadow_exit_price, stopped, reanchored)."""
    cap = entry * 1.3
    live_at_noon = (t0 < NOON < t1)
    sl_after = None
    if live_at_noon:
        i = int(np.searchsorted(times, NOON)); i = min(max(i, 0), len(times) - 1)
        sl_after = min(cap, float(prems[i]) * 1.3)
    for j in range(len(prems)):
        sl = cap if (not live_at_noon or times[j] < NOON) else sl_after
        if prems[j] >= sl:
            return float(prems[j]), True, bool(live_at_noon)
    return real_exit, False, bool(live_at_noon)


def run_day(day):
    oc = sqlite3.connect(ODB)
    out = []
    for db, lab in SYSTEMS:
        p = os.path.join(ROOT, "backtest_data", f"{db}_trading.db")
        if not os.path.exists(p):
            continue
        c = sqlite3.connect(p); c.row_factory = sqlite3.Row
        legs = c.execute("SELECT tradingsymbol, qty, entry_price, exit_price, entry_time, exit_time, exit_reason "
                         "FROM nas_atm_positions WHERE date(entry_time)=? AND transaction_type='SELL' "
                         "AND exit_price IS NOT NULL AND exit_time IS NOT NULL", (day,)).fetchall()
        if not legs:
            continue
        act = shd = 0.0; nstop = 0; matched = 0; nre = 0
        for L in legs:
            e, x, q = L["entry_price"], L["exit_price"], L["qty"]
            act += (e - x) * q
            t0, t1 = isosec(L["entry_time"]), isosec(L["exit_time"])
            if t0 is None or t1 is None or t1 <= t0:
                shd += (e - x) * q; continue
            path = leg_path(oc, L["tradingsymbol"], day, t0, t1)
            if path is None:
                shd += (e - x) * q; continue      # no chain data -> assume unchanged
            matched += 1
            sx, stopped, reanch = shadow_leg(path[0], path[1], e, t0, t1, x)
            shd += (e - sx) * q
            nstop += 1 if stopped else 0
            nre += 1 if reanch else 0
        out.append(dict(day=day, system=lab, legs=len(legs), matched=matched, reanchored=nre,
                        actual=round(act), shadow=round(shd), diff=round(shd - act), shadow_stops=nstop))
    return out


def append(rows):
    new = not os.path.exists(OUTCSV)
    with open(OUTCSV, "a") as f:
        if new:
            f.write("day,system,legs,matched,reanchored,actual,shadow,diff,shadow_stops\n")
        for r in rows:
            f.write("%s,%s,%d,%d,%d,%d,%d,%d,%d\n" % (r["day"], r["system"], r["legs"], r["matched"],
                    r["reanchored"], r["actual"], r["shadow"], r["diff"], r["shadow_stops"]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--backfill":
        oc = sqlite3.connect(ODB)
        days = [r[0] for r in oc.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain ORDER BY 1")]
        if os.path.exists(OUTCSV):
            os.remove(OUTCSV)
        allr = []
        for d in days:
            rr = run_day(d)
            if rr:
                append(rr); allr += rr
        ta = sum(r["actual"] for r in allr); ts = sum(r["shadow"] for r in allr)
        nd = len({r["day"] for r in allr})
        print("BACKFILL: %d rows over %d days (legs matched to chain: %d, re-anchored: %d)" % (
            len(allr), nd, sum(r["matched"] for r in allr), sum(r["reanchored"] for r in allr)))
        print("  ACTUAL (30%% SL)          = Rs %+d" % ta)
        print("  SHADOW (re-anchor 12:00) = Rs %+d" % ts)
        print("  DIFF                     = Rs %+d   (%+d / day)" % (ts - ta, (ts - ta) / max(nd, 1)))
        for _, lab in SYSTEMS:
            rs = [r for r in allr if r["system"] == lab]
            if rs:
                print("   %-9s actual %+7d  shadow %+7d  diff %+7d  (days %d, re-anch %d, shadow-stops %d)" % (
                    lab, sum(r["actual"] for r in rs), sum(r["shadow"] for r in rs), sum(r["diff"] for r in rs),
                    len(rs), sum(r["reanchored"] for r in rs), sum(r["shadow_stops"] for r in rs)))
        dd = sorted(allr, key=lambda r: r["diff"])
        print("  worst 3:", [(r["day"], r["system"], r["diff"]) for r in dd[:3]])
        print("  best  3:", [(r["day"], r["system"], r["diff"]) for r in dd[-3:]])
    else:
        day = args[0] if args else dt.date.today().isoformat()
        rows = run_day(day)
        if not rows:
            print("no legs for", day); sys.exit(0)
        append(rows)
        for r in rows:
            print("%s %-9s actual %+6d  shadow %+6d  diff %+6d  (legs %d, matched %d, re-anch %d, stops %d)" % (
                r["day"], r["system"], r["actual"], r["shadow"], r["diff"], r["legs"], r["matched"],
                r["reanchored"], r["shadow_stops"]))
