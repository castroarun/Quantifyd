#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/104 — ATM straddle sell-timing sweep on REAL 1-min option_chain (SENSEX+NIFTY, 75 days).

For each (symbol, day): SELL the running-ATM straddle (front weekly expiry) at each candidate entry
time, SQUARE at 15:25 (buy back the SAME strike) = MIS same-day, matching the NAS book. Also a
clock-free SIGNAL entry (enter when premium rolls over from its intraday high). Aggregate net P&L
per lot by (symbol, entry, DTE). See STATUS-MD sections 1-4. Real data -> G2.
"""
import sqlite3, csv, os
from collections import defaultdict
from datetime import datetime

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
DETAIL = os.path.join(HERE, "..", "results", "selltime_detail.csv")
SUMMARY = os.path.join(HERE, "..", "results", "selltime_summary.csv")
LOG = os.path.join(HERE, "..", "results", "run.log")
LOTS = {"SENSEX": 20, "NIFTY": 65}
BROKERAGE = 160.0
SLIP_PT = 0.5                       # per leg
ENTRY_TIMES = ["09:20","09:45","10:15","10:45","11:15","11:45","12:15","12:45","13:15","13:45"]
EXIT_HM = "15:25"
SIG_START = "09:45"
SIG_DROP = 0.92                    # enter when prem <= 0.92 x running high

def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)

def day_snaps(c, sym, day):
    """{ 'HH:MM': (spot, front_expiry, {strike:{CE:ltp,PE:ltp}}) } for the front weekly expiry."""
    rows = c.execute(
        "SELECT snapshot_time,expiry_date,strike,instrument_type,ltp,underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time LIKE ? AND ltp IS NOT NULL",
        (sym, day + "%")).fetchall()
    if not rows:
        return {}
    # front expiry = smallest expiry_date >= day (weekly)
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return {}
    fexp = exps[0]
    snaps = {}
    for st, e, k, it, ltp, sp in rows:
        if e != fexp:
            continue
        hm = st[11:16]
        d = snaps.setdefault(hm, [None, fexp, {}])
        if sp:
            d[0] = sp
        d[2].setdefault(k, {})[it] = ltp
    # keep only minutes with a spot
    return {hm: tuple(v) for hm, v in snaps.items() if v[0]}

def atm(snap):
    """(strike, straddle_prem) for the strike nearest spot that has both legs; else None."""
    spot, _, ks = snap
    cand = [k for k, v in ks.items() if "CE" in v and "PE" in v]
    if not cand:
        return None
    k = min(cand, key=lambda x: abs(x - spot))
    return k, ks[k]["CE"] + ks[k]["PE"]

def strike_prem(snap, k):
    ks = snap[2]
    v = ks.get(k) or ks.get(float(k))
    if v and "CE" in v and "PE" in v:
        return v["CE"] + v["PE"]
    return None

def main():
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    done = set()
    if os.path.exists(DETAIL):
        with open(DETAIL) as f:
            for r in csv.DictReader(f):
                done.add((r["symbol"], r["day"], r["entry_label"]))
    new = not os.path.exists(DETAIL)
    fd = open(DETAIL, "a", newline="")
    dw = csv.DictWriter(fd, fieldnames=["symbol","day","dte","entry_label","entry_time","net_per_lot"])
    if new:
        dw.writeheader()
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    for sym in ("SENSEX", "NIFTY"):
        lot = LOTS[sym]
        cost = BROKERAGE + SLIP_PT * 2 * lot          # 2 legs slippage per round-trip
        days = [r[0] for r in c.execute(
            "SELECT DISTINCT substr(snapshot_time,1,10) d FROM option_chain WHERE symbol=? ORDER BY d",
            (sym,))]
        log("[%s] %d days, lot=%d, cost/lot=%.0f" % (sym, len(days), lot, cost))
        for di, day in enumerate(days):
            snaps = day_snaps(c, sym, day)
            if not snaps:
                continue
            exit_times = sorted(t for t in snaps if t >= EXIT_HM)
            if not exit_times:
                continue
            exit_snap = snaps[exit_times[0]]
            fexp = exit_snap[1]
            dte = (datetime.strptime(fexp, "%Y-%m-%d").date() - datetime.strptime(day, "%Y-%m-%d").date()).days
            # fixed-clock entries
            for lbl in ENTRY_TIMES:
                if (sym, day, lbl) in done:
                    continue
                cand_t = sorted(t for t in snaps if t >= lbl and t < EXIT_HM)
                if not cand_t:
                    continue
                esnap = snaps[cand_t[0]]
                a = atm(esnap)
                if not a:
                    continue
                k, prem_e = a
                prem_x = strike_prem(exit_snap, k)
                if prem_x is None:
                    continue
                net = (prem_e - prem_x) * lot - cost
                dw.writerow(dict(symbol=sym, day=day, dte=dte, entry_label=lbl,
                                 entry_time=cand_t[0], net_per_lot=round(net, 1)))
            # SIGNAL entry: first minute after SIG_START where ATM prem <= SIG_DROP x running high
            if (sym, day, "SIGNAL") not in done:
                mins = sorted(t for t in snaps if SIG_START <= t < EXIT_HM)
                runhi = 0.0; picked = None
                for t in mins:
                    a = atm(snaps[t])
                    if not a:
                        continue
                    _, pr = a
                    runhi = max(runhi, pr)
                    if runhi > 0 and pr <= SIG_DROP * runhi:
                        picked = (t, a[0], pr); break
                if picked:
                    t, k, prem_e = picked
                    prem_x = strike_prem(exit_snap, k)
                    if prem_x is not None:
                        net = (prem_e - prem_x) * lot - cost
                        dw.writerow(dict(symbol=sym, day=day, dte=dte, entry_label="SIGNAL",
                                         entry_time=t, net_per_lot=round(net, 1)))
            if di % 15 == 0:
                fd.flush(); log("  [%s] %d/%d days (%s)" % (sym, di + 1, len(days), day))
        fd.flush()
    c.close(); fd.close()
    # aggregate
    agg = defaultdict(list)
    with open(DETAIL) as f:
        for r in csv.DictReader(f):
            dte_b = "0" if r["dte"] == "0" else ("1" if r["dte"] == "1" else "2plus")
            agg[(r["symbol"], r["entry_label"], dte_b)].append(float(r["net_per_lot"]))
    def pctile(v, p):
        s = sorted(v); return s[min(len(s)-1, int(p*len(s)))] if s else 0
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol","entry","dte_bucket","n","mean_net_lot",
                                          "median_net_lot","win_pct","p05_tail"])
        w.writeheader()
        for key in sorted(agg):
            v = agg[key]
            w.writerow(dict(symbol=key[0], entry=key[1], dte_bucket=key[2], n=len(v),
                            mean_net_lot=round(sum(v)/len(v), 0),
                            median_net_lot=round(pctile(v, 0.5), 0),
                            win_pct=round(100*sum(1 for x in v if x > 0)/len(v), 0),
                            p05_tail=round(pctile(v, 0.05), 0)))
    log("wrote " + SUMMARY)
    # console: DTE0 by entry, both symbols
    for sym in ("SENSEX", "NIFTY"):
        log("\n=== %s DTE0 — net/lot by entry time ===" % sym)
        for lbl in ENTRY_TIMES + ["SIGNAL"]:
            v = agg.get((sym, lbl, "0"))
            if v:
                log("  %-8s mean=%+7.0f  median=%+7.0f  win%%=%3.0f  p05=%+7.0f  n=%d" % (
                    lbl, sum(v)/len(v), pctile(v, 0.5),
                    100*sum(1 for x in v if x > 0)/len(v), pctile(v, 0.05), len(v)))

if __name__ == "__main__":
    main()
