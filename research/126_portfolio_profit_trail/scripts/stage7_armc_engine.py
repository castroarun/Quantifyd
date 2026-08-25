#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 7 - ARM C ENGINE: replay the THREE REAL 9:16 exit rules at every
strike offset, so the exit-rule x offset interaction is MEASURED, not assumed.

The three systems all sell the SAME straddle at the SAME 09:16 minute and differ ONLY in
how they exit (config.py NAS_ATM_DEFAULTS / NAS_ATM2_DEFAULTS / NAS_ATM4_DEFAULTS):

RECONCILIATION FIRST (v2, after v1 failed): a faithful replay of config.py's documented
per-leg 30% SL + trail-to-cost + re-enter produced -Rs437,588 on NIFTY against the live
book's +Rs164,988, because it cascaded 4.04 cycles/day against the live book's 1.04
trades/day. The reason is in the live trade table: the REAL 916_ATM exit reasons are
58 eod_squareoff, 10 ST_EXIT and **ZERO SL_HIT** - the documented per-leg 30% SL is
DORMANT in the live system (a SuperTrend trail exits first). A replay built on it is
therefore not the live book and its numbers were discarded.

So Arm C is measured on the constructions that DO reconcile:

  HOLD      09:16 -> 15:15, no stop. This is what the live 9:16 suite actually does on
            84% of days (58/71 eod_squareoff), so it is the best faithful proxy for the
            suite's offset behaviour.
  COMB      09:16 -> 15:20 with the live NAS_COMB20 per-DTE COMBINED SL (25/30/30/20).
            This is the r/116-validated shape and reconciles to r/122 to the rupee.
  RUPEE2500 09:16 -> 15:15 with ONLY the ATM2 rupee stop (Rs2,500/lot on combined MTM).
            >>> THE KEY INTERACTION, isolated: Rs2,500/lot = 38.46 premium points on NIFTY
            no matter what credit was sold. At ATM (credit ~200) that is ~19% of credit;
            four steps OTM (credit ~90) it is ~43%. The same rupee stop is a LARGER
            %-of-credit move off-ATM and must fire LATER. Measured, not assumed.

Offsets swept: 0, +-1, +-2, +-3, +-4 strike steps (NIFTY 50, SENSEX 100), applied to EVERY
entry including re-entries (the offset is a standing policy, not a one-off).

READ-ONLY on the DB. Writes results/armc_cells.csv (day x venue x system x offset).
"""
import sqlite3
import csv
import os
import time
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "stage7.log")

TODAY = date.today().isoformat()
FROZEN = {"2026-05-01", "2026-05-28", "2026-06-26"}
VEN = {
    "NIFTY": dict(lot=65, step=50, margin=165000.0),
    "SENSEX": dict(lot=20, step=100, margin=204000.0),
}
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
ENTRY_HM, EXIT_HM = "09:16", "15:15"
LEG_SL = 0.30
RUPEE_STOP_PER_LOT = 2500.0
MAX_REENTRY = 5
LOTS = 2
SLIP_TIME, SLIP_STOP = 0.178, 6.548


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def leg_cost(entry_px, exit_px, lot, nlots, forced):
    """Rs cost for ONE option leg round trip (sell then buy back), exact rate card."""
    sell, buy = entry_px * lot * nlots, exit_px * lot * nlots
    tot = sell + buy
    brok = 40.0
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = SLIP_STOP if forced else SLIP_TIME
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot * nlots


def rec_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? "
        "ORDER BY d", (sym,)) if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, sym, day, band):
    fexp = c.execute(
        "SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? AND snapshot_time>=? "
        "AND snapshot_time<? AND expiry_date>=?",
        (sym, day, day + "z", day)).fetchone()[0]
    if not fexp:
        return None
    per, spot, last = {}, {}, ""
    for st_, k, it, ltp, sp in c.execute(
            "SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
            "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? AND expiry_date=? "
            "AND ltp IS NOT NULL AND ltp>0", (sym, day, day + "z", fexp)):
        hm = st_[11:16]
        if st_ > last:
            last = st_
        if sp and hm not in spot:
            spot[hm] = sp
        if sp and abs(k - sp) > band:
            continue
        per.setdefault(hm, {}).setdefault(k, {})[it] = ltp
    if not per or not spot:
        return None
    if last[11:16] < "15:10":
        return ("PARTIAL", None, None)
    if len(set(spot.values())) < 50:
        return ("FROZEN", None, None)
    return fexp, per, spot


def dte_of(day, exp, days):
    if exp == day:
        return 0
    if exp in days and day in days:
        return days.index(exp) - days.index(day)
    n, cur, e = 0, date.fromisoformat(day), date.fromisoformat(exp)
    while cur < e:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5:
            n += 1
    return n


def px(per, hm, K, it):
    d = per.get(hm, {}).get(K)
    return d.get(it) if d else None


def open_straddle(per, spot, mins, i, step, off):
    hm = mins[i]
    sp = spot.get(hm)
    if not sp:
        return None
    K = round(sp / step) * step + off * step
    ce, pe = px(per, hm, K, "CE"), px(per, hm, K, "PE")
    if ce is None or pe is None or ce + pe <= 0:
        return None
    return K, ce, pe


COMB_SL = {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.20, 4: 0.30}


def run_system(sysname, per, spot, mins, step, lot, off, dte):
    """Replay ONE construction at ONE offset for one day. Single entry, no cascade -
    which is what the live books actually do. -> dict or None."""
    exit_hm = "15:20" if sysname == "COMB" else "15:15"
    i0 = next((i for i, m in enumerate(mins) if m >= ENTRY_HM), None)
    if i0 is None:
        return None
    op = open_straddle(per, spot, mins, i0, step, off)
    if not op:
        return None
    K, ce0, pe0 = op
    credit = ce0 + pe0
    if credit <= 0:
        return None
    rupee_stop_pts = RUPEE_STOP_PER_LOT / lot
    comb_thr = credit * (1 + COMB_SL.get(dte, 0.30))
    exit_comb, exit_hm_act, reason = credit, mins[i0], "TIME"
    peak = trough = 0.0
    stopped = False
    for hm in mins:
        if hm <= mins[i0] or hm > exit_hm:
            continue
        d = per.get(hm, {}).get(K)
        if not d or "CE" not in d or "PE" not in d:
            continue
        comb = d["CE"] + d["PE"]
        pnl = (credit - comb) * lot * LOTS
        peak = max(peak, pnl)
        trough = min(trough, pnl)
        exit_comb, exit_hm_act = comb, hm
        if sysname == "COMB" and comb >= comb_thr:
            stopped, reason = True, "SL"
            break
        if sysname == "RUPEE2500" and (comb - credit) >= rupee_stop_pts:
            stopped, reason = True, "SL"
            break
    if exit_hm_act == mins[i0]:
        return None
    gross = (credit - exit_comb) * lot * LOTS
    cost = 2 * leg_cost(credit / 2.0, exit_comb / 2.0, lot, LOTS, stopped)
    return dict(net=gross - cost, gross=gross, credit=credit,
                fires=1 if stopped else 0, cycles=1, peak=peak, trough=trough,
                roll=0, cost=cost)


def main():
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    f = open(os.path.join(RES, "armc_cells.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=[
        "day", "weekday", "venue", "dte", "expiry", "system", "offset", "credit",
        "net_rs", "gross_rs", "cost_rs", "fires", "cycles", "roll", "peak_rs", "trough_rs"])
    w.writeheader()
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    t0 = time.time()
    for sym, V in VEN.items():
        lot, step = V["lot"], V["step"]
        days = rec_days(c, sym)
        band = 14 * step
        kept = 0
        for day in days:
            if day == TODAY or day in FROZEN:
                continue
            d = load_day(c, sym, day, band)
            if not d or d[0] in ("PARTIAL", "FROZEN"):
                log("  %s %s SKIP %s" % (sym, day, d[0] if d else "nodata"))
                continue
            fexp, per, spot = d
            mins = sorted(per)
            if len(mins) < 200:
                log("  %s %s SKIP thin(%d)" % (sym, day, len(mins)))
                continue
            dte = dte_of(day, fexp, days)
            wd = WD[date.fromisoformat(day).weekday()]
            kept += 1
            n = 0
            for sysname in ("HOLD", "COMB", "RUPEE2500"):
                for off in OFFSETS:
                    r = run_system(sysname, per, spot, mins, step, lot, off, dte)
                    if not r:
                        continue
                    w.writerow(dict(
                        day=day, weekday=wd, venue=sym, dte=dte, expiry=fexp,
                        system=sysname, offset=off, credit=round(r["credit"], 2),
                        net_rs=round(r["net"]), gross_rs=round(r["gross"]),
                        cost_rs=round(r["cost"]), fires=r["fires"], cycles=r["cycles"],
                        roll=r["roll"], peak_rs=round(r["peak"]),
                        trough_rs=round(r["trough"])))
                    n += 1
            log("  %s %s %s dte=%d cells=%d [%.0fs]" % (sym, day, wd, dte, n, time.time() - t0))
            f.flush()
        log("%s kept %d days" % (sym, kept))
    f.close()
    log("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
