#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 4 - ARM C: strike / entry-time diversification (the FREE defence).

CONFIRMED BY CONFIG, not just by one day: NAS_916_ATM / ATM2 / ATM4 all sell the SAME
ATM straddle, same venue, same expiry, at the SAME 09:16 minute. They differ ONLY in
exit machinery (ATM2 = Rs2,500/lot rupee stop one-and-done; ATM4 = max_rolls 1). So the
9:16 "suite" is one position at 6 lots with three exit rules, not three systems.

This stage asks whether spreading the SAME notional across STRIKES or ENTRY MINUTES
reduces the joint tail. It costs no premium and no firing cost, so if it works it is
the cheapest defence available.

Method: replay the COMB-shape construction (09:16 -> 15:20, per-DTE combined SL from
the live NAS_COMB20 config) at strike offsets -2..+2 steps and entry minutes
09:16/09:31/09:46/10:01, on the real 1-minute chain. Then build equal-notional
3-clone portfolios and compare the tail.

READ-ONLY. Writes results/diversify_cells.csv
"""
import sqlite3, csv, os, time
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
LOG = os.path.join(RES, "stage4.log")
TODAY = date.today().isoformat()
FROZEN = {"2026-05-01", "2026-05-28", "2026-06-26"}
VEN = {"NIFTY": dict(lot=65, step=50), "SENSEX": dict(lot=20, step=100)}
SLIP_TIME, SLIP_STOP = 0.178, 6.548
# per-DTE combined SL of the live NAS_COMB20 book
COMB_SL = {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.20, 4: 0.30}
OFFSETS = [-2, -1, 0, 1, 2]
ENTRIES = ["09:16", "09:31", "09:46", "10:01"]
EXIT = "15:20"
LOTS = 2


def log(m):
    with open(LOG, "a") as f: f.write(m + "\n")
    print(m, flush=True)


def cost_short(credit, exitp, lot, nlots, forced):
    sell, buy = credit * lot * nlots, exitp * lot * nlots
    tot = sell + buy
    brok, stt = 80.0, 0.001 * sell
    txn, ipft, sebi = 0.0003503 * tot, 0.0000050 * tot, 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * (SLIP_STOP if forced else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot * nlots


def rec_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? "
        "ORDER BY d", (sym,)) if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, sym, day, band):
    fexp = c.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? "
                     "AND snapshot_time>=? AND snapshot_time<? AND expiry_date>=?",
                     (sym, day, day + "z", day)).fetchone()[0]
    if not fexp: return None
    per, spot, last = {}, {}, ""
    for st_, k, it, ltp, sp in c.execute(
            "SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
            "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? AND expiry_date=? "
            "AND ltp IS NOT NULL", (sym, day, day + "z", fexp)):
        hm = st_[11:16]
        if st_ > last: last = st_
        if sp and hm not in spot: spot[hm] = sp
        if sp and abs(k - sp) > band: continue
        per.setdefault(hm, {}).setdefault(k, {})[it] = ltp
    if not per or not spot or last[11:16] < "15:15": return None
    if len(set(spot.values())) < 50: return None
    return fexp, per, spot


def dte_of(day, exp, days):
    if exp == day: return 0
    if exp in days and day in days: return days.index(exp) - days.index(day)
    n, cur, e = 0, date.fromisoformat(day), date.fromisoformat(exp)
    while cur < e:
        cur = date.fromordinal(cur.toordinal() + 1)
        if cur.weekday() < 5: n += 1
    return n


def main():
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    f = open(os.path.join(RES, "diversify_cells.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=["day", "weekday", "venue", "dte", "expiry", "entry",
                                      "offset", "strike", "credit", "exit_hm", "exit_comb",
                                      "reason", "gross_rs", "net_rs", "peak_rs", "trough_rs"])
    w.writeheader()
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    t0 = time.time()
    for sym, V in VEN.items():
        lot, step = V["lot"], V["step"]
        days = rec_days(c, sym)
        for day in days:
            if day == TODAY or day in FROZEN: continue
            d = load_day(c, sym, day, 4 * step)
            if not d:
                log("  %s %s SKIP" % (sym, day)); continue
            fexp, per, spot = d
            dte = dte_of(day, fexp, days)
            sl = COMB_SL.get(dte, 0.30)
            mins = sorted(per)
            wd = WD[date.fromisoformat(day).weekday()]
            n = 0
            for e_hm in ENTRIES:
                cand = [m for m in mins if m >= e_hm]
                if not cand: continue
                m0 = cand[0]
                sp0 = spot.get(m0)
                if not sp0: continue
                atm = round(sp0 / step) * step
                for off in OFFSETS:
                    K = atm + off * step
                    d0 = per.get(m0, {}).get(K)
                    if not d0 or "CE" not in d0 or "PE" not in d0: continue
                    credit = d0["CE"] + d0["PE"]
                    if credit <= 0: continue
                    thr = credit * (1 + sl)
                    exit_hm, exit_comb, reason = m0, credit, "TIME"
                    peak = trough = 0.0
                    for hm in mins:
                        if hm < m0 or hm > EXIT: continue
                        dd = per[hm].get(K)
                        if not dd or "CE" not in dd or "PE" not in dd: continue
                        comb = dd["CE"] + dd["PE"]
                        pnl = (credit - comb) * lot * LOTS
                        peak = max(peak, pnl); trough = min(trough, pnl)
                        exit_hm, exit_comb = hm, comb
                        if comb >= thr:
                            reason = "SL"; break
                    if exit_hm == m0: continue
                    gross = (credit - exit_comb) * lot * LOTS
                    cst = cost_short(credit, exit_comb, lot, LOTS, reason == "SL")
                    w.writerow(dict(day=day, weekday=wd, venue=sym, dte=dte, expiry=fexp,
                                    entry=e_hm, offset=off, strike=K,
                                    credit=round(credit, 2), exit_hm=exit_hm,
                                    exit_comb=round(exit_comb, 2), reason=reason,
                                    gross_rs=round(gross), net_rs=round(gross - cst),
                                    peak_rs=round(peak), trough_rs=round(trough)))
                    n += 1
            log("  %s %s %s dte=%d cells=%d [%.0fs]" % (sym, day, wd, dte, n, time.time() - t0))
            f.flush()
    f.close()
    log("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
