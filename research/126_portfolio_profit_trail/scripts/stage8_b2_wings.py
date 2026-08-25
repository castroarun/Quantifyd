#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/126 Stage 8 - ARM B2: PROFIT-TRIGGERED PORTFOLIO WINGS.

Arun's actual question: "buying wings only after we achieve a profit level at the
PORTFOLIO in order to lock it, not wings from the beginning."

Mechanism under test: when the book is UP it is up BECAUSE premium has decayed, so at the
moment protection is wanted the wings are at their cheapest of the day AND are funded out
of profit already earned. Unlike the trail, a wing CAPS the tail without surrendering the
remaining theta and without paying the measured +6.548 pt/leg-side forced-exit slippage.
Counter-hypothesis to test honestly: wings bought when the book is up are far from the
money (the market has not moved), so they are cheap but rarely pay.

Scope fix (2026-08-25): the live book now includes CSL_TIMEB2_NIFTY - 8 REAL lots
(qty 520), 13:15->14:30, combined-SL 30% - run by a standalone one-shot that never wrote
a REAL record into the daemon's state, so every book-list-derived harness missed it.

ONE chain pass per day per venue does everything: rebuilds the sleeves, builds the
portfolio path (with the live 9:16 suite MTM rescaled to the deployed 2 lots/system),
finds each trigger's arm minute, then prices the wings at ASK and unwinds at BID.

READ-ONLY. Writes results/b2_cells.csv, results/b2_worked_0825.txt
"""
import sqlite3
import csv
import os
import json
import time
from collections import defaultdict
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
CFG = Q + "backtest_data/csl_paper_config.json"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)
LOG = os.path.join(RES, "stage8.log")

TODAY = date.today().isoformat()
FROZEN = {"2026-05-01", "2026-05-28", "2026-06-26"}
VEN = {"NIFTY": dict(lot=65, step=50, dists=[100, 150, 200, 250, 300, 400, 500]),
       "SENSEX": dict(lot=20, step=100, dists=[400, 600, 800, 1000, 1200, 1600, 2000])}
SLIP_TIME, SLIP_STOP = 0.178, 6.548
SUITE_DBS = [("916_ATM", "nas_916_atm_trading.db"),
             ("916_ATM2", "nas_916_atm2_trading.db"),
             ("916_ATM4", "nas_916_atm4_trading.db")]
SUITE_LOTS_NOW = 2
BOOK_END = "15:20"

TRIG_ABS = [5000, 8000, 10000, 12000, 15000, 20000]
TRIG_PCT = [0.20, 0.30, 0.40]          # of total credit collected (Rs)
TRIG_TIME = [("T1300_10000", "13:00", 10000), ("T1300_5000", "13:00", 5000),
             ("T1400_8000", "14:00", 8000)]
COVERAGE = ["ALL", "BIGGEST", "ADVERSE"]
UNWIND = ["EOD", "RECOVER"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
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


def charges_long(buyp, sellp, lot, nlots):
    buy, sell = buyp * lot * nlots, sellp * lot * nlots
    tot = buy + sell
    brok, stt = 80.0, 0.001 * sell
    txn, ipft, sebi = 0.0003503 * tot, 0.0000050 * tot, 0.0000010 * tot
    stamp = 0.00003 * buy
    return brok + stt + txn + ipft + sebi + stamp + 0.18 * (brok + txn + ipft + sebi)


def sleeves_spec():
    """LIVE-money books. TIMEB2 is added from its standalone one-shot (NOT in the daemon)."""
    cfg = json.load(open(CFG))["books"]
    spec = [("TB_NIFTY", "NIFTY", "CSL_TIMEB_NIFTY", 8),
            ("COMB20", "NIFTY", "NAS_COMB20", 2),
            ("TB_SENSEX", "SENSEX", "CSL_TIMEB_SENSEX", 8),
            ("SXWED", "SENSEX", "CSL30F_SENSEX_WED", 3)]
    out = []
    for nm, ven, bk, dl in spec:
        cells = {}
        for dte, c in (cfg.get(bk) or {}).items():
            sl = c["sl"]
            cells[int(dte)] = (c["entry"], c["exit"],
                               None if sl == "none" else float(sl) / 100.0,
                               int(c.get("lots", dl)))
        if cells:
            out.append((nm, ven, cells))
    # TIMEB2: 8 real lots, 13:15->14:30, combined-SL 30%, every DTE it is armed for.
    out.append(("TIMEB2", "NIFTY", {d: ("13:15", "14:30", 0.30, 8) for d in range(5)}))
    return out


BACKSTOP = 0.50


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
    for st_, k, it, ltp, bid, ask, vol in c.execute(
            "SELECT snapshot_time,strike,instrument_type,ltp,bid,ask,volume FROM option_chain "
            "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? AND expiry_date=? "
            "AND ltp IS NOT NULL", (sym, day, day + "z", fexp)):
        hm = st_[11:16]
        if st_ > last:
            last = st_
        per.setdefault(hm, {}).setdefault(k, {})[it] = (ltp, bid, ask, vol)
    for st_, sp in c.execute(
            "SELECT snapshot_time,underlying_spot FROM option_chain WHERE symbol=? "
            "AND snapshot_time>=? AND snapshot_time<? AND underlying_spot IS NOT NULL",
            (sym, day, day + "z")):
        hm = st_[11:16]
        if hm not in spot:
            spot[hm] = sp
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


def replay_sleeve(per, spot, mins, step, lot, e_hm, x_hm, slp, lots):
    """-> (path {hm:(pnl, open)}, net, credit, entry_hm)"""
    cand = [m for m in mins if m >= e_hm]
    if not cand:
        return None
    m0 = cand[0]
    sp0 = spot.get(m0)
    if not sp0:
        return None
    K = round(sp0 / step) * step
    d0 = per.get(m0, {}).get(K)
    if not d0 or "CE" not in d0 or "PE" not in d0:
        return None
    credit = d0["CE"][0] + d0["PE"][0]
    if credit <= 0:
        return None
    thr = credit * (1 + (slp if slp is not None else BACKSTOP))
    path, exit_comb, reason, stopped = {}, credit, "TIME", False
    for hm in mins:
        if hm < m0 or hm > x_hm:
            continue
        dd = per.get(hm, {}).get(K)
        if not dd or "CE" not in dd or "PE" not in dd:
            continue
        comb = dd["CE"][0] + dd["PE"][0]
        if stopped:
            path[hm] = ((credit - exit_comb) * lot * lots, False)
            continue
        path[hm] = ((credit - comb) * lot * lots, True)
        exit_comb = comb
        if comb >= thr:
            stopped, reason = True, "SL"
    if len(path) < 5:
        return None
    gross = (credit - exit_comb) * lot * lots
    net = gross - cost_short(credit, exit_comb, lot, lots, reason == "SL")
    return path, net, credit, m0, lots


def load_suite():
    mtm, openlots = defaultdict(dict), defaultdict(lambda: defaultdict(int))
    for nm, db in SUITE_DBS:
        c = sqlite3.connect("file:%sbacktest_data/%s?mode=ro" % (Q, db), uri=True)
        lots_by_day, win = {}, defaultdict(list)
        for d, et, xt, lots in c.execute(
                "SELECT trade_date,entry_time,exit_time,lots FROM nas_atm_trades"):
            if not d or not et or not lots:
                continue
            lots_by_day[d] = max(lots_by_day.get(d, 0), int(lots))
            win[d].append((et[11:16], xt[11:16] if xt else "15:30"))
        for d, ts, dp in c.execute("SELECT snap_date,ts,day_pnl FROM nas_mtm_snapshots"):
            if dp is None:
                continue
            t = ts[11:16] if "T" in ts else ts[:5]
            L = lots_by_day.get(d)
            mtm[d][t] = mtm[d].get(t, 0.0) + float(dp) * (float(SUITE_LOTS_NOW) / L if L else 1.0)
        for d, ws in win.items():
            for h in range(9, 16):
                for m in range(60):
                    t = "%02d:%02d" % (h, m)
                    if any(a <= t < b for a, b in ws):
                        openlots[d][t] += SUITE_LOTS_NOW
        c.close()
    return mtm, openlots


def wing_quote(per, hm, K, kind):
    d = per.get(hm, {}).get(K)
    if not d or kind not in d:
        return None
    ltp, bid, ask, vol = d[kind]
    return ltp, (bid or 0.0), (ask or 0.0), (vol or 0)


def main():
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    SL = sleeves_spec()
    smtm, solots = load_suite()
    daysby = {v: rec_days(c, v) for v in VEN}
    alldays = sorted(set(daysby["NIFTY"]) | set(daysby["SENSEX"]))
    alldays = [d for d in alldays if d != TODAY and d not in FROZEN]

    f = open(os.path.join(RES, "b2_cells.csv"), "w", newline="")
    w = csv.DictWriter(f, fieldnames=[
        "day", "weekday", "trigger", "dist_n", "coverage", "unwind", "armed", "arm_hm",
        "base_net", "wing_pnl", "hedged_net", "wing_ask_rs", "wing_sold_rs",
        "peak_mtm", "final_mtm", "n_pairs", "skip_vol"])
    w.writeheader()
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    t0 = time.time()
    worked = []

    for day in alldays:
        wd = WD[date.fromisoformat(day).weekday()]
        chains, spots, dtes = {}, {}, {}
        for ven in VEN:
            if day not in daysby[ven]:
                continue
            d = load_day(c, ven, day, max(VEN[ven]["dists"]) + 4 * VEN[ven]["step"])
            if not d or d[0] in ("PARTIAL", "FROZEN"):
                continue
            fexp, per, spot = d
            chains[ven], spots[ven] = per, spot
            dtes[ven] = dte_of(day, fexp, daysby[ven])
        if not chains:
            log("  %s SKIP no chain" % day)
            continue

        # ---- sleeves
        sl_paths, sl_net, sl_credit, sl_ven, sl_lots = {}, {}, {}, {}, {}
        for nm, ven, cells in SL:
            if ven not in chains:
                continue
            dte = dtes[ven]
            if dte not in cells:
                continue
            e_hm, x_hm, slp, lots = cells[dte]
            mins = sorted(chains[ven])
            r = replay_sleeve(chains[ven], spots[ven], mins, VEN[ven]["step"],
                              VEN[ven]["lot"], e_hm, x_hm, slp, lots)
            if not r:
                continue
            path, net, credit, m0, lts = r
            sl_paths[nm], sl_net[nm], sl_credit[nm] = path, net, credit
            sl_ven[nm], sl_lots[nm] = ven, lts

        has_suite = day in smtm and any(v != 0 for v in smtm[day].values())
        if not sl_paths and not has_suite:
            continue

        allmins = sorted({m for p in sl_paths.values() for m in p} | set(smtm.get(day, {})))
        allmins = [m for m in allmins if "09:15" <= m <= BOOK_END]
        if len(allmins) < 20:
            continue
        sm, last = {}, 0.0
        for m in allmins:
            if m in smtm.get(day, {}):
                last = smtm[day][m]
            sm[m] = last
        sp_ff, sop = {}, {}
        for nm, path in sl_paths.items():
            cur, seen, mp, op = 0.0, False, {}, {}
            for m in allmins:
                v = path.get(m)
                if v is not None:
                    cur, seen = v[0], True
                    op[m] = v[1]
                else:
                    op[m] = False
                mp[m] = cur if seen else 0.0
            sp_ff[nm], sop[nm] = mp, op
        curve = {m: sm[m] + sum(sp_ff[nm][m] for nm in sl_paths) for m in allmins}
        base_net = (sm[allmins[-1]] if allmins else 0.0) + sum(sl_net.values())
        peak_mtm = max(curve[m] for m in allmins)
        final_mtm = curve[allmins[-1]]
        total_credit_rs = sum(sl_credit[nm] * VEN[sl_ven[nm]]["lot"] * sl_lots[nm]
                              for nm in sl_credit)

        # ---- open short-lots per venue per minute (for wing sizing)
        openlots = {v: {} for v in VEN}
        for m in allmins:
            for v in VEN:
                n = 0
                if v == "NIFTY":
                    n += solots.get(day, {}).get(m, 0)
                for nm in sl_paths:
                    if sl_ven[nm] == v and sop[nm].get(m):
                        n += sl_lots[nm]
                openlots[v][m] = n

        # ---- trigger definitions
        trigs = []
        for a in TRIG_ABS:
            trigs.append(("ABS_%d" % a, None, a))
        for p in TRIG_PCT:
            trigs.append(("PCT_%d" % int(p * 100), None, p * total_credit_rs))
        for lbl, tm, a in TRIG_TIME:
            trigs.append((lbl, tm, a))

        wingmemo = {}

        def wing_leg(v, dist_i, arm, unw, thr):
            """price ONE venue's wing pair bought at ASK at `arm`, unwound at BID.
            Memoised: cov/unwind combinations reuse the same quotes."""
            key = (v, dist_i, arm, unw)
            if key in wingmemo:
                return wingmemo[key]
            V = VEN[v]
            dist = V["dists"][dist_i]
            per, spot = chains[v], spots[v]
            res = None
            sp = spot.get(arm)
            nlots = openlots[v].get(arm, 0)
            if sp and nlots > 0:
                atm = round(sp / V["step"]) * V["step"]
                kc, kp = atm + dist, atm - dist
                qc = wing_quote(per, arm, kc, "CE")
                qp = wing_quote(per, arm, kp, "PE")
                if qc and qp and qc[2] > 0 and qp[2] > 0:
                    vmins = [m for m in sorted(per) if arm <= m <= BOOK_END]
                    maxvol = 0
                    for m in vmins:
                        a1 = wing_quote(per, m, kc, "CE")
                        b1 = wing_quote(per, m, kp, "PE")
                        if a1 and a1[3] > maxvol:
                            maxvol = a1[3]
                        if b1 and b1[3] > maxvol:
                            maxvol = b1[3]
                    if maxvol <= 0:
                        res = ("SKIPVOL",)
                    else:
                        buy = qc[2] + qp[2]
                        sell_m = vmins[-1] if vmins else arm
                        if unw == "RECOVER":
                            for m in vmins:
                                if m > arm and curve.get(m, 0) >= thr:
                                    sell_m = m
                                    break
                        sc = wing_quote(per, sell_m, kc, "CE")
                        spq = wing_quote(per, sell_m, kp, "PE")
                        sell = (sc[1] if sc else 0.0) + (spq[1] if spq else 0.0)
                        pnl = (sell - buy) * V["lot"] * nlots - charges_long(
                            buy, sell, V["lot"], nlots)
                        res = (pnl, buy * V["lot"] * nlots, sell * V["lot"] * nlots,
                               buy, sell, nlots, atm, dist)
            wingmemo[key] = res
            return res

        for tname, tmin, thr in trigs:
            arm = None
            for m in allmins:
                if tmin and m < tmin:
                    continue
                if curve[m] >= thr:
                    arm = m
                    break
            for dist_i in range(len(VEN["NIFTY"]["dists"])):
                dn = VEN["NIFTY"]["dists"][dist_i]
                for cov in COVERAGE:
                    for unw in UNWIND:
                        if arm is None:
                            w.writerow(dict(
                                day=day, weekday=wd, trigger=tname, dist_n=dn,
                                coverage=cov, unwind=unw, armed=0, arm_hm="",
                                base_net=round(base_net), wing_pnl=0,
                                hedged_net=round(base_net), wing_ask_rs=0,
                                wing_sold_rs=0, peak_mtm=round(peak_mtm),
                                final_mtm=round(final_mtm), n_pairs=0, skip_vol=0))
                            continue
                        vens = [v for v in chains if openlots[v].get(arm, 0) > 0]
                        if cov == "BIGGEST" and vens:
                            vens = [max(vens, key=lambda v: openlots[v][arm])]
                        elif cov == "ADVERSE":
                            adv = []
                            for v in vens:
                                vp = (sm[arm] if v == "NIFTY" else 0.0) + sum(
                                    sp_ff[nm][arm] for nm in sl_paths if sl_ven[nm] == v)
                                if vp < 0:
                                    adv.append(v)
                            vens = adv
                        wing_pnl, ask_rs, sold_rs, npairs, skipv = 0.0, 0.0, 0.0, 0, 0
                        for v in vens:
                            r = wing_leg(v, dist_i, arm, unw, thr)
                            if r is None:
                                continue
                            if r[0] == "SKIPVOL":
                                skipv += 1
                                continue
                            wing_pnl += r[0]
                            ask_rs += r[1]
                            sold_rs += r[2]
                            npairs += 1
                        w.writerow(dict(
                            day=day, weekday=wd, trigger=tname, dist_n=dn, coverage=cov,
                            unwind=unw, armed=1, arm_hm=arm, base_net=round(base_net),
                            wing_pnl=round(wing_pnl),
                            hedged_net=round(base_net + wing_pnl),
                            wing_ask_rs=round(ask_rs), wing_sold_rs=round(sold_rs),
                            peak_mtm=round(peak_mtm), final_mtm=round(final_mtm),
                            n_pairs=npairs, skip_vol=skipv))
        log("  %s %s books=%s peak=%d base=%d [%.0fs]" % (
            day, wd, ",".join(sorted(sl_paths)) + ("+SUITE" if has_suite else ""),
            peak_mtm, base_net, time.time() - t0))
        f.flush()
    f.close()
    if worked:
        with open(os.path.join(RES, "b2_worked_0825.txt"), "w") as g:
            g.write("2026-08-25 wing purchases by cell\n")
            for r in worked:
                g.write("%s d=%s %s %s %s arm=%s atm=%d dist=%d ask=%.2f bid_out=%.2f lots=%d pnl=%.0f\n" % r)
    log("DONE %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
