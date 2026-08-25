#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 2 - ARM A: sweep the PORTFOLIO profit trail over the live book.

Portfolio path per day = live 9:16 suite per-minute MTM (REAL, nas_mtm_snapshots)
                       + replayed CSL sleeves (TimeB N/SX, COMB20, SXWED) from stage 1.

Trail: arms at ARM, tracks the running peak, closes EVERYTHING still open when the
portfolio falls to peak - GIVEBACK. CAUSAL - the peak tested at bar t is the peak
carried in from bars < t; a bar can never trigger a trail it just set.

Cost of firing = MEASURED outcome-aware model. A forced mid-session exit pays
+6.548 pt/leg-side against +0.178 for a time exit. Sleeve costs are recomputed in
full off the exact rate card; the suite pays the incremental slippage on its real open lots.

READ-ONLY. Writes results/trail_grid.csv, trail_daily.csv, stage2_report.txt
"""
import sqlite3, csv, os, gzip, random, statistics as st
from collections import defaultdict

Q = "/home/arun/quantifyd/"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []
SLIP_TIME, SLIP_STOP = 0.178, 6.548
SUITE_DBS = [("916_ATM", "nas_916_atm_trading.db"),
             ("916_ATM2", "nas_916_atm2_trading.db"),
             ("916_ATM4", "nas_916_atm4_trading.db")]
NIFTY_LOT = 65
SLEEVE_VEN = {"TB_NIFTY": "NIFTY", "COMB20": "NIFTY", "TB_SENSEX": "SENSEX", "SXWED": "SENSEX"}


def log(m):
    REP.append(str(m)); print(m, flush=True)


def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def cost_short(credit, exitp, lot, nlots, forced):
    sell, buy = credit * lot * nlots, exitp * lot * nlots
    tot = sell + buy
    brok, stt = 80.0, 0.001 * sell
    txn, ipft, sebi = 0.0003503 * tot, 0.0000050 * tot, 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * (SLIP_STOP if forced else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot * nlots


# --------------------------------------------------------------------- load
def load_sleeves():
    days = defaultdict(dict)
    with open(os.path.join(RES, "sleeve_days.csv")) as f:
        for r in csv.DictReader(f):
            days[r["day"]][r["sleeve"]] = dict(
                credit=float(r["credit"]), lot=int(r["lot"]), lots=int(r["lots"]),
                entry=r["entry"], exit_hm=r["exit_hm"], reason=r["reason"],
                gross=float(r["gross_rs"]), cost=float(r["cost_rs"]),
                net=float(r["net_rs"]), dte=int(r["dte"]), venue=r["venue"],
                weekday=r["weekday"], pmap={})
    with gzip.open(os.path.join(RES, "book_minute.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            s = days.get(r["day"], {}).get(r["sleeve"])
            if s is not None:
                s["pmap"][r["t"]] = (float(r["pnl_rs"]), r["state"])
    return days


SUITE_LOTS_NOW = 2          # currently deployed size per 9:16 system


def load_suite():
    """REAL per-minute suite MTM, RESCALED to the CURRENTLY DEPLOYED size.

    The live 9:16 suite has run 5 / 1 / 10 / 2 / 3 / 2 lots per system across the
    recorded window. An absolute-rupee trail grid is meaningless across a size change,
    so every day's per-system MTM is scaled by (SUITE_LOTS_NOW / lots_that_day) --
    i.e. the study replays TODAY'S book over every recorded day, exactly as r/90 did
    ("replays the CURRENT config over all 64 days"). P&L is linear in lots, so the
    rescale is exact; what it cannot rescale is the market impact of a bigger clip.
    """
    mtm = defaultdict(dict)
    openlots = defaultdict(lambda: defaultdict(int))
    lots_seen = defaultdict(set)
    for nm, db in SUITE_DBS:
        c = sqlite3.connect("file:%sbacktest_data/%s?mode=ro" % (Q, db), uri=True)
        lots_by_day, win = {}, defaultdict(list)
        for d, et, xt, lots in c.execute(
                "SELECT trade_date,entry_time,exit_time,lots FROM nas_atm_trades"):
            if not d or not et or not lots:
                continue
            lots_by_day[d] = max(lots_by_day.get(d, 0), int(lots))
            lots_seen[d].add(int(lots))
            win[d].append((et[11:16], xt[11:16] if xt else "15:30"))
        for d, ts, dp in c.execute("SELECT snap_date,ts,day_pnl FROM nas_mtm_snapshots"):
            if dp is None:
                continue
            t = ts[11:16] if "T" in ts else ts[:5]
            L = lots_by_day.get(d)
            v = float(dp) * (float(SUITE_LOTS_NOW) / L) if L else float(dp)
            mtm[d][t] = mtm[d].get(t, 0.0) + v
        for d, ws in win.items():
            for a, b in ws:
                for h in range(9, 16):
                    for m in range(60):
                        t = "%02d:%02d" % (h, m)
                        if a <= t < b:
                            openlots[d][t] = max(openlots[d][t], 0) + 0
            # one system open at t contributes SUITE_LOTS_NOW lots, counted once
            for h in range(9, 16):
                for m in range(60):
                    t = "%02d:%02d" % (h, m)
                    if any(a <= t < b for a, b in ws):
                        openlots[d][t] += SUITE_LOTS_NOW
        c.close()
    return mtm, openlots, lots_seen


class Day(object):
    def __init__(self, day, sleeves, smtm, solots, has_suite):
        self.day, self.sleeves, self.has_suite = day, sleeves, has_suite
        self.solots = solots
        ts = set(smtm) | {t for s in sleeves.values() for t in s["pmap"]}
        self.mins = sorted(t for t in ts if "09:15" <= t <= "15:30")
        self.sm, last = {}, 0.0
        for t in self.mins:
            if t in smtm: last = smtm[t]
            self.sm[t] = last
        self.sp, self.sopen = {}, {}
        for n, s in sleeves.items():
            mp, op, cur, seen = {}, {}, 0.0, False
            for t in self.mins:
                v = s["pmap"].get(t)
                if v is not None:
                    cur, seen = v[0], True
                    op[t] = (v[1] == "OPEN")
                else:
                    op[t] = False
                mp[t] = cur if seen else 0.0
            self.sp[n], self.sopen[n] = mp, op
        self.curve = {t: self.sm[t] + sum(self.sp[n][t] for n in sleeves) for t in self.mins}
        self.base = (self.sm[self.mins[-1]] if self.mins else 0.0) + \
                    sum(s["net"] for s in sleeves.values())
        pk = max(self.mins, key=lambda t: self.curve[t]) if self.mins else None
        self.peak = self.curve[pk] if pk else 0.0
        self.peak_t = pk or ""
        self.final_mtm = self.curve[self.mins[-1]] if self.mins else 0.0

    def scope(self, t, venue):
        if venue is None: return self.curve[t]
        v = self.sm[t] if venue == "NIFTY" else 0.0
        for n in self.sleeves:
            if SLEEVE_VEN[n] == venue: v += self.sp[n][t]
        return v

    def close_at(self, t, venue=None, only_losers=False):
        """net day P&L if in-scope open positions are flattened at t. -> (net, extra_cost, nforced)"""
        net, extra, nf = 0.0, 0.0, 0
        suite_in = venue in (None, "NIFTY")
        if suite_in:
            ol = self.solots.get(t, 0)
            if ol > 0 and not (only_losers and self.sm[t] >= 0):
                inc = (SLIP_STOP - SLIP_TIME) * NIFTY_LOT * ol * 2
                net += self.sm[t] - inc; extra += inc; nf += 1
            else:
                net += self.sm[self.mins[-1]]
        for n, s in self.sleeves.items():
            if venue is not None and SLEEVE_VEN[n] != venue:
                net += s["net"]; continue
            g = self.sp[n][t]
            if self.sopen[n].get(t) and not (only_losers and g >= 0):
                comb = s["credit"] - g / (s["lot"] * s["lots"])
                cst = cost_short(s["credit"], comb, s["lot"], s["lots"], True)
                net += g - cst; extra += cst - s["cost"]; nf += 1
            else:
                net += s["net"]
        return net, extra, nf

    def trail(self, arm, gb, pctgb=False, arm_after=None, tighten_after=None,
              tighten_mul=0.5, venue=None, only_losers=False):
        peak, fire = None, None
        for t in self.mins:
            v = self.scope(t, venue)
            if peak is None:
                if arm_after and t < arm_after: continue
                if v >= arm: peak = v
                continue
            g = peak * gb if pctgb else gb
            if tighten_after and t >= tighten_after: g *= tighten_mul
            if v <= peak - g:
                fire = t; break
            if v > peak: peak = v
        if fire is None:
            return self.base, None, 0.0, 0
        net, extra, nf = self.close_at(fire, venue, only_losers)
        return net, fire, extra, nf

    def fixed_tp(self, tp):
        for t in self.mins:
            if self.curve[t] >= tp:
                net, extra, nf = self.close_at(t)
                return net, t, extra, nf
        return self.base, None, 0.0, 0

    def suite_trail(self, arm_per_lot=2000.0, gb_per_lot=350.0):
        """the EXISTING overlay: watches the 9:16 suite ONLY, arms +Rs2,000/lot,
        gives back Rs350/lot, and closes only the suite."""
        peak, fire = None, None
        for t in self.mins:
            lots = self.solots.get(t, 0)
            if lots <= 0:
                continue
            v = self.sm[t]
            arm, gb = arm_per_lot * lots, gb_per_lot * lots
            if peak is None:
                if v >= arm: peak = v
                continue
            if v <= peak - gb:
                fire = t; break
            if v > peak: peak = v
        if fire is None:
            return self.base, None, 0.0, 0
        ol = self.solots.get(fire, 0)
        inc = (SLIP_STOP - SLIP_TIME) * NIFTY_LOT * ol * 2
        net = self.sm[fire] - inc + sum(s["net"] for s in self.sleeves.values())
        return net, fire, inc, 1


# --------------------------------------------------------------------- main
def summarise(name, nets, bases, fires, extras, days):
    d = [n - b for n, b in zip(nets, bases)]
    nf = sum(1 for f in fires if f)
    needless = sum(1 for x, f in zip(d, fires) if f and x < -1)
    rescue = sum(1 for x, f in zip(d, fires) if f and x > 1)
    cost_needless = sum(x for x, f in zip(d, fires) if f and x < 0)
    return dict(variant=name, n=len(nets), total=round(sum(nets)),
                mean=round(st.mean(nets)), median=round(st.median(nets)),
                win=round(100.0 * sum(1 for x in nets if x > 0) / len(nets), 1),
                worst=round(min(nets)), p10=round(pct(nets, 10)),
                d_total=round(sum(d)), d_mean=round(st.mean(d)),
                d_median=round(st.median(d)), d_worst=round(min(d)), d_best=round(max(d)),
                fires=nf, fire_pct=round(100.0 * nf / len(nets), 1),
                needless=needless, rescue=rescue,
                cost_needless=round(cost_needless),
                firing_cost=round(sum(extras)),
                worst_day_delta=round(min(nets) - min(bases)))


def main():
    sl = load_sleeves()
    smtm, solots, lots_seen = load_suite()
    eras = {}
    for d in sorted(lots_seen):
        eras[tuple(sorted(lots_seen[d]))] = eras.get(tuple(sorted(lots_seen[d])), 0) + 1
    log('suite lot-size eras observed (day counts): %s' % eras)
    log('ALL suite MTM rescaled to the currently deployed %d lots/system' % SUITE_LOTS_NOW)
    alldays = sorted(set(sl) | set(smtm))
    days = []
    for d in alldays:
        has_suite = d in smtm and any(v != 0 for v in smtm[d].values())
        if not sl.get(d) and not has_suite:
            continue
        D = Day(d, sl.get(d, {}), smtm.get(d, {}), solots.get(d, {}), has_suite)
        if len(D.mins) < 20:
            continue
        days.append(D)
    log("days built: %d  %s..%s" % (len(days), days[0].day, days[-1].day))

    # PRIMARY sample = days where the FULL book is represented (suite live + sleeves)
    full = [D for D in days if D.has_suite and D.sleeves]
    sleeve_only = [D for D in days if D.sleeves]
    log("FULL-BOOK sample (suite MTM + >=1 CSL sleeve): n=%d  %s..%s"
        % (len(full), full[0].day, full[-1].day))
    log("SLEEVE-ONLY sample (CSL sleeves, longer history): n=%d  %s..%s"
        % (len(sleeve_only), sleeve_only[0].day, sleeve_only[-1].day))

    for label, sample in (("FULL", full), ("SLEEVEONLY", sleeve_only)):
        base = [D.base for D in sample]
        log("")
        log("=== %s sample: baseline (NO DEFENCE, as deployed) ===" % label)
        log("  n=%d total=%d mean=%d median=%d win%%=%.1f worst=%d p10=%d"
            % (len(sample), sum(base), st.mean(base), st.median(base),
               100.0 * sum(1 for x in base if x > 0) / len(base), min(base), pct(base, 10)))
        gbk = [D.peak - D.final_mtm for D in sample]
        log("  give-back (peak MTM - final MTM): median=%d p75=%d p90=%d max=%d"
            % (st.median(gbk), pct(gbk, 75), pct(gbk, 90), max(gbk)))
        log("  peak MTM: median=%d p90=%d ; days peaking >= 5k: %d ; >=10k: %d"
            % (st.median([D.peak for D in sample]), pct([D.peak for D in sample], 90),
               sum(1 for D in sample if D.peak >= 5000), sum(1 for D in sample if D.peak >= 10000)))

    # ---------------------------------------------------------------- the grid
    ARMS = [3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000]
    GBS = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500]
    PCTS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows, daily = [], []
    sample = full
    base = [D.base for D in sample]

    def run(name, fn):
        nets, fires, extras = [], [], []
        for D in sample:
            n, f, e, nf = fn(D)
            nets.append(n); fires.append(f); extras.append(e)
            daily.append(dict(variant=name, day=D.day, base=round(D.base), net=round(n),
                              delta=round(n - D.base), fire=f or "", extra=round(e),
                              peak=round(D.peak), peak_t=D.peak_t,
                              final_mtm=round(D.final_mtm)))
        r = summarise(name, nets, base, fires, extras, sample)
        rows.append(r)
        return r

    run("NULL_NAKED", lambda D: (D.base, None, 0.0, 0))
    run("NULL_SUITETRAIL_2000_350", lambda D: D.suite_trail(2000, 350))
    for tp in (5000, 7500, 10000, 15000, 20000, 30000):
        run("NULL_FIXEDTP_%d" % tp, lambda D, tp=tp: D.fixed_tp(tp))
    for a in ARMS:
        for g in GBS:
            if g >= a: continue
            run("TRAIL_A%d_G%d" % (a, g), lambda D, a=a, g=g: D.trail(a, g))
    for a in ARMS:
        for p in PCTS:
            run("TRAILPCT_A%d_P%d" % (a, int(p * 100)),
                lambda D, a=a, p=p: D.trail(a, p, pctgb=True))
    # time-conditioned + structural variants around a mid-grid anchor
    for a, g in ((5000, 2500), (8000, 3000), (10000, 3000), (12000, 4000)):
        run("TRAIL_A%d_G%d_ARMAFTER1200" % (a, g),
            lambda D, a=a, g=g: D.trail(a, g, arm_after="12:00"))
        run("TRAIL_A%d_G%d_TIGHT1400" % (a, g),
            lambda D, a=a, g=g: D.trail(a, g, tighten_after="14:00"))
        run("TRAIL_A%d_G%d_ONLYLOSERS" % (a, g),
            lambda D, a=a, g=g: D.trail(a, g, only_losers=True))

        def pv(D, a=a, g=g):
            """independent per-venue trails: each closes only its own venue.
            net = net_N + net_S - base  (each net_X already carries the other
            venue at its baseline, so the baseline is double-counted once)."""
            nn, fn_, en, _ = D.trail(a * 0.6, g * 0.6, venue="NIFTY")
            ns, fs, es, _ = D.trail(a * 0.6, g * 0.6, venue="SENSEX")
            return nn + ns - D.base, (fn_ or fs), en + es, 0
        run("TRAIL_PERVENUE_A%d_G%d" % (a, g), pv)

    # ---------------- PLACEBO: does the trail's TIMING carry information? ----------
    # For each ARM level, fire at a UNIFORMLY RANDOM minute after the book first clears
    # ARM (same arming rule, no peak logic). If a real trail cannot beat this, the
    # peak-tracking machinery is adding nothing over "exit early sometimes".
    rnd = random.Random(20260825)
    log("")
    log("=== PLACEBO: random-minute exit after arming (200 draws per ARM) ===")
    log("%-10s %12s %12s %12s %12s %12s" % ("arm", "placebo_p05", "placebo_med",
                                            "placebo_p95", "best_real_trail", "NULL_NAKED"))
    for a in (5000, 8000, 10000, 12000, 20000):
        draws = []
        for _ in range(200):
            tot = 0.0
            for D in sample:
                armed = [t for t in D.mins if D.curve[t] >= a]
                if not armed:
                    tot += D.base; continue
                after = [t for t in D.mins if t > armed[0]]
                if not after:
                    tot += D.base; continue
                t = rnd.choice(after)
                n_, _e, _nf = D.close_at(t)
                tot += n_
            draws.append(tot)
        real = [r["total"] for r in rows
                if r["variant"].startswith("TRAIL_A%d_G" % a) and "_" not in r["variant"][len("TRAIL_A%d_G" % a):]]
        best = max([r["total"] for r in rows if r["variant"].startswith("TRAIL_A%d_" % a)] or [0])
        log("%-10d %12d %12d %12d %12d %12d" % (
            a, pct(draws, 5), st.median(draws), pct(draws, 95), best, sum(base)))

    rows.sort(key=lambda r: -r["total"])
    fields = list(rows[0].keys())
    with open(os.path.join(RES, "trail_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    with open(os.path.join(RES, "trail_daily.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(daily[0].keys())); w.writeheader()
        for r in daily: w.writerow(r)

    log("")
    log("=== TOP 25 by total net (FULL sample, n=%d) ===" % len(sample))
    hdr = "%-32s %9s %8s %8s %6s %9s %9s %6s %5s %5s %10s"
    log(hdr % ("variant", "total", "mean", "median", "win%", "worst", "d_total",
               "fires", "ndl", "rsc", "cost_ndl"))
    for r in rows[:25]:
        log(hdr % (r["variant"], r["total"], r["mean"], r["median"], r["win"],
                   r["worst"], r["d_total"], r["fires"], r["needless"], r["rescue"],
                   r["cost_needless"]))
    log("")
    log("=== the three NULLS ===")
    for r in rows:
        if r["variant"].startswith("NULL"):
            log(hdr % (r["variant"], r["total"], r["mean"], r["median"], r["win"],
                       r["worst"], r["d_total"], r["fires"], r["needless"], r["rescue"],
                       r["cost_needless"]))
    open(os.path.join(RES, "stage2_report.txt"), "w").write("\n".join(REP) + "\n")
    log("\nwrote trail_grid.csv (%d variants), trail_daily.csv" % len(rows))


if __name__ == "__main__":
    main()
