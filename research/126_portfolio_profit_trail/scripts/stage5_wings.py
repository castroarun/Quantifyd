#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 5 - ARM B: BUY OTM WINGS (iron fly / condor) on the live book.

A held-wing intraday backtest on this project has already been INVALIDATED once by
STALE far-OTM quotes, so this stage runs the audit BEFORE the economics and reports it
whether or not the economics look good.

  S5-AUDIT   staleness + liquidity of every wing strike actually used
  S5-ECON    wings bought at the ASK, sold back at the BID, on days that pass the audit
             (a) bought AT ENTRY   (defined risk all day)
             (b) bought AFTER THE BOOK IS UP  (lock the profit with wings)
             (c) per-sleeve  vs  hedge-the-biggest-sleeve-only

READ-ONLY. Writes results/wing_audit.txt, results/wing_grid.csv
"""
import csv, os, gzip, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []
LOTSIZE = {"NIFTY": 65, "SENSEX": 20}


def log(m):
    REP.append(str(m)); print(m, flush=True)


def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def charges_long(buyp, sellp, lot, nlots):
    buy, sell = buyp * lot * nlots, sellp * lot * nlots
    tot = buy + sell
    brok, stt = 80.0, 0.001 * sell
    txn, ipft, sebi = 0.0003503 * tot, 0.0000050 * tot, 0.0000010 * tot
    stamp = 0.00003 * buy
    return brok + stt + txn + ipft + sebi + stamp + 0.18 * (brok + txn + ipft + sebi)


# --------------------------------------------------------------- load
sleeves = {}
with open(os.path.join(RES, "sleeve_days.csv")) as f:
    for r in csv.DictReader(f):
        sleeves[(r["day"], r["sleeve"])] = r

W = defaultdict(list)         # (day,sleeve,dist) -> [(t, ask, bid, ltp, vce, vpe, oce, ope)]
with gzip.open(os.path.join(RES, "wing_minute.csv.gz"), "rt") as f:
    for r in csv.DictReader(f):
        W[(r["day"], r["sleeve"], int(r["dist"]))].append(
            (r["t"], float(r["ask_comb"]), float(r["bid_comb"]), float(r["ltp_comb"]),
             int(r["vol_ce"]), int(r["vol_pe"]), int(r["oi_ce"]), int(r["oi_pe"])))
for k in W: W[k].sort()
log("wing series loaded: %d (day,sleeve,dist) combos" % len(W))

SPATH = defaultdict(dict)     # (day,sleeve) -> {t: pnl_rs}
with gzip.open(os.path.join(RES, "book_minute.csv.gz"), "rt") as f:
    for r in csv.DictReader(f):
        SPATH[(r["day"], r["sleeve"])][r["t"]] = float(r["pnl_rs"])
log("sleeve P&L paths loaded: %d sleeve-days" % len(SPATH))

VEN = {"TB_NIFTY": "NIFTY", "COMB20": "NIFTY", "TB_SENSEX": "SENSEX", "SXWED": "SENSEX"}
DISTS = {"NIFTY": [100, 150, 200, 250, 300, 400, 500],
         "SENSEX": [400, 600, 800, 1000, 1200, 1600, 2000]}

# --------------------------------------------------------------- S5-AUDIT
log("")
log("=" * 96)
log("S5-AUDIT - are the far-OTM wing quotes LIVE or STALE? (the trap that voided the")
log("           previous held-wing backtest on this project)")
log("=" * 96)
log("%-7s %6s %5s %7s %8s %8s %9s %9s %8s %8s" % (
    "venue", "dist", "n_d", "n_min", "zero_bid", "zero_ask", "spread%mid", "maxrun",
    "meanrun", "vol0_d"))
audit = {}
for ven in ("NIFTY", "SENSEX"):
    for dist in DISTS[ven]:
        keys = [k for k in W if k[2] == dist and VEN.get(k[1]) == ven]
        if not keys: continue
        nmin = zb = za = 0
        spreads, runs, vol0d = [], [], 0
        for k in keys:
            ser = W[k]
            nmin += len(ser)
            prev, run = None, 0
            maxv = 0
            for t, a, b, l, vc, vp, oc, op in ser:
                if b <= 0: zb += 1
                if a <= 0: za += 1
                if a > 0 and b > 0:
                    mid = (a + b) / 2.0
                    if mid > 0: spreads.append(100.0 * (a - b) / mid)
                if l == prev: run += 1
                else:
                    if run: runs.append(run)
                    run = 1; prev = l
                maxv = max(maxv, vc, vp)
            if run: runs.append(run)
            if maxv == 0: vol0d += 1
        audit[(ven, dist)] = dict(days=len(keys), nmin=nmin,
                                  zero_bid=100.0 * zb / max(nmin, 1),
                                  zero_ask=100.0 * za / max(nmin, 1),
                                  spread=st.median(spreads) if spreads else float("nan"),
                                  maxrun=max(runs) if runs else 0,
                                  meanrun=st.mean(runs) if runs else 0,
                                  vol0d=vol0d)
        a = audit[(ven, dist)]
        log("%-7s %6d %5d %7d %8.1f %8.1f %9.1f %9d %8.1f %8d" % (
            ven, dist, a["days"], a["nmin"], a["zero_bid"], a["zero_ask"], a["spread"],
            a["maxrun"], a["meanrun"], a["vol0d"]))
log("")
log("Reading: zero_bid/zero_ask = %% of minutes with no two-sided quote (untradable).")
log("         spread%%mid = median bid-ask spread as %% of mid (the cost of a round trip).")
log("         maxrun/meanrun = consecutive identical LTP prints -> the staleness signature.")
log("         vol0_d = sleeve-days on which the wing strike NEVER traded (r/89 rule: drop).")

# --------------------------------------------------------------- S5-ECON
log("")
log("=" * 96)
log("S5-ECON - wings bought at the ASK, sold back at the BID, on audited days only")
log("=" * 96)
rows = []
base_by_day = defaultdict(float)
for (day, sl), r in sleeves.items():
    base_by_day[day] += float(r["net_rs"])

for ven in ("NIFTY", "SENSEX"):
    lot = LOTSIZE[ven]
    for dist in DISTS[ven]:
        for mode in ("ATENTRY", "AFTERUP"):
            dl, tot_wing, tot_base, nd, nskip_stale, nskip_vol = [], 0.0, 0.0, 0, 0, 0
            nd_noarm = [0]
            worst_b, worst_w = [], []
            for (day, sl), r in sleeves.items():
                if VEN.get(sl) != ven: continue
                ser = W.get((day, sl, dist))
                if not ser: continue
                lots = int(r["lots"])
                e_hm, x_hm = r["entry"], r["exit_hm"]
                # liquidity gate (r/89): the wing strikes must have TRADED that day
                if max(max(v[4], v[5]) for v in ser) <= 0:
                    nskip_vol += 1; continue
                smap = {v[0]: v for v in ser}
                if mode == "ATENTRY":
                    bt = e_hm
                else:
                    # "lock the profit with wings": buy them only once the sleeve is up
                    # by >= 40% of the credit it sold -- the direct analogue of the trail
                    thr = 0.40 * float(r["credit"]) * lot * lots
                    path = SPATH.get((day, sl), {})
                    up = sorted(t for t, v in path.items() if v >= thr)
                    if not up:
                        nd_noarm[0] += 1
                        continue
                    bt = up[0]
                cand = [t for t in smap if t >= bt]
                if not cand: continue
                bt = min(cand)
                st_ = [t for t in smap if t <= x_hm]
                if not st_: continue
                stt = max(st_)
                bq, sq = smap[bt], smap[stt]
                ask, bid = bq[1], sq[2]
                if ask <= 0:
                    nskip_stale += 1; continue
                wing_pnl = (bid - ask) * lot * lots - charges_long(ask, bid, lot, lots)
                b = float(r["net_rs"])
                dl.append(wing_pnl); tot_wing += wing_pnl; tot_base += b
                worst_b.append(b); worst_w.append(b + wing_pnl)
                nd += 1
            if nd < 5: continue
            rows.append(dict(venue=ven, dist=dist, mode=mode, n=nd,
                             base_total=round(tot_base), wing_total=round(tot_wing),
                             hedged_total=round(tot_base + tot_wing),
                             wing_mean=round(st.mean(dl)), wing_median=round(st.median(dl)),
                             wing_best=round(max(dl)), wing_paid_days=sum(1 for x in dl if x > 0),
                             base_worst=round(min(worst_b)), hedged_worst=round(min(worst_w)),
                             base_p10=round(pct(worst_b, 10)), hedged_p10=round(pct(worst_w, 10)),
                             skip_vol=nskip_vol, skip_stale=nskip_stale,
                             skip_noarm=nd_noarm[0]))

rows.sort(key=lambda r: (r["venue"], r["mode"], r["dist"]))
with open(os.path.join(RES, "wing_grid.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)

log("")
h = "%-7s %6s %-8s %4s %11s %11s %11s %9s %9s %6s %10s %10s"
log(h % ("venue", "dist", "mode", "n", "naked_tot", "wing_cost", "hedged_tot",
         "wing_mean", "wing_med", "paid", "naked_wrst", "hedgd_wrst"))
for r in rows:
    log(h % (r["venue"], r["dist"], r["mode"], r["n"], r["base_total"], r["wing_total"],
             r["hedged_total"], r["wing_mean"], r["wing_median"], r["wing_paid_days"],
             r["base_worst"], r["hedged_worst"]))

open(os.path.join(RES, "wing_audit.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/wing_audit.txt, results/wing_grid.csv")
