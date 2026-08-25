#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/125 Stage 3 - ARM C: is the book six systems, or ONE bet at 6x size?

Measures, on the recorded sample:
  1. strike overlap - how often the live NIFTY books are short the SAME strike
  2. cross-book correlation of daily net P&L, overall vs on the worst decile of days
  3. exit clustering - how tightly the real live exits bunch in time
  4. what strike/entry DIVERSIFICATION would have done (the free defence):
     stagger the 9:16 suite / COMB / TimeB across strikes and entry minutes and
     re-measure the joint tail. Priced with the same measured cost model.

READ-ONLY. Writes results/concentration.txt, results/diversify_grid.csv
"""
import sqlite3, csv, os, gzip, statistics as st
from collections import defaultdict

Q = "/home/arun/quantifyd/"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
REP = []
SUITE_DBS = [("916_ATM", "nas_916_atm_trading.db"),
             ("916_ATM2", "nas_916_atm2_trading.db"),
             ("916_ATM4", "nas_916_atm4_trading.db")]


def log(m):
    REP.append(str(m)); print(m, flush=True)


def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def corr(a, b):
    n = len(a)
    if n < 3: return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va = sum((x - ma) ** 2 for x in a); vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0: return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va ** .5 * vb ** .5)


# ---------------------------------------------------------------- 1. strikes
log("=" * 78)
log("1. STRIKE OVERLAP - do the live NIFTY books sell the same strike?")
log("=" * 78)
suite_strike = defaultdict(dict)      # day -> system -> strike
suite_exit = defaultdict(dict)
suite_pnl = defaultdict(dict)
for nm, db in SUITE_DBS:
    c = sqlite3.connect("file:%sbacktest_data/%s?mode=ro" % (Q, db), uri=True)
    for d, ck, pk, xt, npnl, lots, rsn in c.execute(
            "SELECT trade_date,call_strike,put_strike,exit_time,net_pnl,lots,exit_reason "
            "FROM nas_atm_trades ORDER BY trade_date, id"):
        if not d: continue
        suite_strike[d].setdefault(nm, []).append(ck)
        suite_exit[d].setdefault(nm, []).append((xt or "")[11:19])
        suite_pnl[d][nm] = suite_pnl[d].get(nm, 0.0) + (npnl or 0.0) * (2.0 / (lots or 2))
    c.close()

sleeve = defaultdict(dict)
sleeve_meta = {}
with open(os.path.join(RES, "sleeve_days.csv")) as f:
    for r in csv.DictReader(f):
        sleeve[r["day"]][r["sleeve"]] = float(r["strike"])
        sleeve_meta[(r["day"], r["sleeve"])] = r

NIFTY_BOOKS = ["916_ATM", "916_ATM2", "916_ATM4", "COMB20", "TB_NIFTY"]
same, tot, rows = 0, 0, []
for d in sorted(set(suite_strike) | set(sleeve)):
    ks = {}
    for nm in ("916_ATM", "916_ATM2", "916_ATM4"):
        v = suite_strike.get(d, {}).get(nm)
        if v: ks[nm] = v[0]
    for nm in ("COMB20", "TB_NIFTY"):
        v = sleeve.get(d, {}).get(nm)
        if v: ks[nm] = v
    if len(ks) < 2: continue
    tot += 1
    uniq = len(set(ks.values()))
    if uniq == 1: same += 1
    rows.append((d, len(ks), uniq, ks))
log("NIFTY days with >=2 live books trading: %d" % tot)
log("  ALL books on the SAME strike : %d  (%.0f%%)" % (same, 100.0 * same / max(tot, 1)))
byu = defaultdict(int)
for d, nb, u, ks in rows: byu[u] += 1
log("  distinct strikes across the book: %s" % dict(sorted(byu.items())))
# how concentrated among the three suite systems alone
s3 = [r for r in rows if all(k in r[3] for k in ("916_ATM", "916_ATM2", "916_ATM4"))]
s3same = sum(1 for r in s3 if len({r[3][k] for k in ("916_ATM", "916_ATM2", "916_ATM4")}) == 1)
log("  the 3 suite systems alone identical strike: %d / %d (%.0f%%)"
    % (s3same, len(s3), 100.0 * s3same / max(len(s3), 1)))
log("")
log("  last 12 days:")
for d, nb, u, ks in rows[-12:]:
    log("   %s  books=%d distinct=%d  %s" % (d, nb, u, ks))

# ---------------------------------------------------------------- 2. correlation
log("")
log("=" * 78)
log("2. CROSS-BOOK CORRELATION of daily net P&L - overall vs the worst decile")
log("=" * 78)
daily = defaultdict(dict)
for d, m in suite_pnl.items():
    for nm, v in m.items(): daily[d][nm] = v
for (d, nm), r in sleeve_meta.items():
    daily[d][nm] = float(r["net_rs"])
books = ["916_ATM", "916_ATM2", "916_ATM4", "COMB20", "TB_NIFTY", "TB_SENSEX", "SXWED"]
tots = {d: sum(v.values()) for d, v in daily.items()}
alld = sorted(daily)
cut = pct([tots[d] for d in alld], 10)
worst = [d for d in alld if tots[d] <= cut]
log("n days=%d ; worst-decile cutoff = Rs%.0f ; worst-decile days = %d" % (len(alld), cut, len(worst)))
log("")
log("%-11s %-11s %6s %8s %6s %8s" % ("book A", "book B", "n_all", "corr_all", "n_bad", "corr_bad"))
for i in range(len(books)):
    for j in range(i + 1, len(books)):
        a, b = books[i], books[j]
        pa = [(daily[d][a], daily[d][b]) for d in alld if a in daily[d] and b in daily[d]]
        pb = [(daily[d][a], daily[d][b]) for d in worst if a in daily[d] and b in daily[d]]
        if len(pa) < 8: continue
        ca = corr([x for x, _ in pa], [y for _, y in pa])
        cb = corr([x for x, _ in pb], [y for _, y in pb]) if len(pb) >= 3 else float("nan")
        log("%-11s %-11s %6d %8.2f %6d %8.2f" % (a, b, len(pa), ca, len(pb), cb))

# fraction of the book losing together
log("")
allneg, anyneg = 0, 0
for d in alld:
    vs = list(daily[d].values())
    if len(vs) < 2: continue
    if all(v < 0 for v in vs): allneg += 1
    if any(v < 0 for v in vs): anyneg += 1
log("days where EVERY live book lost: %d ; where any lost: %d ; of %d multi-book days"
    % (allneg, anyneg, sum(1 for d in alld if len(daily[d]) >= 2)))
badall = sum(1 for d in worst if len(daily[d]) >= 2 and all(v < 0 for v in daily[d].values()))
log("  on the WORST DECILE days, every book lost on %d of %d" % (badall, len(worst)))

# ---------------------------------------------------------------- 3. exit clustering
log("")
log("=" * 78)
log("3. EXIT CLUSTERING - do the books flatten into the same 90 seconds?")
log("=" * 78)
clus = []
for d in sorted(suite_exit):
    xs = sorted(t for v in suite_exit[d].values() for t in v if t and t < "15:10")
    if len(xs) < 2: continue
    def sec(t): return int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    span = sec(xs[-1]) - sec(xs[0])
    clus.append((d, len(xs), span, xs))
log("days with >=2 non-EOD suite exits: %d" % len(clus))
if clus:
    sp = [s for _, _, s, _ in clus]
    log("  span between first and last mid-session exit: median=%.0fs p25=%.0fs p75=%.0fs"
        % (st.median(sp), pct(sp, 25), pct(sp, 75)))
    log("  days where ALL mid-session exits fell inside 120s: %d (%.0f%%)"
        % (sum(1 for s in sp if s <= 120), 100.0 * sum(1 for s in sp if s <= 120) / len(sp)))
    log("  recent examples:")
    for d, n, s, xs in clus[-8:]:
        log("   %s  n=%d span=%4ds  %s" % (d, n, s, xs))

open(os.path.join(RES, "concentration.txt"), "w").write("\n".join(REP) + "\n")
print("\nwrote results/concentration.txt")
