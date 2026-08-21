#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/119 - analysis of the forward-snap A/B replay. Pure stdlib."""
import csv, os, math, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DETAIL = os.path.join(RES, "fs_detail.csv")
OUT = os.path.join(RES, "analysis.txt")
_buf = []


def P(m=""):
    _buf.append(m)
    print(m, flush=True)


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def stats(xs):
    if not xs:
        return None
    return dict(n=len(xs), total=sum(xs), mean=st.mean(xs),
                med=st.median(xs), win=100.0 * sum(1 for x in xs if x > 0) / len(xs),
                worst=min(xs), best=max(xs), p05=pct(xs, 0.05))


def ttest1(xs):
    if len(xs) < 3:
        return float("nan")
    s = st.pstdev(xs) * math.sqrt(len(xs) / (len(xs) - 1.0))
    return st.mean(xs) / (s / math.sqrt(len(xs))) if s > 0 else float("nan")


def ols(x, y):
    """-> slope, r, t(slope), n"""
    n = len(x)
    if n < 4:
        return (float("nan"),) * 3 + (n,)
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return (float("nan"),) * 3 + (n,)
    b = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    sse = syy - b * sxy
    se = math.sqrt(max(sse, 0) / (n - 2) / sxx) if n > 2 else float("nan")
    return b, r, (b / se if se and se > 0 else float("nan")), n


rows = [r for r in csv.DictReader(open(DETAIL)) if r["dwell"] == "2"]
sens = [r for r in csv.DictReader(open(DETAIL)) if r["dwell"] == "0"]
for r in rows + sens:
    for k in ("fwd_gap", "skew", "credit", "gross_lot", "net_lot", "idx_move_pct", "spot0", "ba_pct"):
        r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
    r["changed"] = r["strike_changed"] == "1"

P("=" * 96)
P("research/119 - FORWARD-ATM (B) vs SPOT-ATM (A) ENTRY, live CSL constructions, real 1-min chain")
P("=" * 96)
days = sorted({r["day"] for r in rows})
P("days covered: %d  (%s .. %s)   detail rows: %d" % (len(days), days[0], days[-1], len(rows)))
P("books x DTE cells: %d" % len({(r["book"], r["dte"]) for r in rows}))

# ---------------------------------------------------------------- 1. DIVERGENCE
P("\n" + "-" * 96)
P("1. DIVERGENCE - how often do the two rules even differ?")
P("-" * 96)
P("%-14s %5s | %8s %8s %8s %8s | %s" % ("venue", "n", "mean|g|", "med|g|", "p90|g|", "max|g|", "strike changed"))
for sym in ("NIFTY", "SENSEX"):
    # one observation per (day, book, dte) - the gap is arm-independent
    seen, gaps, chg = set(), [], []
    for r in rows:
        if r["sym"] != sym or r["arm"] != "B":
            continue
        key = (r["day"], r["book"], r["dte"])
        if key in seen:
            continue
        seen.add(key)
        gaps.append(abs(r["fwd_gap"]))
        chg.append(1 if r["changed"] else 0)
    if not gaps:
        continue
    P("%-14s %5d | %8.1f %8.1f %8.1f %8.1f | %d/%d = %.0f%%" % (
        sym, len(gaps), st.mean(gaps), st.median(gaps), pct(gaps, 0.90), max(gaps),
        sum(chg), len(chg), 100.0 * sum(chg) / len(chg)))
P("\nsigned forward-spot gap (basis; +ve = forward above spot):")
for sym in ("NIFTY", "SENSEX"):
    g = [r["fwd_gap"] for r in rows if r["sym"] == sym and r["arm"] == "B"]
    if g:
        P("  %-8s mean %+7.1f  median %+7.1f  min %+7.1f  max %+7.1f  (%.3f%% of spot at median)"
          % (sym, st.mean(g), st.median(g), min(g), max(g),
             100.0 * st.median(g) / st.mean([r["spot0"] for r in rows if r["sym"] == sym])))
P("\nby DTE - the basis is a cost-of-carry term, so it shrinks toward 0 on expiry day:")
bd = defaultdict(list)
for r in rows:
    if r["arm"] == "B":
        bd[(r["sym"], r["dte"])].append(r)
for k in sorted(bd):
    g = [x["fwd_gap"] for x in bd[k]]
    P("  %-8s DTE%s n=%3d  mean gap %+7.1f  mean |gap| %6.1f  strike changed %3.0f%%" % (
        k[0], k[1], len(g), st.mean(g), st.mean([abs(x) for x in g]),
        100.0 * sum(1 for x in bd[k] if x["changed"]) / len(bd[k])))

P("\nby entry time (09:16/09:20 = the COMB books incl. recorder-gap slippage; rest = TimeB windows):")
bt = defaultdict(list)
for r in rows:
    if r["arm"] == "B":
        bt[(r["sym"], r["entry_hm"][:5])].append(r["fwd_gap"])
for k in sorted(bt):
    if len(bt[k]) >= 5:
        P("  %-8s %-6s n=%3d  mean gap %+7.1f  |gap| mean %6.1f  changed %.0f%%" % (
            k[0], k[1], len(bt[k]), st.mean(bt[k]), st.mean([abs(x) for x in bt[k]]),
            100.0 * sum(1 for r in rows if r["arm"] == "B" and r["sym"] == k[0]
                        and r["entry_hm"][:5] == k[1] and r["changed"]) / len(bt[k])))

# ---------------------------------------------------------------- 3. SKEW
P("\n" + "-" * 96)
P("3. ENTRY SKEW (CE-PE at the strike actually sold) - the mechanism")
P("-" * 96)
P("%-14s %-8s | %6s %9s %9s %9s %9s" % ("book", "arm", "n", "mean|skew|", "med|skew|", "p90|skew|", "mean skew"))
for book in sorted({r["book"] for r in rows}):
    for arm in ("A", "B"):
        s = [r["skew"] for r in rows if r["book"] == book and r["arm"] == arm]
        if s:
            P("%-14s %-8s | %6d %9.1f %9.1f %9.1f %9.1f" % (
                book, arm, len(s), st.mean([abs(x) for x in s]), st.median([abs(x) for x in s]),
                pct([abs(x) for x in s], 0.90), st.mean(s)))
# paired skew reduction
P("\npaired |skew| reduction B vs A (same day/book/dte):")
pair = defaultdict(dict)
for r in rows:
    pair[(r["book"], r["day"], r["dte"])][r["arm"]] = r
for sym in ("NIFTY", "SENSEX"):
    d = [abs(v["B"]["skew"]) - abs(v["A"]["skew"]) for k, v in pair.items()
         if "A" in v and "B" in v and v["A"]["sym"] == sym]
    if d:
        P("  %-8s n=%3d  mean d|skew| %+7.2f   median %+7.2f   t %+6.2f   improved on %.0f%% of days"
          % (sym, len(d), st.mean(d), st.median(d), ttest1(d), 100.0 * sum(1 for x in d if x < 0) / len(d)))

# ---------------------------------------------------------------- 2. P&L
P("\n" + "-" * 96)
P("2. P&L - net Rs per LOT (NIFTY lot 65, SENSEX lot 20), same days both arms")
P("-" * 96)


def pnl_table(groups, title):
    P("\n%s" % title)
    P("%-22s %-4s | %4s %9s %8s %8s %6s %9s %9s" %
      ("cell", "arm", "n", "total", "mean", "median", "win%", "worst", "p05"))
    for g in groups:
        for arm in ("A", "B"):
            xs = [r["net_lot"] for r in rows if groups[g](r) and r["arm"] == arm]
            s = stats(xs)
            if not s:
                continue
            P("%-22s %-4s | %4d %9.0f %8.0f %8.0f %5.0f%% %9.0f %9.0f" % (
                g, arm, s["n"], s["total"], s["mean"], s["med"], s["win"], s["worst"], s["p05"]))
        pa = [(v["A"]["net_lot"], v["B"]["net_lot"]) for k, v in pair.items()
              if "A" in v and "B" in v and groups[g](v["A"])]
        if len(pa) >= 3:
            d = [b - a for a, b in pa]
            P("%-22s %-4s | %4d %9.0f %8.0f %8.0f %5s %9s   t=%+.2f" % (
                "  -> B minus A", "d", len(d), sum(d), st.mean(d), st.median(d),
                "%.0f%%" % (100.0 * sum(1 for x in d if x > 0) / len(d)), "", ttest1(d)))


pnl_table({"ALL": lambda r: True,
           "NIFTY (all books)": lambda r: r["sym"] == "NIFTY",
           "SENSEX (all books)": lambda r: r["sym"] == "SENSEX"}, "-- pooled --")
pnl_table({b: (lambda b: (lambda r: r["book"] == b))(b) for b in sorted({r["book"] for r in rows})},
          "-- per book --")
cells = sorted({(r["book"], r["dte"]) for r in rows})
pnl_table({"%s DTE%s" % (b, d): (lambda b, d: (lambda r: r["book"] == b and r["dte"] == d))(b, d)
           for b, d in cells}, "-- per book x DTE --")

# only the days where the strike actually changed
P("\n-- restricted to days where B actually picked a DIFFERENT strike --")
chgkeys = {k for k, v in pair.items() if "B" in v and v["B"]["changed"]}
for sym in ("NIFTY", "SENSEX"):
    pa = [(v["A"]["net_lot"], v["B"]["net_lot"]) for k, v in pair.items()
          if k in chgkeys and "A" in v and v["A"]["sym"] == sym]
    if len(pa) >= 3:
        d = [b - a for a, b in pa]
        P("  %-8s n=%3d  A mean %8.0f  B mean %8.0f  diff %+8.0f  t %+5.2f  B better on %.0f%%" % (
            sym, len(d), st.mean([a for a, _ in pa]), st.mean([b for _, b in pa]),
            st.mean(d), ttest1(d), 100.0 * sum(1 for x in d if x > 0) / len(d)))

# exit-reason mix
P("\nexit-reason mix (stop-out frequency is where a skewed entry shows up):")
for arm in ("A", "B"):
    cnt = defaultdict(int)
    for r in rows:
        if r["arm"] == arm:
            cnt[r["reason"]] += 1
    tot = sum(cnt.values())
    P("  arm %s: %s" % (arm, "  ".join("%s %d (%.0f%%)" % (k, v, 100.0 * v / tot)
                                       for k, v in sorted(cnt.items(), key=lambda x: -x[1]))))

# ---------------------------------------------------------------- 4. DIRECTIONALITY
P("\n" + "-" * 96)
P("4. DIRECTIONALITY - regress day net P&L (Rs/lot) on the index move over the hold")
P("-" * 96)
P("%-22s %-4s | %4s %11s %8s %8s | %s" % ("cell", "arm", "n", "slope Rs/%", "r", "t", "|r| verdict"))
for g, f in [("ALL", lambda r: True), ("NIFTY", lambda r: r["sym"] == "NIFTY"),
             ("SENSEX", lambda r: r["sym"] == "SENSEX")] + \
            [(b, (lambda b: (lambda r: r["book"] == b))(b)) for b in sorted({r["book"] for r in rows})]:
    for arm in ("A", "B"):
        sub = [r for r in rows if f(r) and r["arm"] == arm and not math.isnan(r["idx_move_pct"])]
        if len(sub) < 6:
            continue
        b_, r_, t_, n_ = ols([x["idx_move_pct"] for x in sub], [x["net_lot"] for x in sub])
        P("%-22s %-4s | %4d %11.0f %8.3f %8.2f | %s" % (
            g, arm, n_, b_, r_, t_, "directional" if abs(t_) >= 2 else "no sig. tilt"))
P("\nsame, on ABS index move (a neutral straddle should be short |move|; both arms will be):")
for g, f in [("NIFTY", lambda r: r["sym"] == "NIFTY"), ("SENSEX", lambda r: r["sym"] == "SENSEX")]:
    for arm in ("A", "B"):
        sub = [r for r in rows if f(r) and r["arm"] == arm and not math.isnan(r["idx_move_pct"])]
        if len(sub) < 6:
            continue
        b_, r_, t_, n_ = ols([abs(x["idx_move_pct"]) for x in sub], [x["net_lot"] for x in sub])
        P("  %-8s %-4s n=%3d slope %8.0f r %6.3f t %6.2f" % (g, arm, n_, b_, r_, t_))
P("\nup-days vs down-days mean net (a delta tilt shows as an asymmetry):")
for g, f in [("NIFTY", lambda r: r["sym"] == "NIFTY"), ("SENSEX", lambda r: r["sym"] == "SENSEX")]:
    for arm in ("A", "B"):
        up = [r["net_lot"] for r in rows if f(r) and r["arm"] == arm and r["idx_move_pct"] > 0]
        dn = [r["net_lot"] for r in rows if f(r) and r["arm"] == arm and r["idx_move_pct"] < 0]
        if up and dn:
            P("  %-8s %-4s up n=%3d mean %8.0f | down n=%3d mean %8.0f | asymmetry %+8.0f"
              % (g, arm, len(up), st.mean(up), len(dn), st.mean(dn), st.mean(up) - st.mean(dn)))

# ---------------------------------------------------------------- 5. MONOTONICITY
P("\n" + "-" * 96)
P("5. MONOTONICITY - does (B-A) scale with the size of the spot-forward gap?")
P("-" * 96)
prs = [(abs(v["B"]["fwd_gap"]), v["B"]["net_lot"] - v["A"]["net_lot"], v["A"]["sym"],
        abs(v["B"]["skew"]) - abs(v["A"]["skew"]))
       for k, v in pair.items() if "A" in v and "B" in v]
for sym in ("NIFTY", "SENSEX", "ALL"):
    sub = [p for p in prs if sym == "ALL" or p[2] == sym]
    if len(sub) < 12:
        continue
    sub.sort(key=lambda p: p[0])
    q = len(sub) // 4
    P("  %s (n=%d) quartiles of |forward-spot|:" % (sym, len(sub)))
    for i in range(4):
        s = sub[i * q:(i + 1) * q] if i < 3 else sub[3 * q:]
        P("    Q%d |gap| %6.1f-%6.1f  n=%3d  mean (B-A) net %+8.0f  t %+5.2f  mean d|skew| %+7.1f"
          % (i + 1, s[0][0], s[-1][0], len(s), st.mean([x[1] for x in s]),
             ttest1([x[1] for x in s]), st.mean([x[3] for x in s])))
    b_, r_, t_, n_ = ols([p[0] for p in sub], [p[1] for p in sub])
    P("    regression (B-A) on |gap|: slope %.1f Rs per gap-point, r %.3f, t %.2f" % (b_, r_, t_))

# ---------------------------------------------------------------- 6. LIQUIDITY
P("\n" + "-" * 96)
P("6. COST OF SWITCHING - bid-ask at the strike actually sold (% of mid, entry minute)")
P("-" * 96)
for sym in ("NIFTY", "SENSEX"):
    for arm in ("A", "B"):
        b = [r["ba_pct"] for r in rows if r["sym"] == sym and r["arm"] == arm and not math.isnan(r["ba_pct"])]
        if b:
            P("  %-8s %-4s n=%3d  mean %6.3f%%  median %6.3f%%  p90 %6.3f%%" % (
                sym, arm, len(b), st.mean(b), st.median(b), pct(b, 0.90)))
    dd = [v["B"]["ba_pct"] - v["A"]["ba_pct"] for k, v in pair.items()
          if "A" in v and "B" in v and v["A"]["sym"] == sym
          and not math.isnan(v["A"]["ba_pct"]) and not math.isnan(v["B"]["ba_pct"])]
    if dd:
        P("           paired d(bid-ask%%) B-A: mean %+.4f  t %+.2f  (>0 = B is the wider/less liquid strike)"
          % (st.mean(dd), ttest1(dd)))
P("\ncredit collected (B sells the balanced strike; if it collects less, that is a real cost):")
for sym in ("NIFTY", "SENSEX"):
    dd = [v["B"]["credit"] - v["A"]["credit"] for k, v in pair.items()
          if "A" in v and "B" in v and v["A"]["sym"] == sym]
    if dd:
        P("  %-8s paired d(credit) B-A: mean %+.2f pts  median %+.2f  t %+.2f" % (
            sym, st.mean(dd), st.median(dd), ttest1(dd)))

# ---------------------------------------------------------------- SENSITIVITY
P("\n" + "-" * 96)
P("ROBUSTNESS - same test with the SL dwell removed (exit on first breach minute)")
P("-" * 96)
sp = defaultdict(dict)
for r in sens:
    sp[(r["book"], r["day"], r["dte"])][r["arm"]] = r
for sym in ("NIFTY", "SENSEX"):
    d = [v["B"]["net_lot"] - v["A"]["net_lot"] for k, v in sp.items()
         if "A" in v and "B" in v and v["A"]["sym"] == sym]
    a = [v["A"]["net_lot"] for k, v in sp.items() if "A" in v and "B" in v and v["A"]["sym"] == sym]
    b = [v["B"]["net_lot"] for k, v in sp.items() if "A" in v and "B" in v and v["A"]["sym"] == sym]
    if d:
        P("  %-8s n=%3d  A mean %8.0f  B mean %8.0f  diff %+8.0f  t %+5.2f" % (
            sym, len(d), st.mean(a), st.mean(b), st.mean(d), ttest1(d)))

P("\n" + "-" * 96)
P("PER-MONTH stability of (B-A), net Rs/lot")
P("-" * 96)
mo = defaultdict(list)
for k, v in pair.items():
    if "A" in v and "B" in v:
        mo[(v["A"]["sym"], v["A"]["day"][:7])].append(v["B"]["net_lot"] - v["A"]["net_lot"])
for k in sorted(mo):
    P("  %-8s %s  n=%3d  mean (B-A) %+8.0f" % (k[0], k[1], len(mo[k]), st.mean(mo[k])))

open(OUT, "w").write("\n".join(_buf) + "\n")
print("\nwritten -> %s" % OUT)
