#!/usr/bin/env python3
"""research/134 Stage A — characterise the problem before proposing any solution.

The concern is that the whole book is short-vol and delta-neutral, so a trending
market hurts everything at once. Before searching for a diversifier, establish
whether that is actually true, and if so, WHAT STATE the joint losses live in.

If the bad months turn out to be idiosyncratic and unrelated to index moves, then
no index-level sleeve can help and the study must redirect. That is the gate.

Reads only: research CSVs + market_data.db. Writes only into results/.
"""
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(HERE, "results")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
C1 = os.path.join(ROOT, "research", "127_stock_neutral_wings", "results", "phase_e_equity.csv")
S45 = os.path.join(ROOT, "research", "119_45dte_short_straddle", "results", "trades_daily.csv")

# The 45-DTE book's own sizing, from research/119: 3 lots on Rs11.96L.
LOT, LOTS, CAP45 = 65, 3, 1_196_000.0


def out(name):
    os.makedirs(RES, exist_ok=True)
    return os.path.join(RES, name)


def index_monthly():
    """NIFTY month-end closes -> monthly return, plus max intra-month drawdown and
    the largest 21-day move inside the month (the 'trend' measure that matters to a
    45-day straddle: it is not the month's net change that kills you, it is the run)."""
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    rows = [(r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol='NIFTY50' AND timeframe='day' AND close IS NOT NULL ORDER BY date")]
    vix = {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified "
        "WHERE symbol='INDIAVIX' AND timeframe='day' AND close IS NOT NULL ORDER BY date")}
    con.close()

    bym = defaultdict(list)
    for d, c in rows:
        bym[d[:7]].append((d, c))
    months = sorted(bym)
    out_ = {}
    prev_close = None
    for i, m in enumerate(months):
        ser = bym[m]
        closes = [c for _, c in ser]
        first, last = closes[0], closes[-1]
        base = prev_close if prev_close else first
        ret = 100.0 * (last / base - 1)
        # deepest peak-to-trough inside the month, both directions
        peak = trough = closes[0]
        dd = ru = 0.0
        for c in closes:
            peak = max(peak, c)
            trough = min(trough, c)
            dd = min(dd, 100.0 * (c / peak - 1))
            ru = max(ru, 100.0 * (c / trough - 1))
        # trailing 21-session absolute move ending in this month (the straddle's horizon)
        idx_end = sum(len(bym[x]) for x in months[:i + 1]) - 1
        idx_45 = max(0, idx_end - 44)
        run45 = 100.0 * (rows[idx_end][1] / rows[idx_45][1] - 1)
        vlist = [vix[d] for d, _ in ser if d in vix]
        out_[m] = dict(ret=ret, intra_dd=dd, intra_ru=ru, run45=run45,
                       vix_avg=(sum(vlist) / len(vlist)) if vlist else None,
                       close=last)
        prev_close = last
    return out_


def c1_monthly():
    """C1 stock winged-strangle portfolio: monthly return, already net of costs."""
    outm = {}
    with open(C1) as f:
        for r in csv.DictReader(f):
            outm[r["cycle"]] = 100.0 * float(r["ret"])
    return outm


def s45_monthly():
    """45-DTE NIFTY straddle: net points -> % of the book's own capital.

    Attributed to the EXIT month, which is when the P&L is actually realised and
    when the drawdown is felt. Attributing to entry would smear a loss backwards.
    """
    outm = defaultdict(float)
    n = defaultdict(int)
    with open(S45) as f:
        for r in csv.DictReader(f):
            rs = float(r["net_pts"]) * LOT * LOTS
            m = r["exit_date"][:7]
            outm[m] += 100.0 * rs / CAP45
            n[m] += 1
    return dict(outm), dict(n)


def stats(xs):
    n = len(xs)
    if not n:
        return None
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd, n


def corr(a, b):
    common = sorted(set(a) & set(b))
    if len(common) < 12:
        return None, len(common)
    xa = [a[m] for m in common]
    xb = [b[m] for m in common]
    ma, sa, _ = stats(xa)
    mb, sb, _ = stats(xb)
    if sa == 0 or sb == 0:
        return None, len(common)
    cov = sum((x - ma) * (y - mb) for x, y in zip(xa, xb)) / (len(common) - 1)
    return cov / (sa * sb), len(common)


def drawdown(series_by_month):
    """Compound the monthly returns and return (maxDD %, the month it bottomed)."""
    eq, peak, mdd, at = 1.0, 1.0, 0.0, None
    for m in sorted(series_by_month):
        eq *= (1 + series_by_month[m] / 100.0)
        peak = max(peak, eq)
        d = 100.0 * (eq / peak - 1)
        if d < mdd:
            mdd, at = d, m
    return mdd, at, 100.0 * (eq - 1)


def main():
    idx = index_monthly()
    c1 = c1_monthly()
    s45, s45n = s45_monthly()

    print("=" * 78)
    print("STAGE A - is the book actually one bet?")
    print("=" * 78)
    print("C1 stock wings : %d months  %s -> %s" % (len(c1), min(c1), max(c1)))
    print("45-DTE straddle: %d months  %s -> %s (%d trades)"
          % (len(s45), min(s45), max(s45), sum(s45n.values())))

    # ---- 1. correlation between the two neutral sleeves ---------------------
    r, n = corr(c1, s45)
    print("\n1. CORRELATION between the two neutral sleeves")
    print("   corr(C1, 45-DTE) = %s   over %d common months"
          % ("%.3f" % r if r is not None else "n/a", n))

    # ---- 2. the combined book ----------------------------------------------
    # Equal risk: scale each to unit monthly vol, then half weight each.
    common = sorted(set(c1) & set(s45))
    _, sd_c1, _ = stats([c1[m] for m in common])
    _, sd_45, _ = stats([s45[m] for m in common])
    w1, w2 = 0.5 / sd_c1, 0.5 / sd_45
    comb = {m: w1 * c1[m] + w2 * s45[m] for m in common}
    # rescale the combined series to 4% monthly vol so numbers are readable
    _, sd_c, _ = stats(list(comb.values()))
    k = 4.0 / sd_c
    comb = {m: v * k for m, v in comb.items()}

    mu, sd, n = stats(list(comb.values()))
    mdd, at, total = drawdown(comb)
    print("\n2. COMBINED neutral book (equal-risk, scaled to 4%%/mo vol), %d months" % n)
    print("   mean %+.2f%%/mo   vol %.2f%%   worst %+.2f%%   maxDD %.1f%% (bottom %s)"
          % (mu, sd, min(comb.values()), mdd, at))

    # ---- 3. what was the index doing in the worst months? -------------------
    ranked = sorted(comb.items(), key=lambda kv: kv[1])
    worst = ranked[:10]
    best = ranked[-10:]
    print("\n3. THE WORST 10 MONTHS - and what NIFTY was doing")
    print("   %-8s %8s %8s %9s %9s %8s %7s %7s" % (
        "month", "book%", "C1%", "45DTE%", "NIFTY%", "45d run", "intraDD", "VIX"))
    for m, v in worst:
        i = idx.get(m, {})
        print("   %-8s %+8.2f %+8.2f %+9.2f %+9.2f %+8.2f %7.1f %7s" % (
            m, v, c1.get(m, 0), s45.get(m, 0), i.get("ret", 0),
            i.get("run45", 0), i.get("intra_dd", 0),
            "%.1f" % i["vix_avg"] if i.get("vix_avg") else "-"))

    # ---- 4. is the pain trend-linked, or idiosyncratic? ---------------------
    print("\n4. IS THE PAIN TREND-LINKED?  (the gate for this whole study)")
    absret = {m: abs(idx[m]["ret"]) for m in comb if m in idx}
    absrun = {m: abs(idx[m]["run45"]) for m in comb if m in idx}
    dnrun = {m: -min(0.0, idx[m]["run45"]) for m in comb if m in idx}
    uprun = {m: max(0.0, idx[m]["run45"]) for m in comb if m in idx}
    for name, ser in (("|NIFTY monthly return|", absret),
                      ("|NIFTY 45-day run|", absrun),
                      ("NIFTY 45-day run DOWN only", dnrun),
                      ("NIFTY 45-day run UP only", uprun)):
        r2, n2 = corr(comb, ser)
        print("   corr(book, %-28s) = %s   n=%d"
              % (name, "%+.3f" % r2 if r2 is not None else "n/a", n2))

    # decile table: book return by |45-day index run|
    pairs = sorted(((absrun[m], comb[m]) for m in absrun), key=lambda p: p[0])
    q = max(1, len(pairs) // 5)
    print("\n   book return by size of the 45-day index move (quintiles):")
    print("   %-22s %6s %9s %9s" % ("|45d run| bucket", "n", "mean", "worst"))
    for i in range(0, len(pairs), q):
        ch = pairs[i:i + q]
        if len(ch) < 3:
            break
        vals = [v for _, v in ch]
        m2, _, _ = stats(vals)
        print("   %5.1f%% - %5.1f%%%9s %6d %+9.2f %+9.2f"
              % (ch[0][0], ch[-1][0], "", len(ch), m2, min(vals)))

    # ---- 5. direction of the damaging move ---------------------------------
    dn = [comb[m] for m in comb if m in idx and idx[m]["run45"] <= -5]
    up = [comb[m] for m in comb if m in idx and idx[m]["run45"] >= 5]
    fl = [comb[m] for m in comb if m in idx and abs(idx[m]["run45"]) < 5]
    print("\n5. WHICH DIRECTION HURTS?  (decides put vs call vs both)")
    for lab, xs in (("45d run <= -5%% (down trend)", dn),
                    ("45d run >= +5%% (up trend)", up),
                    ("|45d run| <  5%% (chop)", fl)):
        if xs:
            m2, s2, n2 = stats(xs)
            print("   %-28s n=%3d  mean %+6.2f%%  worst %+6.2f%%" % (lab, n2, m2, min(xs)))

    # ---- write the series out ----------------------------------------------
    with open(out("stage_a_monthly.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "combined_pct", "c1_pct", "s45_pct", "nifty_ret_pct",
                    "nifty_run45_pct", "nifty_intra_dd_pct", "vix_avg"])
        for m in sorted(comb):
            i = idx.get(m, {})
            w.writerow([m, round(comb[m], 4), round(c1.get(m, 0), 4),
                        round(s45.get(m, 0), 4), round(i.get("ret", 0), 3),
                        round(i.get("run45", 0), 3), round(i.get("intra_dd", 0), 3),
                        round(i["vix_avg"], 2) if i.get("vix_avg") else ""])
    print("\nwrote %s  (%d months)" % (out("stage_a_monthly.csv"), len(comb)))
    print("\nBEST 5 months, for contrast: %s"
          % ", ".join("%s %+.1f%%" % (m, v) for m, v in reversed(best[-5:])))


if __name__ == "__main__":
    sys.exit(main())
