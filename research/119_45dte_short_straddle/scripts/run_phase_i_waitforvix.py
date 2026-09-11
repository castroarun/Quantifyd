#!/usr/bin/env python3
"""Phase I — if the VIX filter fails on the entry day but passes a day or two
later, should you still take the campaign?

Arun's question, and a real gap: the rule of record checks India VIX percentile
rank ONCE, on the 45-DTE entry session. Rank <= 25 and the entire monthly cycle
is skipped - 28 of 89 campaigns. But volatility is regime-y, so a cycle that
misses by a whisker on day 45 may clear the bar on day 44 or 43. Nobody has ever
tested whether taking those late is worth doing.

Arms (campaign = one monthly expiry; exit is ALWAYS the 21-DTE close, so a later
entry simply has a shorter hold and collects less time):

  BASE      enter on the nominal entry day if rank > 25, else SKIP   <- rule of record
  WAIT-N    if the nominal day fails, scan forward up to N sessions and enter on
            the FIRST one whose rank clears 25; still skip if none does

The strike is re-picked from that day's spot and the credit re-read from that
day's closes, so a late entry is priced honestly rather than borrowed from the
nominal day.

Two things this must report, because either could make a gain illusory:
  * how many EXTRA campaigns each N actually adds (a gain from 2 trades is noise)
  * what the added trades earn ON THEIR OWN, not blended with the 61 that the
    baseline already takes - that is the only number that says whether waiting
    is a good idea

READ ONLY against market_data.db.
"""
import csv
import os
import sqlite3
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if not os.path.exists(os.path.join(ROOT, "backtest_data")):
    ROOT = "/home/arun/quantifyd"
RES = os.path.join(os.path.dirname(HERE), "results")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
TRADES = os.path.join(RES, "trades_daily.csv")

LOT, LOTS = 65, 3
QTY = LOT * LOTS
STEP = 50
SLIP = 0.0025
MIN_VOL, MIN_OI = 100, 500
DTE_OUT = 21
VIX_RANK_MIN = 25
WAITS = [0, 1, 2, 3, 5, 10]


def costs_points(a, b):
    return (SLIP * (a + b) + 0.0010 * a + 0.0005 * (a + b)
            + (20.0 * 4) / QTY + 0.18 * (0.0005 * (a + b) + (20.0 * 4) / QTY))


def roll_off_weekend(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def stats(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5
    return mu, (mu / (sd / n ** 0.5)) if sd else float("nan")


def maxdd(xs):
    eq = pk = dd = 0.0
    for x in xs:
        eq += x
        pk = max(pk, eq)
        dd = min(dd, eq - pk)
    return dd


def main():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    spot = {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' "
        "AND timeframe='day' AND close IS NOT NULL")}
    vx = sorted((r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' AND close IS NOT NULL"))
    vpos = {d: i for i, (d, _) in enumerate(vx)}

    def rank(day):
        i = vpos.get(day)
        if i is None or i < 252:
            return None
        w = [v for _, v in vx[i - 252:i]]
        return 100.0 * sum(1 for x in w if x < vx[i][1]) / len(w)

    sess = sorted(spot)

    def chain(expiry, day):
        out = {}
        for K, ot, c, v, oi in con.execute(
                "SELECT strike, option_type, close, contracts, open_interest "
                "FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? "
                "AND trade_date=?", (expiry, day)):
            out[(float(K), ot)] = (float(c or 0), int(v or 0), int(oi or 0))
        return out

    def straddle(ch, K, liquid=True):
        ce, pe = ch.get((float(K), "CE")), ch.get((float(K), "PE"))
        if not ce or not pe or ce[0] <= 0 or pe[0] <= 0:
            return None
        if liquid and (min(ce[1], pe[1]) < MIN_VOL or min(ce[2], pe[2]) < MIN_OI):
            return None
        return ce[0] + pe[0]

    camps = list(csv.DictReader(open(TRADES)))
    print("monthly campaigns: %d" % len(camps))

    def run(expiry, ed):
        """Enter on session `ed`, exit at the 21-DTE close. Returns (net, hold_days)."""
        xd = roll_off_weekend(
            date(*map(int, expiry.split("-"))) - timedelta(days=DTE_OUT)).isoformat()
        xs = [d for d in sess if d <= xd]
        if not xs:
            return None
        xd = xs[-1]
        if xd <= ed:
            return None
        sp = spot.get(ed)
        if not sp:
            return None
        K = round(sp / STEP) * STEP
        ce = chain(expiry, ed)
        cred = straddle(ce, K)
        if cred is None:
            return None
        ex = straddle(chain(expiry, xd), K, liquid=False)
        if ex is None:
            return None
        held = (date(*map(int, xd.split("-"))) - date(*map(int, ed.split("-")))).days
        return (cred - ex) - costs_points(cred, ex), held

    results = {}
    added_detail = {}
    for N in WAITS:
        nets, added, skipped = [], [], 0
        for c in camps:
            exp, ed0 = c["expiry"], c["entry_date"]
            i = [j for j, d in enumerate(sess) if d == ed0]
            if not i:
                continue
            i = i[0]
            chosen, late = None, 0
            for step in range(0, N + 1):
                if i + step >= len(sess):
                    break
                d = sess[i + step]
                r = rank(d)
                if r is not None and r > VIX_RANK_MIN:
                    chosen, late = d, step
                    break
            if chosen is None:
                skipped += 1
                continue
            got = run(exp, chosen)
            if not got:
                continue
            net, held = got
            nets.append(net)
            if late > 0:
                added.append(dict(expiry=exp, nominal=ed0, entered=chosen,
                                  late_sessions=late, net=round(net, 1),
                                  held=held, rank=round(rank(chosen), 1)))
        mu, t = stats(nets)
        results[N] = dict(n=len(nets), extra=len(added), skipped=skipped,
                          net=mu, t=t, total=sum(nets), maxdd=maxdd(nets),
                          win=100.0 * sum(1 for x in nets if x > 0) / len(nets))
        added_detail[N] = added

    print("\n" + "=" * 86)
    print("WAIT-N: if the entry day fails the VIX filter, look forward N sessions")
    print("=" * 86)
    print("%-8s %6s %7s %8s %10s %7s %10s %8s %9s"
          % ("arm", "trades", "extra", "skipped", "net/camp", "t", "total", "win%", "maxDD"))
    for N in WAITS:
        r = results[N]
        print("%-8s %6d %7d %8d %10.1f %7.2f %10.1f %7.1f%% %9.1f"
              % ("BASE" if N == 0 else "WAIT-%d" % N, r["n"], r["extra"],
                 r["skipped"], r["net"], r["t"], r["total"], r["win"], r["maxdd"]))

    print("\n" + "=" * 86)
    print("THE ONLY NUMBER THAT MATTERS: what do the LATE entries earn on their own?")
    print("=" * 86)
    for N in WAITS[1:]:
        a = added_detail[N]
        if not a:
            print("  WAIT-%-2d  no late entries" % N)
            continue
        xs = [x["net"] for x in a]
        mu, t = stats(xs)
        print("  WAIT-%-2d  n=%2d  mean %+8.1f pts  t %+5.2f  win %4.0f%%  avg hold %.0fd  "
              "(baseline on-time trades earn %+.1f)"
              % (N, len(xs), mu, t, 100.0 * sum(1 for x in xs if x > 0) / len(xs),
                 sum(x["held"] for x in a) / len(a), results[0]["net"]))

    print("\n  every late entry WAIT-10 would have taken:")
    print("  %-12s %-12s %-12s %5s %6s %7s %8s"
          % ("expiry", "nominal", "entered", "late", "rank", "held", "net"))
    for x in sorted(added_detail[10], key=lambda z: z["expiry"]):
        print("  %-12s %-12s %-12s %5d %6.1f %6dd %+8.1f"
              % (x["expiry"], x["nominal"], x["entered"], x["late_sessions"],
                 x["rank"], x["held"], x["net"]))

    with open(os.path.join(RES, "phase_i_waitforvix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wait_n", "trades", "extra", "skipped", "net_per_camp", "t",
                    "total", "win_pct", "maxdd"])
        for N in WAITS:
            r = results[N]
            w.writerow([N, r["n"], r["extra"], r["skipped"], round(r["net"], 2),
                        round(r["t"], 3), round(r["total"], 1), round(r["win"], 1),
                        round(r["maxdd"], 1)])
    print("\nwrote %s" % os.path.join(RES, "phase_i_waitforvix.csv"))


if __name__ == "__main__":
    sys.exit(main())
