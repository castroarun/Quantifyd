#!/usr/bin/env python3
"""Phase G — premium-triggered management: re-deploy, or ADD a second straddle.

Phase E refuted MOVE-triggered management (cut at x% underlying move, optionally
re-centre) with a mechanism: cutting converts a winner-in-waiting into a booked
loser. This asks a different question on two axes — the trigger is the PREMIUM
ratio (the live 200%-of-credit stop), and one family of arms ADDS a second
straddle while keeping the first. An arm that never closes the original cannot
suffer Phase E's mechanism; whether it suffers a worse one is the open question.

Arms, all on real NSE bhavcopy closes, campaign (45 -> 21 DTE) as the unit:

  HOLD        baseline, no stop, run to 21 DTE
  STOP        close everything at the trigger, flat to 21 DTE  (the LIVE rule at 200%)
  RECENTRE    close, sell a fresh ATM straddle, run on
  ADD_ATM     keep the original, sell a SECOND straddle at the then-ATM
  ADD_MIRROR  keep the original, sell a second at K1 = 2*S - K0, so the pair
              brackets spot symmetrically  <- the ask's "equidistant" variant

A rule cannot be applied only to the trades already known to have lost — that is
look-ahead. Every arm runs on ALL campaigns; the fired subset is a diagnostic.

The ADD arms hold TWO straddles after the trigger, so raw points would reward
them for nothing but being bigger. Size-normalised net (per unit of PEAK
straddles held) is the column that decides.

READ ONLY against market_data.db.
"""
import csv
import os
import sqlite3
import sys
from collections import defaultdict
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
MIN_VOL, MIN_OI = 100, 500      # a leg must really have traded to be sellable
TRIGGERS = [1.30, 1.50, 1.75, 2.00]
CAP = 1                          # fire at most once per campaign (the ask's shape)


def costs_points(entry_prem, exit_prem):
    """Round trip for ONE straddle, in NIFTY points. Same model as engine45."""
    slip = SLIP * (entry_prem + exit_prem)
    stt = 0.0010 * entry_prem
    txn = 0.0005 * (entry_prem + exit_prem)
    brok = (20.0 * 4) / QTY
    return slip + stt + txn + brok + 0.18 * (txn + brok)


def true_exit(sessions, expiry, dte_out=21):
    """The 21-DTE exit session, computed from the EXPIRY.

    trades_daily.csv is the OUTPUT of the stopped book: for the 3 campaigns that
    hit the stop or the target its exit_date is that event's date, not the 21-DTE
    date. Taking it at face value would truncate the HOLD baseline on exactly the
    campaigns the management arms are about, and would hide the trigger day from
    the scan. The window is therefore rebuilt from the expiry for every arm.
    """
    e = date(*map(int, expiry.split("-")))
    tgt = (e - timedelta(days=dte_out)).isoformat()
    cand = [d for d in sessions if d <= tgt]
    return cand[-1] if cand else None


def load_market():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    spot = {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='NIFTY50' "
        "AND timeframe='day' AND close IS NOT NULL")}
    vx = sorted((r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' AND close IS NOT NULL"))
    sess = sorted(spot)
    return con, spot, vx, sess


def vix_rank(vx, day):
    idx = [i for i, (d, _) in enumerate(vx) if d <= day]
    if not idx:
        return None
    i = idx[-1]
    if i < 253:
        return None
    w = [v for _, v in vx[i - 252:i]]
    return 100.0 * sum(1 for x in w if x < vx[i][1]) / len(w)


def chains_for(con, expiry, d0, d1):
    """Whole daily chain for one expiry over the campaign window, in one query."""
    out = defaultdict(dict)
    for td, K, ot, c, v, oi in con.execute(
            "SELECT trade_date, strike, option_type, close, contracts, open_interest "
            "FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? "
            "AND trade_date>=? AND trade_date<=?", (expiry, d0, d1)):
        out[td[:10]][(float(K), ot)] = (float(c or 0), int(v or 0), int(oi or 0))
    return out


def straddle(ch, K, need_liquid=False):
    ce, pe = ch.get((float(K), "CE")), ch.get((float(K), "PE"))
    if not ce or not pe or ce[0] <= 0 or pe[0] <= 0:
        return None
    if need_liquid and (min(ce[1], pe[1]) < MIN_VOL or min(ce[2], pe[2]) < MIN_OI):
        return None
    return ce[0] + pe[0]


def run_campaign(camp, arm, T, chains, spot):
    """One campaign under one arm. Returns (net_pts, peak_units, fired, tag)."""
    K0, C0 = float(camp["strike"]), float(camp["credit"])
    days = sorted(d for d in chains if camp["entry_date"] < d <= camp["xd"])
    if not days:
        return None
    book = [dict(K=K0, credit=C0)]          # open straddles
    realised, cost = 0.0, 0.0
    fired, peak = 0, 1
    tag = "RAN_TO_21DTE"

    for i, d in enumerate(days):
        ch = chains[d]
        last = (i == len(days) - 1)
        prices = [straddle(ch, p["K"]) for p in book]
        if any(p is None for p in prices):
            continue
        if book and fired < CAP and arm != "HOLD" and not (last and arm.startswith("ADD")):
            ratio = sum(prices) / sum(p["credit"] for p in book)
            if ratio >= T:
                S = spot.get(d)
                if S is None:
                    continue
                if arm in ("STOP", "RECENTRE"):
                    for p, px in zip(book, prices):
                        realised += p["credit"] - px
                        cost += costs_points(p["credit"], px)
                    book = []
                    fired += 1
                    tag = "STOPPED"
                    if arm == "RECENTRE":
                        Kn = round(S / STEP) * STEP
                        pn = straddle(ch, Kn, need_liquid=True)
                        if pn:
                            book = [dict(K=float(Kn), credit=pn)]
                            tag = "RECENTRED"
                        else:
                            tag = "RECENTRE_NO_FILL"
                    if not book:
                        break
                else:                                   # ADD_ATM / ADD_MIRROR
                    Kn = (round(S / STEP) * STEP if arm == "ADD_ATM"
                          else round((2.0 * S - K0) / STEP) * STEP)
                    pn = straddle(ch, Kn, need_liquid=True)
                    if pn:
                        book.append(dict(K=float(Kn), credit=pn))
                        fired += 1
                        peak = max(peak, len(book))
                        tag = "ADDED"
                    else:
                        tag = "ADD_NO_FILL"

    for p in book:                                       # close what is left at 21 DTE
        px = None
        for d in reversed(days):
            px = straddle(chains[d], p["K"])
            if px is not None:
                break
        if px is None:
            return None
        realised += p["credit"] - px
        cost += costs_points(p["credit"], px)
    return realised - cost, peak, fired, tag


def stats(xs):
    n = len(xs)
    if n < 2:
        return (float("nan"),) * 3
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5
    return mu, sd, (mu / (sd / n ** 0.5)) if sd else float("nan")


def maxdd(xs):
    eq = peak = dd = 0.0
    for x in xs:
        eq += x
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return dd


def main():
    con, spot, vx, sess = load_market()
    camps = list(csv.DictReader(open(TRADES)))
    trunc = 0
    for c in camps:
        c["vix_rank"] = vix_rank(vx, c["entry_date"])
        c["xd"] = true_exit(sess, c["expiry"]) or c["exit_date"]
        if c["xd"] != c["exit_date"]:
            trunc += 1
    print("campaigns: %d total (%d had a truncated exit_date in the CSV -> rebuilt "
          "from expiry so HOLD really holds)" % (len(camps), trunc))

    pre = {}
    for c in camps:
        key = (c["expiry"], c["entry_date"], c["xd"])
        if key not in pre:
            pre[key] = chains_for(con, c["expiry"], c["entry_date"], c["xd"])
    print("chains preloaded for %d campaigns\n" % len(pre))

    arms = ["HOLD", "STOP", "RECENTRE", "ADD_ATM", "ADD_MIRROR"]
    rows = []
    for T in TRIGGERS:
        for arm in arms:
            if arm == "HOLD" and T != TRIGGERS[0]:
                continue
            res = []
            for c in camps:
                ch = pre[(c["expiry"], c["entry_date"], c["xd"])]
                r = run_campaign(c, arm, T, ch, spot)
                if r:
                    res.append((c, r))
            if not res:
                continue
            for scope, sel in (("ALL", lambda c: True),
                               ("VIX>25", lambda c: (c["vix_rank"] or 0) > 25)):
                sub = [(c, r) for c, r in res if sel(c)]
                if len(sub) < 5:
                    continue
                nets = [r[0] for _, r in sub]
                peaks = [r[1] for _, r in sub]
                norm = [r[0] / r[1] for _, r in sub]
                fires = sum(1 for _, r in sub if r[3] in
                            ("STOPPED", "RECENTRED", "ADDED"))
                mu, sd, t = stats(nets)
                nmu, _, nt = stats(norm)
                rows.append(dict(
                    trigger=int(T * 100), arm=arm, scope=scope, n=len(sub),
                    fires=fires, net=round(mu, 2), t=round(t, 2),
                    norm_net=round(nmu, 2), norm_t=round(nt, 2),
                    win=round(100.0 * sum(1 for x in nets if x > 0) / len(nets), 1),
                    maxdd=round(maxdd(nets), 1),
                    peak_units=round(sum(peaks) / len(peaks), 2),
                    total=round(sum(nets), 1)))

    hdr = ("%-6s %-11s %-7s %4s %5s %9s %6s %10s %6s %6s %9s %6s"
           % ("trig", "arm", "scope", "n", "fires", "net/camp", "t",
              "norm net", "normT", "win%", "maxDD", "units"))
    for scope in ("VIX>25", "ALL"):
        print("=" * len(hdr))
        print("SCOPE: %s   %s" % (scope, "(the LIVE ruleset)" if scope == "VIX>25" else ""))
        print("=" * len(hdr))
        print(hdr)
        for r in rows:
            if r["scope"] != scope:
                continue
            print("%-6s %-11s %-7s %4d %5d %9.1f %6.2f %10.1f %6.2f %6.1f %9.1f %6.2f"
                  % (("%d%%" % r["trigger"]) if r["arm"] != "HOLD" else "-",
                     r["arm"], r["scope"], r["n"], r["fires"], r["net"], r["t"],
                     r["norm_net"], r["norm_t"], r["win"], r["maxdd"], r["peak_units"]))
        print()

    with open(os.path.join(RES, "phase_g_management.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % os.path.join(RES, "phase_g_management.csv"))
    base = [r for r in rows if r["arm"] == "HOLD" and r["scope"] == "VIX>25"]
    if base:
        b = base[0]
        print("\nBASELINE (VIX>25, HOLD): net %.1f pts/campaign, t %.2f, maxDD %.1f"
              % (b["net"], b["t"], b["maxdd"]))
        better = [r for r in rows if r["scope"] == "VIX>25"
                  and r["arm"] != "HOLD" and r["norm_net"] > b["net"]]
        print("arms beating it on SIZE-NORMALISED net: %s"
              % (", ".join("%s@%d%%" % (r["arm"], r["trigger"]) for r in better) or "NONE"))


if __name__ == "__main__":
    sys.exit(main())
