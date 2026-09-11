#!/usr/bin/env python3
"""Phase H — does it matter WHEN in the entry session you sell?

Arun wants to take today's campaign now (11:10) rather than wait for the 15:20
window. Every one of the 89 backtested campaigns was struck at the CLOSE, so an
intraday entry has no evidence behind it. This measures the question the data
CAN answer.

What cannot be tested, stated first: expired-contract intraday option prices are
unobtainable from Kite, and our own 1-minute recorder only covers a contract from
~27 DTE, so a 45-DTE entry has no recorded intraday history. There is no way to
price an 11:10 fill historically.

What CAN be tested: NSE's bhavcopy carries OPEN as well as CLOSE for every
contract. The open is the session's first trade (~09:15) and the close is the
tested convention (15:30), so OPEN-vs-CLOSE brackets any intraday entry time.
If the two arms are indistinguishable, entry timing within the day does not
matter and taking the trade early is harmless. If the open is materially worse,
waiting for the close is the tested rule AND the better one.

Arms, campaign (45 -> 21 DTE) as the unit, n = 89 / 61 on the VIX>25 book:

  CLOSE  strike from the close spot, credit from option CLOSES   <- the tested rule
  OPEN   strike from the OPEN spot,  credit from option OPENS    <- entering early

Exits are identical in both arms - the same 21-DTE close - so the only thing that
differs is what you were paid at entry. Both legs must carry real volume and OI.

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
MIN_VOL, MIN_OI = 100, 500
DTE_OUT = 21


def costs_points(entry_prem, exit_prem):
    slip = SLIP * (entry_prem + exit_prem)
    stt = 0.0010 * entry_prem
    txn = 0.0005 * (entry_prem + exit_prem)
    brok = (20.0 * 4) / QTY
    return slip + stt + txn + brok + 0.18 * (txn + brok)


def roll_off_weekend(d):
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def load(con):
    spot = {}
    for d, o, c in con.execute(
            "SELECT date, open, close FROM market_data_unified WHERE symbol='NIFTY50' "
            "AND timeframe='day' AND close IS NOT NULL"):
        spot[d[:10]] = (float(o) if o else None, float(c))
    return spot


def chain(con, expiry, day):
    """{(strike, type): (open, close, vol, oi)} for one expiry on one session."""
    out = {}
    for K, ot, o, c, v, oi in con.execute(
            "SELECT strike, option_type, open, close, contracts, open_interest "
            "FROM nse_options_bhav WHERE symbol='NIFTY' AND expiry_date=? AND trade_date=?",
            (expiry, day)):
        out[(float(K), ot)] = (float(o or 0), float(c or 0), int(v or 0), int(oi or 0))
    return out


def straddle(ch, K, field, need_liquid=True):
    """field 0 = open, 1 = close."""
    ce, pe = ch.get((float(K), "CE")), ch.get((float(K), "PE"))
    if not ce or not pe:
        return None
    a, b = ce[field], pe[field]
    if a <= 0 or b <= 0:
        return None
    if need_liquid and (min(ce[2], pe[2]) < MIN_VOL or min(ce[3], pe[3]) < MIN_OI):
        return None
    return a + b


def stats(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5
    return mu, (mu / (sd / n ** 0.5)) if sd else float("nan")


def maxdd(xs):
    eq = peak = dd = 0.0
    for x in xs:
        eq += x
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    return dd


def main():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    spot = load(con)
    sess = sorted(spot)
    camps = list(csv.DictReader(open(TRADES)))

    vx = sorted((r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' AND close IS NOT NULL"))

    def vrank(day):
        idx = [i for i, (d, _) in enumerate(vx) if d <= day]
        if not idx or idx[-1] < 253:
            return None
        i = idx[-1]
        w = [v for _, v in vx[i - 252:i]]
        return 100.0 * sum(1 for x in w if x < vx[i][1]) / len(w)

    rows = []
    for c in camps:
        ed, exp = c["entry_date"], c["expiry"]
        xd = roll_off_weekend(date(*map(int, exp.split("-"))) - timedelta(days=DTE_OUT)).isoformat()
        xs = [d for d in sess if d <= xd]
        xd = xs[-1] if xs else None
        if not xd or xd <= ed:
            continue
        ch_e, ch_x = chain(con, exp, ed), chain(con, exp, xd)
        if not ch_e or not ch_x:
            continue
        o_sp, c_sp = spot.get(ed, (None, None))
        if not o_sp or not c_sp:
            continue
        K_close = round(c_sp / STEP) * STEP
        K_open = round(o_sp / STEP) * STEP
        cred_c = straddle(ch_e, K_close, 1)
        cred_o = straddle(ch_e, K_open, 0)
        if cred_c is None or cred_o is None:
            continue
        ex_c = straddle(ch_x, K_close, 1, need_liquid=False)
        ex_o = straddle(ch_x, K_open, 1, need_liquid=False)
        if ex_c is None or ex_o is None:
            continue
        net_c = (cred_c - ex_c) - costs_points(cred_c, ex_c)
        net_o = (cred_o - ex_o) - costs_points(cred_o, ex_o)
        rows.append(dict(entry=ed, expiry=exp, vrank=vrank(ed),
                         K_open=K_open, K_close=K_close,
                         open_spot=o_sp, close_spot=c_sp,
                         cred_open=cred_o, cred_close=cred_c,
                         net_open=net_o, net_close=net_c))
    print("campaigns priced both ways: %d of %d" % (len(rows), len(camps)))

    for scope, sel in (("VIX>25 (the live book)", lambda r: (r["vrank"] or 0) > 25),
                       ("ALL", lambda r: True)):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 5:
            continue
        nc = [r["net_close"] for r in sub]
        no = [r["net_open"] for r in sub]
        d = [r["net_open"] - r["net_close"] for r in sub]
        mc, tc = stats(nc)
        mo, to = stats(no)
        md, td = stats(d)
        print("\n" + "=" * 78)
        print("SCOPE: %s   n=%d" % (scope, len(sub)))
        print("=" * 78)
        print("  %-26s %10s %8s %9s %9s" % ("arm", "net/camp", "t", "win%", "maxDD"))
        print("  %-26s %10.1f %8.2f %8.1f%% %9.1f"
              % ("CLOSE (the tested rule)", mc, tc,
                 100.0 * sum(1 for x in nc if x > 0) / len(nc), maxdd(nc)))
        print("  %-26s %10.1f %8.2f %8.1f%% %9.1f"
              % ("OPEN  (entering early)", mo, to,
                 100.0 * sum(1 for x in no if x > 0) / len(no), maxdd(no)))
        print("  %-26s %10.1f %8.2f   %s"
              % ("PAIRED DIFF (open-close)", md, td,
                 "open is BETTER" if md > 0 else "open is WORSE"))
        print("  open beat close on %d of %d campaigns" % (sum(1 for x in d if x > 0), len(d)))
        cdiff = [r["cred_open"] - r["cred_close"] for r in sub]
        mcd, tcd = stats(cdiff)
        print("  credit collected: open %.1f vs close %.1f pts (diff %+.1f, t %+.2f)"
              % (sum(r["cred_open"] for r in sub) / len(sub),
                 sum(r["cred_close"] for r in sub) / len(sub), mcd, tcd))
        ks = sum(1 for r in sub if r["K_open"] != r["K_close"])
        print("  ATM strike differed between open and close on %d of %d campaigns (%.0f%%)"
              % (ks, len(sub), 100.0 * ks / len(sub)))

    with open(os.path.join(RES, "phase_h_entrytime.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % os.path.join(RES, "phase_h_entrytime.csv"))


if __name__ == "__main__":
    sys.exit(main())
