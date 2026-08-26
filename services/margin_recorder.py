#!/usr/bin/env python3
"""
Daily margin recorder — builds REAL margin history from today forward.

Reconstructing historical SPAN failed its calibration gate (RMS 12.0% vs a 10%
limit) and cannot be rescued: the wing strikes needed to calibrate the moneyness
response carry ZERO open interest and zero volume, so no market vol exists there
to measure, and NSE's .spn parameter files are not publicly reachable. Rather than
tune parameters until the gate passes — which would be fitting to the gate — this
records the real thing going forward.

Each run stores, for one timestamp:
  * account SPAN and exposure actually utilised (kite.margins)
  * the margin a REFERENCE 3-lot ATM straddle would require right now, on the
    front monthly — a clean series that exists whether or not the book has a
    position, so margin-vs-volatility can be tracked continuously
  * the margin the BOOK's own open position requires, when it has one
  * India VIX level and its 252-session percentile rank, and NIFTY spot

Within a few months this answers the question the reconstruction could not:
how does this book's margin actually move as volatility moves.

READ ONLY against the broker. Places no orders.
"""
import json
import os
import sqlite3
import sys
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "backtest_data", "straddle45_margin_log.db")
PAPER_DB = os.path.join(ROOT, "backtest_data", "straddle45_paper.db")
MKT = os.path.join(ROOT, "backtest_data", "market_data.db")
LOT, LOTS = 65, 3


def init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS margin_log (
        ts TEXT, d TEXT, spot REAL, vix REAL, vix_rank REAL,
        ref_expiry TEXT, ref_strike REAL, ref_dte INTEGER, ref_margin_per_lot REAL,
        book_expiry TEXT, book_strike REAL, book_dte INTEGER, book_margin_total REAL,
        acct_span REAL, acct_exposure REAL, acct_net REAL, acct_available REAL,
        UNIQUE(d))""")
    con.commit()
    return con


def vix_and_rank():
    con = sqlite3.connect("file:%s?mode=ro" % MKT, uri=True)
    vx = [(r[0][:10], float(r[1])) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='INDIAVIX' "
        "AND timeframe='day' ORDER BY date") if r[1]]
    con.close()
    if len(vx) < 253:
        return None, None
    lvl = vx[-1][1]
    w = [v for _, v in vx[-253:-1]]
    return lvl, 100.0 * sum(1 for x in w if x < lvl) / len(w)


def basket(k, ce, pe, lots):
    return [dict(exchange="NFO", tradingsymbol=t, transaction_type="SELL",
                 variety="regular", product="NRML", order_type="MARKET",
                 quantity=LOT * lots) for t in (ce, pe)]


def main():
    from kiteconnect import KiteConnect
    tok = json.load(open(os.path.join(ROOT, "backtest_data", "access_token.json")))
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(tok["access_token"])

    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    atm = int(round(spot / 50.0) * 50)
    vix, rank = vix_and_rank()
    today = date.today()

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by = {}
    for i in ins:
        by.setdefault(str(i["expiry"]), {})[(i["strike"], i["instrument_type"])] = i

    # reference: the nearest monthly at least 21 days out, struck ATM
    monthlies = sorted(e for e in by
                       if (date(*map(int, e.split("-"))) - today).days >= 21)
    ref = ref_m = ref_dte = None
    for e in monthlies:
        ce, pe = by[e].get((float(atm), "CE")), by[e].get((float(atm), "PE"))
        if ce and pe:
            try:
                ref_m = k.basket_order_margins(
                    basket(k, ce["tradingsymbol"], pe["tradingsymbol"], 1),
                    consider_positions=False, mode="compact")["initial"]["total"]
                ref, ref_dte = e, (date(*map(int, e.split("-"))) - today).days
                break
            except Exception:
                continue

    # the book's own open position, if any
    bexp = bstrike = bdte = bmar = None
    if os.path.exists(PAPER_DB):
        p = sqlite3.connect("file:%s?mode=ro" % PAPER_DB, uri=True)
        row = p.execute("SELECT expiry, strike FROM trades WHERE status='OPEN' "
                        "ORDER BY entry_date DESC LIMIT 1").fetchone()
        p.close()
        if row:
            bexp, bstrike = row[0], row[1]
            ce = by.get(bexp, {}).get((float(bstrike), "CE"))
            pe = by.get(bexp, {}).get((float(bstrike), "PE"))
            if ce and pe:
                try:
                    bmar = k.basket_order_margins(
                        basket(k, ce["tradingsymbol"], pe["tradingsymbol"], LOTS),
                        consider_positions=False, mode="compact")["initial"]["total"]
                    bdte = (date(*map(int, bexp.split("-"))) - today).days
                except Exception:
                    pass

    m = k.margins("equity")
    u = m.get("utilised", {})
    con = init()
    con.execute("INSERT OR REPLACE INTO margin_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(today), spot, vix, rank,
                 ref, float(atm) if ref else None, ref_dte, ref_m,
                 bexp, bstrike, bdte, bmar,
                 u.get("span"), u.get("exposure"), m.get("net"),
                 m.get("available", {}).get("cash")))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM margin_log").fetchone()[0]
    con.close()

    print("%s  spot %.2f  VIX %.2f (rank %.0f)" % (today, spot, vix or 0, rank or 0))
    print("  reference 1-lot ATM straddle %s (dte %s): Rs %s per lot"
          % (ref, ref_dte, "{:,.0f}".format(ref_m) if ref_m else "n/a"))
    if bmar:
        print("  BOOK position %s @ %.0f (dte %s), %d lots: Rs %s"
              % (bexp, bstrike, bdte, LOTS, "{:,.0f}".format(bmar)))
    print("  account span Rs %s | exposure Rs %s"
          % ("{:,.0f}".format(u.get("span") or 0), "{:,.0f}".format(u.get("exposure") or 0)))
    print("  %d day(s) of real margin history recorded" % n)


if __name__ == "__main__":
    main()
