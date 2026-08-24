#!/usr/bin/env python3
"""
Per-DTE margin for a NIFTY short ATM straddle, straight from Kite.

Margin is not persisted anywhere in this repo — the live executors call Kite at
order time and never store the answer. So we ask Kite the same question they do,
via basket_order_margins, for every listed NIFTY expiry.

  consider_positions=False  -> the STANDALONE requirement for a fresh book, which
                               is what a new 45-DTE sleeve would actually block.
                               (True would net against the live NAS/CSL positions
                               and flatter the number.)
  NRML  -> the correct product for a 45->21 DTE positional straddle.
  MIS   -> what the live NAS book uses (intraday only); shown for comparison
           because that is the "live systems" reference point.

READ ONLY. Places no orders.
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect

LOT = 65
CAPITAL = 2_148_956.0        # Rs 15L margin + 2 x MaxDD(5 lots, real prices Rs 3,24,478)


def straddle_basket(tsym_ce, tsym_pe, lots, product):
    q = LOT * lots
    return [
        dict(exchange="NFO", tradingsymbol=tsym_ce, transaction_type="SELL",
             variety="regular", product=product, order_type="MARKET", quantity=q),
        dict(exchange="NFO", tradingsymbol=tsym_pe, transaction_type="SELL",
             variety="regular", product=product, order_type="MARKET", quantity=q),
    ]


def main():
    tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(tok["access_token"])

    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    atm = int(round(spot / 50.0) * 50)
    today = date.today()
    print("NIFTY spot %.2f  -> ATM strike %d   (as of %s)" % (spot, atm, today))
    print("lot %d | capital Rs %.2fL\n" % (LOT, CAPITAL / 1e5))

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by_exp = {}
    for i in ins:
        by_exp.setdefault(str(i["expiry"]), {})[(i["strike"], i["instrument_type"])] = i

    rows = []
    for exp in sorted(by_exp):
        legs = by_exp[exp]
        ce = legs.get((float(atm), "CE"))
        pe = legs.get((float(atm), "PE"))
        if not ce or not pe:
            continue
        dte = (date(*map(int, exp.split("-"))) - today).days
        if dte < 0 or dte > 70:
            continue
        out = dict(expiry=exp, dte=dte)
        for product in ("NRML", "MIS"):
            try:
                m = k.basket_order_margins(
                    straddle_basket(ce["tradingsymbol"], pe["tradingsymbol"], 1, product),
                    consider_positions=False, mode="compact")
                tot = m["initial"]["total"]
                out[product] = tot
            except Exception as e:
                out[product] = None
                out[product + "_err"] = str(e)[:60]
        rows.append(out)

    print("%-12s %4s %14s %8s %14s %8s" %
          ("expiry", "DTE", "NRML /lot", "max lots", "MIS /lot", "max lots"))
    print("-" * 68)
    for r in rows:
        n, m = r.get("NRML"), r.get("MIS")
        ln = "Rs %10s" % ("{:,.0f}".format(n)) if n else "  n/a"
        lm = "Rs %10s" % ("{:,.0f}".format(m)) if m else "  n/a"
        mn = int(CAPITAL // n) if n else 0
        mm = int(CAPITAL // m) if m else 0
        print("%-12s %4d %14s %8d %14s %8d" % (r["expiry"], r["dte"], ln, mn, lm, mm))

    # linearity check — does 5 lots cost exactly 5x 1 lot?
    print("\nlinearity check (NRML, nearest expiry with DTE>=21):")
    tgt = next((r for r in rows if r["dte"] >= 21), rows[-1] if rows else None)
    if tgt:
        legs = by_exp[tgt["expiry"]]
        ce, pe = legs[(float(atm), "CE")], legs[(float(atm), "PE")]
        for lots in (1, 2, 5, 10):
            m = k.basket_order_margins(
                straddle_basket(ce["tradingsymbol"], pe["tradingsymbol"], lots, "NRML"),
                consider_positions=False, mode="compact")
            t = m["initial"]["total"]
            print("   %2d lot(s): Rs %10s   (per lot Rs %9s)"
                  % (lots, "{:,.0f}".format(t), "{:,.0f}".format(t / lots)))

    json.dump(rows, open("/tmp/margin_by_dte.json", "w"), indent=1)
    print("\nwrote /tmp/margin_by_dte.json")


if __name__ == "__main__":
    main()
