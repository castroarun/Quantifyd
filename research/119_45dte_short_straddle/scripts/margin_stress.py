#!/usr/bin/env python3
"""
Margin is not static over a trade's life. Two effects, measured separately.

The previous table was a CROSS-SECTION: seven different contracts priced at one
instant. It cannot show what happens to a single position as it ages, which is the
thing that actually determines how much capital a 45->21 DTE hold must reserve.

We cannot time-travel a contract, and historical margin is not stored anywhere. But
the two drivers can both be measured right now:

  A) TENOR      how margin differs by DTE, holding strike fixed at today's ATM.
  B) MONEYNESS  how margin moves when the position goes against you - i.e. the
                straddle's strike is now far from spot. This is the effect that
                actually bites, because margin inflates in exactly the scenario
                that is also draining the P&L. Simulated by pricing the SAME
                straddle structure at strikes 1/2/3/5/7% away from spot.

Peak margin over the hold, not entry margin, is what the sleeve must reserve.

READ ONLY. Places no orders.
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect

LOT = 65
CAPITAL = 2_148_956.0
DD_PER_LOT = 998.4 * 65          # real-price MaxDD per lot


def basket(ce, pe, lots=1, product="NRML"):
    q = LOT * lots
    return [dict(exchange="NFO", tradingsymbol=ce, transaction_type="SELL",
                 variety="regular", product=product, order_type="MARKET", quantity=q),
            dict(exchange="NFO", tradingsymbol=pe, transaction_type="SELL",
                 variety="regular", product=product, order_type="MARKET", quantity=q)]


def main():
    tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(tok["access_token"])
    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    atm = int(round(spot / 50.0) * 50)
    today = date.today()
    print("NIFTY %.2f  ATM %d  %s\n" % (spot, atm, today))

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by_exp = {}
    for i in ins:
        by_exp.setdefault(str(i["expiry"]), {})[(i["strike"], i["instrument_type"])] = i

    def marg(exp, strike, lots=1):
        legs = by_exp.get(exp, {})
        ce, pe = legs.get((float(strike), "CE")), legs.get((float(strike), "PE"))
        if not ce or not pe:
            return None
        try:
            m = k.basket_order_margins(basket(ce["tradingsymbol"], pe["tradingsymbol"], lots),
                                       consider_positions=False, mode="compact")
            return m["initial"]["total"]
        except Exception:
            return None

    exps = sorted(e for e in by_exp
                  if 0 <= (date(*map(int, e.split("-"))) - today).days <= 70)

    # ---------------- A. tenor, strike fixed at ATM ------------------------
    print("A) TENOR - same strike (%d), different expiries" % atm)
    print("%-12s %5s %14s %10s" % ("expiry", "DTE", "margin/lot", "vs 22-DTE"))
    ref = None
    tenor = []
    for e in exps:
        d = (date(*map(int, e.split("-"))) - today).days
        m = marg(e, atm)
        if m is None:
            continue
        tenor.append((d, e, m))
        if 20 <= d <= 24:
            ref = m
    for d, e, m in tenor:
        rel = ("%+.1f%%" % (100 * (m / ref - 1))) if ref else "-"
        print("%-12s %5d %14s %10s" % (e, d, "{:,.0f}".format(m), rel))

    # ---------------- B. moneyness, tenor fixed ----------------------------
    print("\nB) MONEYNESS - the position moves against you (tenor held fixed)")
    for want in (22, 36):
        cand = [t for t in tenor if abs(t[0] - want) <= 4]
        if not cand:
            continue
        d, e, base = cand[0]
        print("\n   expiry %s (DTE %d) - straddle struck away from spot:" % (e, d))
        print("   %-14s %8s %14s %10s %12s" % ("strike", "move", "margin/lot", "vs ATM", "max lots"))
        for pct in (0, 0.01, 0.02, 0.03, 0.05, 0.07):
            for sgn in ((1,) if pct == 0 else (1, -1)):
                strike = int(round(spot * (1 + sgn * pct) / 50.0) * 50)
                m = marg(e, strike)
                if m is None:
                    continue
                lots = int(CAPITAL // (m + 2 * DD_PER_LOT))
                print("   %-14d %7s%% %14s %9s%% %12d"
                      % (strike, ("%+.0f" % (100 * sgn * pct)) if pct else "0",
                         "{:,.0f}".format(m), "%+.1f" % (100 * (m / base - 1)), lots))

    # ---------------- C. what the sleeve must actually reserve -------------
    print("\nC) SIZING on PEAK margin across the 21-45 DTE hold window")
    band = [t for t in tenor if 21 <= t[0] <= 50]
    if band:
        peak = max(band, key=lambda t: t[2])
        print("   contracts in band: " + ", ".join("DTE %d Rs %s" % (d, "{:,.0f}".format(m))
                                                   for d, e, m in band))
        print("   PEAK margin/lot in the hold window: Rs %s (DTE %d)"
              % ("{:,.0f}".format(peak[2]), peak[0]))
        for label, m in (("entry-margin sizing", band[0][2]), ("PEAK-margin sizing", peak[2])):
            lots = int(CAPITAL // (m + 2 * DD_PER_LOT))
            print("   %-22s -> %d lots  (Rs %s margin + Rs %s buffer per lot)"
                  % (label, lots, "{:,.0f}".format(m), "{:,.0f}".format(2 * DD_PER_LOT)))
    json.dump(dict(spot=spot, atm=atm, tenor=tenor), open("/tmp/margin_stress.json", "w"), indent=1)
    print("\nwrote /tmp/margin_stress.json")


if __name__ == "__main__":
    main()
