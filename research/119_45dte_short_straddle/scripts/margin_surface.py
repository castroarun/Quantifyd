#!/usr/bin/env python3
"""
Re-measure the margin surface AND the implied-vol surface in one consistent snapshot.

The first calibration failed its gate (RMS 12.0% vs a 10% limit) with a clearly
structured error: far strikes under-predicted by up to 24.6% and the 64-DTE point by
17.0%. Cause is identified, not guessed — every strike was priced at a flat ATM IV
(India VIX 11.49), so the wings and the long tenor were mispriced, and the SPAN
scenario losses came out too small.

The fix is to supply the real volatility smile rather than to loosen the gate. This
script captures, at one instant and for the same (expiry, strike) grid:

  * margin per lot   — Kite basket_order_margins, consider_positions=False
  * CE and PE LTP    — from which the straddle's own implied vol is backed out

so the recalibration runs on measured margins AND measured vols from the same moment.

READ ONLY. Places no orders.
"""
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kiteconnect import KiteConnect
from engine45 import implied_forward, implied_vol_straddle

LOT = 65
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "margin_surface.json")


def main():
    tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(tok["access_token"])
    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    atm = int(round(spot / 50.0) * 50)
    today = date.today()
    print("spot %.2f  ATM %d  %s" % (spot, atm, today))

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by = {}
    for i in ins:
        by.setdefault(str(i["expiry"]), {})[(i["strike"], i["instrument_type"])] = i

    exps = sorted(e for e in by
                  if 0 <= (date(*map(int, e.split("-"))) - today).days <= 70)
    pts = []
    for exp in exps:
        dte = (date(*map(int, exp.split("-"))) - today).days
        if dte < 1:
            continue
        # ATM for every tenor; a moneyness ladder on the two mid tenors
        offs = [0.0]
        if 18 <= dte <= 45:
            offs = [0.0, .01, -.01, .02, -.02, .03, -.03, .05, -.05, .07, -.07]
        for o in offs:
            K = int(round(spot * (1 + o) / 50.0) * 50)
            ce = by[exp].get((float(K), "CE"))
            pe = by[exp].get((float(K), "PE"))
            if not ce or not pe:
                continue
            try:
                q = k.ltp(["NFO:" + ce["tradingsymbol"], "NFO:" + pe["tradingsymbol"]])
                cp = q["NFO:" + ce["tradingsymbol"]]["last_price"]
                pp = q["NFO:" + pe["tradingsymbol"]]["last_price"]
                if cp <= 0 or pp <= 0:
                    continue
                m = k.basket_order_margins(
                    [dict(exchange="NFO", tradingsymbol=t, transaction_type="SELL",
                          variety="regular", product="NRML", order_type="MARKET",
                          quantity=LOT)
                     for t in (ce["tradingsymbol"], pe["tradingsymbol"])],
                    consider_positions=False, mode="compact")["initial"]["total"]
            except Exception as e:
                print("   skip %s %d: %s" % (exp, K, str(e)[:40]))
                continue
            T = max(dte, 1) / 365.0
            F = implied_forward(cp, pp, K, T)
            iv = implied_vol_straddle(cp + pp, F, K, T)
            if not iv:
                continue
            pts.append(dict(expiry=exp, dte=dte, strike=K, ce=cp, pe=pp,
                            straddle=cp + pp, iv=iv, margin=m))
            print("  dte %3d  K %6d  straddle %8.2f  IV %5.1f%%  margin %s"
                  % (dte, K, cp + pp, 100 * iv, "{:,.0f}".format(m)))

    json.dump(dict(spot=spot, atm=atm, asof=str(today), points=pts),
              open(OUT, "w"), indent=1)
    print("\n%d points -> %s" % (len(pts), OUT))
    ivs = [p["iv"] for p in pts]
    print("IV range across the grid: %.1f%% to %.1f%%  <- the smile the flat-VIX model missed"
          % (100 * min(ivs), 100 * max(ivs)))


if __name__ == "__main__":
    main()
