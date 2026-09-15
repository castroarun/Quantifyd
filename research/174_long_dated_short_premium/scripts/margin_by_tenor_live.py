#!/usr/bin/env python3
"""
research/174 — MEASURED (not modelled) margin per tenor, straight from Kite.

Margin is the binding constraint for a long-dated short, not P&L: a 240-day short
straddle blocks capital for eight months. Nothing in this repo persists margin, so we
ask Kite the same question the live executors ask at order time, for EVERY listed NIFTY
expiry (no DTE cap — that cap is what r/119's version had) and for each structure this
study cares about.

  consider_positions=False -> the STANDALONE requirement a fresh sleeve would block.
                              True nets against the live NAS/CSL book and flatters it.
  NRML                     -> the correct product for a positional short.

Leg-ordering trap (inherited from r/119's margin_stress_live): submit the LONG wings
FIRST in the basket, otherwise Kite prices the naked shorts alone and returns a
requirement that the hedged structure never actually needs.

READ ONLY. Places no orders.
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect                          # noqa: E402

LOT = 65
OUT = Path(__file__).resolve().parent.parent / "results" / "margin_by_tenor.json"


def leg(tsym, ttype, lots, product="NRML"):
    return dict(exchange="NFO", tradingsymbol=tsym, transaction_type=ttype,
                variety="regular", product=product, order_type="MARKET",
                quantity=LOT * lots)


def main():
    tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(tok["access_token"])

    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    today = date.today()
    print("NIFTY spot %.2f   %s   lot %d" % (spot, today, LOT))

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by_exp = {}
    for i in ins:
        by_exp.setdefault(str(i["expiry"]), {})[(float(i["strike"]), i["instrument_type"])] = i

    rows = []
    for exp in sorted(by_exp):
        legs = by_exp[exp]
        dte = (date(*map(int, exp.split("-"))) - today).days
        if dte < 1:
            continue
        strikes = sorted({s for s, _ in legs})
        if not strikes:
            continue

        def near(x, side):
            cands = [s for s in strikes if (s, side) in legs]
            return min(cands, key=lambda s: abs(s - x)) if cands else None

        atm = near(spot, "CE")
        if atm is None or (atm, "PE") not in legs:
            continue
        row = dict(expiry=exp, dte=dte, atm=atm, n_strikes=len(strikes),
                   strike_step=(strikes[1] - strikes[0]) if len(strikes) > 1 else None)

        baskets = {}
        baskets["STR"] = [leg(legs[(atm, "CE")]["tradingsymbol"], "SELL", 1),
                          leg(legs[(atm, "PE")]["tradingsymbol"], "SELL", 1)]
        # strangles at every body width the study sweeps - a 5% strangle blocks less
        # than a 2.5% one, so reusing one measurement across widths understates the
        # wider structures' capital efficiency.
        for b in (0.015, 0.025, 0.035, 0.05):
            kc, kp = near(spot * (1 + b), "CE"), near(spot * (1 - b), "PE")
            if kc and kp and kc > kp:
                baskets["STG%.1f" % (b * 100)] = [
                    leg(legs[(kc, "CE")]["tradingsymbol"], "SELL", 1),
                    leg(legs[(kp, "PE")]["tradingsymbol"], "SELL", 1)]
        # iron condors: LONG WINGS FIRST in the basket. If the shorts lead, Kite prices
        # the naked shorts alone and returns a requirement the hedged structure never
        # needs (leg-ordering trap from r/119 margin_stress_live).
        for b in (0.025, 0.05):
            for wd in (0.03, 0.05, 0.07):
                kc, kp = near(spot * (1 + b), "CE"), near(spot * (1 - b), "PE")
                wc, wp = near(spot * (1 + b + wd), "CE"), near(spot * (1 - b - wd), "PE")
                if not (kc and kp and wc and wp and wc > kc > kp > wp):
                    continue
                baskets["IC%.1fw%.0f" % (b * 100, wd * 100)] = [
                    leg(legs[(wc, "CE")]["tradingsymbol"], "BUY", 1),
                    leg(legs[(wp, "PE")]["tradingsymbol"], "BUY", 1),
                    leg(legs[(kc, "CE")]["tradingsymbol"], "SELL", 1),
                    leg(legs[(kp, "PE")]["tradingsymbol"], "SELL", 1)]
        for wd in (0.05, 0.07, 0.10):
            wc, wp = near(spot * (1 + wd), "CE"), near(spot * (1 - wd), "PE")
            if wc and wp and wc > atm > wp:
                baskets["WS%.0f" % (wd * 100)] = [
                    leg(legs[(wc, "CE")]["tradingsymbol"], "BUY", 1),
                    leg(legs[(wp, "PE")]["tradingsymbol"], "BUY", 1),
                    leg(legs[(atm, "CE")]["tradingsymbol"], "SELL", 1),
                    leg(legs[(atm, "PE")]["tradingsymbol"], "SELL", 1)]

        for name, b in baskets.items():
            try:
                m = k.basket_order_margins(b, consider_positions=False, mode="compact")
                row[name] = round(m["initial"]["total"], 0)
            except Exception as exc:                            # noqa: BLE001
                row[name] = None
                row[name + "_err"] = str(exc)[:70]
            time.sleep(0.35)
        rows.append(row)
        print("  %-12s dte=%5d atm=%8.0f STR=%-11s STG2.5=%-11s STG5.0=%-11s "
              "IC2.5w7=%-11s WS7=%s"
              % (exp, dte, atm, row.get("STR"), row.get("STG2.5"), row.get("STG5.0"),
                 row.get("IC2.5w7"), row.get("WS7")), flush=True)

    OUT.parent.mkdir(exist_ok=True)
    json.dump(dict(spot=spot, asof=str(today), lot=LOT, rows=rows), open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
