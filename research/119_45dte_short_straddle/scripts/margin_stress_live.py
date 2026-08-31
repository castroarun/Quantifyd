#!/usr/bin/env python3
"""Phase F(b) — the stress-margin test, MEASURED instead of reconstructed.

The first attempt (margin_reconstruct.py) tried to rebuild historical SPAN across
2019-26 and FAILED its calibration gate at 12.0% RMS against a 10% limit. It was
abandoned rather than tuned, because tuning until the gate passes is fitting to the
gate. The fallback was margin_recorder.py — record the real thing daily and revisit
once there is history. Five days exist, with India VIX pinned in a 10.48–11.13 band,
so there is still nothing to regress. That road is still closed.

This takes a different road that does not need history at all.

  THE TRICK: a straddle struck at K, held while spot rises m%, has the same payoff
  shape — and therefore the same SPAN scenario losses — as a straddle struck m%
  BELOW today's spot. We cannot time-travel a contract, but we can ask the broker
  for the real margin on the equivalent strike RIGHT NOW.

So the adverse-move axis is bought from Kite rather than modelled. Crossed with the
listed expiries, that gives the surface the book actually needs:

      margin  =  f(how far the market has moved against you, how much time is left)

and, from it, the number that matters: how far NIFTY can run before the margin on
3 lots exceeds the Rs 11.96L this book has reserved.

  MARGIN ALONE IS THE WRONG NUMBER. By the time the market is 8% offside you have
  ALSO taken an MTM loss on the position, and that loss has already come out of the
  same reserve. What must be compared to capital is margin + accumulated MTM loss.
  Both are priced here from the same live chain: the stressed straddle's value minus
  what an ATM straddle sells for today, on the same expiry, is exactly the loss.

WHAT THIS CANNOT DO — stated plainly, because it is the difference between this and
a complete answer: in a real sell-off spot moves AND implied vol explodes, and SPAN's
vol-scan adds margin on top of the price-scan measured here. India VIX today is
~10.6, close to its historic floor, so every number below is a BEST CASE. The vol
component still needs the recorder. That is why the review stays dated.

READ ONLY. basket_order_margins + ltp. Places no orders.
"""
import json
import os
import sys
import time
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if not os.path.exists(os.path.join(ROOT, "backtest_data")):
    ROOT = "/home/arun/quantifyd"
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

LOT, LOTS = 65, 3
CAPITAL = 1_196_000.0          # the book's reserved capital, research/119
STEP = 50                      # NIFTY strike grid
# adverse move against a short straddle, in %; sign is the SPOT move
MOVES = [-18, -15, -12, -8, -5, -3, 0, 3, 5, 8, 12, 15, 18]
THROTTLE = 0.40


def kite():
    from kiteconnect import KiteConnect
    k = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    k.set_access_token(json.load(
        open(os.path.join(ROOT, "backtest_data", "access_token.json")))["access_token"])
    return k


def main():
    k = kite()
    spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    today = date.today()

    ins = [i for i in k.instruments("NFO")
           if i["name"] == "NIFTY" and i["instrument_type"] in ("CE", "PE")]
    by_exp = {}
    for i in ins:
        by_exp.setdefault(str(i["expiry"]), {})[(float(i["strike"]), i["instrument_type"])] = i

    # one expiry per DTE band we care about, nearest match
    want = [("DTE ~45 (entry)", 45), ("DTE ~30", 30), ("DTE ~21 (exit)", 21),
            ("DTE ~7", 7), ("DTE ~1", 1)]
    chosen = []
    for label, target in want:
        best, bd = None, 1e9
        for e in by_exp:
            d = (date(*map(int, e.split("-"))) - today).days
            if d < 0:
                continue
            if abs(d - target) < bd:
                best, bd = (e, d), abs(d - target)
        if best and best not in [c[1] for c in chosen]:
            chosen.append((label, best))

    print("=" * 96)
    print("STRESS-MARGIN SURFACE — real Kite basket margin, %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("NIFTY spot %.2f   |   %d lots x %d = qty %d   |   book capital Rs %s"
          % (spot, LOTS, LOT, LOT * LOTS, "{:,.0f}".format(CAPITAL)))
    print("=" * 96)
    print("\nA short straddle after an m%% spot move == a straddle struck m%% away today.")
    print("Columns are the ADVERSE SPOT MOVE the position has already suffered.\n")

    hdr = "%-16s" % "expiry (DTE)"
    for m in MOVES:
        hdr += "%9s" % ("%+d%%" % m if m else "flat")
    print(hdr)

    grid = {}
    for label, (exp, dte) in chosen:
        line = "%-16s" % ("%s (%d)" % (exp[5:], dte))
        row = {}
        for m in MOVES:
            # strike that reproduces "spot has moved m%" relative to entry
            K = round((spot / (1 + m / 100.0)) / STEP) * STEP
            ce = by_exp[exp].get((float(K), "CE"))
            pe = by_exp[exp].get((float(K), "PE"))
            if not ce or not pe:
                line += "%9s" % "-"
                continue
            legs = [dict(exchange="NFO", tradingsymbol=t["tradingsymbol"],
                         transaction_type="SELL", variety="regular", product="NRML",
                         order_type="MARKET", quantity=LOT * LOTS)
                    for t in (ce, pe)]
            val = None
            for attempt in range(3):
                try:
                    r = k.basket_order_margins(legs, consider_positions=False, mode="compact")
                    # `initial` = SPAN + exposure BEFORE the premium credit. That is the
                    # right figure here: the credit actually received was the ATM one at
                    # entry, and the MTM loss is accounted separately below. Taking
                    # `final` would credit today's inflated ITM premium twice over.
                    val = r["initial"]["total"]
                    break
                except Exception as e:
                    if "Too many" in str(e) and attempt < 2:
                        time.sleep(1.2)
                        continue
                    break
            time.sleep(THROTTLE)
            if val is None:
                line += "%9s" % "err"
                continue
            row[m] = val
            over = val > CAPITAL
            line += "%9s" % (("*%.2fL" % (val / 1e5)) if over else ("%.2fL" % (val / 1e5)))
        grid[(exp, dte)] = row
        print(line)

    print("\n  values are Rs lakh for the WHOLE %d-lot position;  * = exceeds the "
          "Rs %.2fL reserved" % (LOTS, CAPITAL / 1e5))

    # ---- the question the book actually needs answered ---------------------
    # ---- MTM loss: what the move already cost, from the same chain ----------
    keys, spec = {}, {}
    for label, (exp, dte) in chosen:
        atmK = round(spot / STEP) * STEP
        for m in MOVES + ["ATM"]:
            K = atmK if m == "ATM" else round((spot / (1 + m / 100.0)) / STEP) * STEP
            ce = by_exp[exp].get((float(K), "CE"))
            pe = by_exp[exp].get((float(K), "PE"))
            if ce and pe:
                spec[(exp, m)] = (ce["tradingsymbol"], pe["tradingsymbol"])
                keys["NFO:" + ce["tradingsymbol"]] = 1
                keys["NFO:" + pe["tradingsymbol"]] = 1
    px = {}
    kl = list(keys)
    for i in range(0, len(kl), 200):
        try:
            px.update(k.ltp(kl[i:i + 200]))
        except Exception as e:
            print("  ltp batch failed: %s" % str(e)[:50])

    def straddle_pts(exp, m):
        t = spec.get((exp, m))
        if not t:
            return None
        a = px.get("NFO:" + t[0], {}).get("last_price")
        b = px.get("NFO:" + t[1], {}).get("last_price")
        return (a + b) if a and b else None

    print("\n" + "=" * 96)
    print("THE MTM LOSS THAT COMES WITH THE MOVE  (priced off the same chain)")
    print("=" * 96)
    print("Sold an ATM straddle; the market moved; this is what buying it back now costs")
    print("you versus the credit taken. Rs, at %d lots.\n" % LOTS)
    hdr2 = "%-16s" % "expiry (DTE)"
    for m in MOVES:
        hdr2 += "%10s" % ("%+d%%" % m if m else "flat")
    print(hdr2)
    mtm = {}
    for label, (exp, dte) in chosen:
        entry = straddle_pts(exp, "ATM")
        line = "%-16s" % ("%s (%d)" % (exp[5:], dte))
        row = {}
        for m in MOVES:
            v = straddle_pts(exp, m)
            if v is None or entry is None:
                line += "%10s" % "-"
                continue
            loss = max(0.0, (v - entry)) * LOT * LOTS
            row[m] = loss
            line += "%10s" % ("%.2fL" % (loss / 1e5) if loss else "0")
        mtm[(exp, dte)] = row
        print(line)

    print("\n" + "=" * 96)
    print("TOTAL CALL ON CAPITAL = margin + MTM loss   (vs Rs %.2fL reserved)" % (CAPITAL / 1e5))
    print("=" * 96)
    hdr3 = "%-16s" % "expiry (DTE)"
    for m in MOVES:
        hdr3 += "%10s" % ("%+d%%" % m if m else "flat")
    print(hdr3)
    worst_use = 0.0
    for (exp, dte), row in grid.items():
        line = "%-16s" % ("%s (%d)" % (exp[5:], dte))
        for m in MOVES:
            g = row.get(m)
            l = mtm.get((exp, dte), {}).get(m)
            if g is None or l is None:
                line += "%10s" % "-"
                continue
            tot = g + l
            worst_use = max(worst_use, tot / CAPITAL)
            line += "%10s" % (("*%.2fL" % (tot / 1e5)) if tot > CAPITAL else ("%.2fL" % (tot / 1e5)))
        print(line)
    print("\n  * = exceeds the reserve.  Peak observed use of capital: %.0f%%" % (100 * worst_use))

    print("\n" + "=" * 96)
    print("MARGIN ALONE — how far before MARGIN by itself exceeds the reserve")
    print("=" * 96)
    print("%-22s %12s %12s %12s %10s" % (
        "expiry (DTE)", "flat margin", "worst @+8%", "worst @-8%", "headroom"))
    for (exp, dte), row in grid.items():
        if 0 not in row:
            continue
        flat = row[0]
        up = row.get(8)
        dn = row.get(-8)
        head = CAPITAL / flat if flat else 0
        print("%-22s %12s %12s %12s %9.2fx" % (
            "%s (%d)" % (exp[5:], dte),
            "{:,.0f}".format(flat),
            "{:,.0f}".format(up) if up else "-",
            "{:,.0f}".format(dn) if dn else "-",
            head))

    # ---- what the recorder can and cannot yet say --------------------------
    import sqlite3
    db = os.path.join(ROOT, "backtest_data", "straddle45_margin_log.db")
    if os.path.exists(db):
        c = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
        rows = list(c.execute("SELECT d, vix, ref_margin_per_lot FROM margin_log "
                              "WHERE ref_margin_per_lot IS NOT NULL ORDER BY d"))
        c.close()
        print("\n" + "=" * 96)
        print("THE VOL AXIS — what the recorder can say so far (it cannot say much)")
        print("=" * 96)
        if len(rows) >= 2:
            vs = [r[1] for r in rows if r[1]]
            ms = [r[2] for r in rows if r[2]]
            print("  %d days recorded, %s -> %s" % (len(rows), rows[0][0], rows[-1][0]))
            print("  India VIX range %.2f - %.2f  (span %.2f pts)"
                  % (min(vs), max(vs), max(vs) - min(vs)))
            print("  reference 1-lot margin range Rs %s - Rs %s"
                  % ("{:,.0f}".format(min(ms)), "{:,.0f}".format(max(ms))))
            print("\n  VERDICT ON THE VOL AXIS: NOT ESTIMABLE. A %.2f-point VIX span, all of it"
                  % (max(vs) - min(vs)))
            print("  inside the calmest decile of the last decade, cannot be extrapolated to a")
            print("  VIX 30-80 event. Fitting a slope to it would be the same mistake as tuning")
            print("  the reconstruction until it passed its gate. The review stays dated.")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "margin_stress_live.json"), "w") as f:
        json.dump(dict(asof=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), spot=spot,
                       lots=LOTS, qty=LOT * LOTS, capital=CAPITAL,
                       grid={"%s|%d" % (e, d): r for (e, d), r in grid.items()}), f, indent=1)
    print("\nwrote %s" % os.path.join(RES, "margin_stress_live.json"))


if __name__ == "__main__":
    sys.exit(main())
