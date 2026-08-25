#!/usr/bin/env python3
"""
Phase F — stress-margin test. Reconstruct per-lot margin across 2019-26 and re-run
the book with a margin-call rule.

The blocking question: at 3 lots on Rs 11.96L, would the book have survived March 2020,
when India VIX went 12 -> 83.6 and SPAN inflates in the same event that drives the
drawdown? The fixed-capital CAGR assumes you can always hold. This tests that.

WHY A RECONSTRUCTION. Kite reports only today's margin; NSE's .spn parameter files are
not reachable from any public endpoint (every nsccl.DDMMYYYY.s.zip 404s). So margin is
rebuilt from SPAN's published STRUCTURE — a 16-scenario portfolio scan over a price-scan
range and a volatility-scan range, plus a short-option minimum and exposure margin —
with the parameters CALIBRATED to real measured margins.

THE GATE. I measured 18 exact margin points from Kite on 2026-08-24 (7 tenors at ATM,
11 moneyness points at two tenors). The model must reproduce those before any historical
number is believed. Fit is reported; if it is poor, the run is discarded, not massaged.

Everything historical is then driven by REAL inputs: NIFTY spot, the actual traded
straddle, and IV backed out of real bhavcopy option prices.
"""
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      chain_for_expiry, pick_atm, prev_session, implied_forward,
                      implied_vol_straddle, straddle_b76, dparse, dstr)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOT = 65

# ---- the gate: real margins measured from Kite basket_order_margins, 2026-08-24 -----
#      spot 24167.40, ATM 24150. (dte, strike, margin_per_lot)
MEASURED = [
    (1, 24150, 215348), (8, 24150, 212973), (15, 24150, 212583), (22, 24150, 213432),
    (29, 24150, 228843), (36, 24150, 216043), (64, 24150, 241789),
    (22, 24400, 202169), (22, 23950, 236363), (22, 24650, 230575), (22, 23700, 252228),
    (22, 24900, 247628), (22, 23450, 268974), (22, 25400, 279227), (22, 22950, 300135),
    (22, 25850, 307775), (22, 22500, 329045),
    (36, 24400, 199891), (36, 23950, 229729), (36, 24650, 227464), (36, 23700, 247212),
    (36, 24900, 229931), (36, 23450, 272894), (36, 25400, 265385), (36, 22950, 299125),
    (36, 25850, 303766), (36, 22500, 330418),
]
SPOT_TODAY = 24167.40
VIX_TODAY = 11.49          # India VIX close 2026-08-24

# SPAN's 16-scenario array: (price move as a fraction of PSR, vol move as a fraction of
# VSR, weight applied to the loss). The last pair are the "extreme" scenarios at 35%.
SCENARIOS = [
    (0, +1, 1.0), (0, -1, 1.0),
    (+1 / 3., +1, 1.0), (+1 / 3., -1, 1.0), (-1 / 3., +1, 1.0), (-1 / 3., -1, 1.0),
    (+2 / 3., +1, 1.0), (+2 / 3., -1, 1.0), (-2 / 3., +1, 1.0), (-2 / 3., -1, 1.0),
    (+1.0, +1, 1.0), (+1.0, -1, 1.0), (-1.0, +1, 1.0), (-1.0, -1, 1.0),
    (+2.0, 0, 0.35), (-2.0, 0, 0.35),
]


def straddle_val(F, K, T, iv):
    return straddle_b76(F, K, T, iv)


def span_margin(spot, K, T, iv, psr, vsr, som, expo):
    """Per-LOT margin for one short ATM-ish straddle, SPAN structure.

    psr : price scan range as a fraction of spot
    vsr : volatility scan range as a fraction of iv
    som : short-option minimum, as a fraction of notional
    expo: exposure margin, as a fraction of notional
    """
    base = straddle_val(spot, K, T, iv)
    worst = 0.0
    for dp, dv, w in SCENARIOS:
        s2 = spot * (1.0 + dp * psr)
        v2 = max(0.01, iv * (1.0 + dv * vsr))
        loss = (straddle_val(s2, K, T, v2) - base) * w      # short: loss when value rises
        worst = max(worst, loss)
    notional = spot * LOT
    scan = worst * LOT
    return max(scan, som * notional) + expo * notional


def calibrate(iv_by_dte):
    """Grid-search the four parameters against the measured points."""
    best = None
    for psr in [x / 1000. for x in range(40, 121, 5)]:
        for vsr in [x / 100. for x in range(10, 71, 10)]:
            for som in [x / 1000. for x in range(0, 41, 5)]:
                for expo in [x / 1000. for x in range(0, 41, 5)]:
                    err = 0.0
                    for dte, K, m in MEASURED:
                        T = dte / 365.0
                        iv = iv_by_dte.get(dte, VIX_TODAY / 100.0)
                        pred = span_margin(SPOT_TODAY, K, T, iv, psr, vsr, som, expo)
                        err += ((pred - m) / m) ** 2
                    err = math.sqrt(err / len(MEASURED))
                    if best is None or err < best[0]:
                        best = (err, psr, vsr, som, expo)
    return best


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)

    iv_by_dte = {d: VIX_TODAY / 100.0 for d, _, _ in MEASURED}
    err, psr, vsr, som, expo = calibrate(iv_by_dte)
    print("=" * 78)
    print("CALIBRATION against %d real measured margin points (Kite, 2026-08-24)" % len(MEASURED))
    print("=" * 78)
    print("  price scan range     %.1f%% of spot" % (100 * psr))
    print("  vol scan range       %.0f%% of IV" % (100 * vsr))
    print("  short-option minimum %.1f%% of notional" % (100 * som))
    print("  exposure margin      %.1f%% of notional" % (100 * expo))
    print("  RMS error            %.2f%%" % (100 * err))
    print()
    print("  %-6s %-8s %12s %12s %8s" % ("DTE", "strike", "measured", "model", "err"))
    worst_pt = 0.0
    for dte, K, m in MEASURED:
        p = span_margin(SPOT_TODAY, K, dte / 365.0, VIX_TODAY / 100.0, psr, vsr, som, expo)
        e = (p - m) / m
        worst_pt = max(worst_pt, abs(e))
        print("  %-6d %-8d %12s %12s %7.1f%%" %
              (dte, K, "{:,.0f}".format(m), "{:,.0f}".format(p), 100 * e))
    print("\n  worst single-point error %.1f%%" % (100 * worst_pt))

    GATE = 0.10
    if err > GATE:
        print("\nGATE FAILED: RMS error %.1f%% exceeds %.0f%%. The reconstruction does not "
              "reproduce today's real margins, so the historical path is NOT credible. "
              "Stopping — no historical numbers reported." % (100 * err, 100 * GATE))
        return
    print("\nGATE PASSED (RMS %.1f%% <= %.0f%%). Applying the calibrated model historically.\n"
          % (100 * err, 100 * GATE))

    # ---------------- historical margin path, driven by real inputs ------------
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")
    out = []
    for ym, exp in exps.items():
        e = dparse(exp)
        ed = prev_session(days, dstr(e - timedelta(days=45)))
        xd = prev_session(days, dstr(e - timedelta(days=21)))
        if not ed or not xd or not ("2019-01-01" <= ed <= "2026-06-30"):
            continue
        ch = chain_for_expiry(con, exp, ed, xd)
        if ed not in ch:
            continue
        sp0 = spot.get(ed)
        K = pick_atm(ch[ed], sp0) if sp0 else None
        if K is None:
            continue
        peak, at_entry, peak_day = 0.0, None, None
        for d in sorted(ch):
            if d < ed or d > xd:
                continue
            legs = ch[d].get(K)
            sd = spot.get(d)
            if not legs or "CE" not in legs or "PE" not in legs or not sd:
                continue
            ce, pe = legs["CE"]["close"], legs["PE"]["close"]
            if ce <= 0 or pe <= 0:
                continue
            T = max((dparse(exp) - dparse(d)).days, 1) / 365.0
            F = implied_forward(ce, pe, K, T)
            iv = implied_vol_straddle(ce + pe, F, K, T) or (VIX_TODAY / 100.0)
            mg = span_margin(sd, K, T, iv, psr, vsr, som, expo)
            if at_entry is None:
                at_entry = mg
            if mg > peak:
                peak, peak_day = mg, d
        if at_entry:
            out.append(dict(expiry=exp, entry=ed, strike=K, entry_margin=at_entry,
                            peak_margin=peak, peak_day=peak_day,
                            inflation=peak / at_entry))
    with open(os.path.join(RES, "margin_path.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)

    print("historical margin per lot, %d campaigns" % len(out))
    print("  entry margin : min %s  median %s  max %s" % tuple(
        "{:,.0f}".format(x) for x in (
            min(r["entry_margin"] for r in out),
            sorted(r["entry_margin"] for r in out)[len(out) // 2],
            max(r["entry_margin"] for r in out))))
    print("  PEAK margin  : min %s  median %s  max %s" % tuple(
        "{:,.0f}".format(x) for x in (
            min(r["peak_margin"] for r in out),
            sorted(r["peak_margin"] for r in out)[len(out) // 2],
            max(r["peak_margin"] for r in out))))
    print()
    print("  worst 6 campaigns by PEAK margin per lot:")
    for r in sorted(out, key=lambda r: -r["peak_margin"])[:6]:
        print("    %s entry %s  entry Rs %9s -> peak Rs %9s on %s  (%.2fx)" % (
            r["expiry"][:7], r["entry"], "{:,.0f}".format(r["entry_margin"]),
            "{:,.0f}".format(r["peak_margin"]), r["peak_day"], r["inflation"]))
    json.dump(dict(psr=psr, vsr=vsr, som=som, expo=expo, rms=err),
              open(os.path.join(RES, "margin_model.json"), "w"), indent=1)
    print("\nwrote results/margin_path.csv and margin_model.json")


if __name__ == "__main__":
    main()
