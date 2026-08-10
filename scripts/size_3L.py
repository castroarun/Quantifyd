"""What a Rs3L cash allocation actually supports — MTF reach, position granularity, and whether the
put hedge is even possible at this size."""
import sys
sys.path.insert(0, "/home/arun/quantifyd")
from services import momentum_paper as mp

CASH = 300_000
mp.init_db()
close, tv = mp._panel()
asof = close.index[-1]
etf = mp._rs_basket(close, tv, asof)
top8 = etf[:8]
live = mp._live_prices(top8)
print("=" * 74)
print(f"SIZING A Rs{CASH:,} ALLOCATION  (shared account, MTF)")
print("=" * 74)

print("\n1. MTF REACH  (Zerodha MTF margin varies by stock; typical bands)")
for m, lbl in ((0.25, "25% margin (Group-1 large caps)"), (0.35, "35%"), (0.50, "50% (many mid caps)")):
    print(f"   {lbl:34} -> up to Rs{CASH/m:,.0f} exposure ({1/m:.1f}x)")
print("   NOTE: our leverage study says 1.3-1.6x is the sweet spot and >1.6x degrades Calmar.")
for lev in (1.0, 1.3, 1.6):
    exp = CASH * lev
    borrowed = exp - CASH
    print(f"   at {lev}x -> Rs{exp:,.0f} exposure, Rs{borrowed:,.0f} borrowed, "
          f"interest @14.6% = Rs{borrowed*0.146:,.0f}/yr ({100*borrowed*0.146/exp:.1f}% of the book)")

print("\n2. POSITION GRANULARITY  (8 equal slots)")
for lev in (1.0, 1.6):
    exp = CASH * lev
    per = exp / 8
    print(f"\n   at {lev}x -> Rs{exp:,.0f} exposure, Rs{per:,.0f} per slot")
    bad = 0
    for s in top8:
        px = live.get(s)
        if not px:
            continue
        q = int(per // px)
        used = q * px
        err = 100 * (used - per) / per if per else 0
        flag = ""
        if q == 0:
            flag = "  <-- UNBUYABLE (1 share > slot)"; bad += 1
        elif abs(err) > 25:
            flag = f"  <-- {err:+.0f}% off target"; bad += 1
        print(f"     {s:<13} Rs{px:>9,.1f}  qty {q:>4}  = Rs{used:>9,.0f}{flag}")
    print(f"     -> {bad}/8 slots badly sized")

print("\n3. CAN THE PUT HEDGE WORK AT THIS SIZE?")
try:
    c = mp._hedge_pick()
    ltp = mp._opt_ltp(c["tsym"])
    lot_notional = c["spot"] * c["lot"]
    print(f"   NIFTY spot {c['spot']:,.0f}, lot {c['lot']} -> ONE lot = Rs{lot_notional:,.0f} notional")
    print(f"   ATM put premium ~Rs{ltp:.1f} -> one lot costs Rs{ltp*c['lot']:,.0f}")
    for lev in (1.0, 1.6):
        eq = CASH * lev
        want = mp.CFG["hedge_ratio"] * eq
        lots = want / lot_notional
        actual = lot_notional / eq
        prem = ltp * c["lot"]
        print(f"\n   at {lev}x (equity Rs{eq:,.0f}):")
        print(f"     target {mp.CFG['hedge_ratio']}x notional = Rs{want:,.0f} -> {lots:.2f} lots")
        print(f"     but the MINIMUM is 1 lot = Rs{lot_notional:,.0f} = {actual:.1f}x the equity")
        print(f"     premium Rs{prem:,.0f} = {100*prem/eq:.1f}% of the book "
              f"(cap is {mp.CFG['hedge_max_premium_pct']*100:.0f}%)")
        print(f"     VERDICT: {'workable' if 0.75 <= lots <= 3 and prem <= eq*mp.CFG['hedge_max_premium_pct'] else 'NOT VIABLE — cannot size a NIFTY lot to this book'}")
    need_eq = lot_notional / mp.CFG["hedge_ratio"]
    print(f"\n   -> the 2x put hedge needs roughly Rs{need_eq:,.0f} of EQUITY to size to one lot.")
except Exception as e:
    print(f"   (could not price the hedge: {e})")
