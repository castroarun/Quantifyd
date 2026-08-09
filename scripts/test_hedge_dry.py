"""DRY TEST of the hedge plumbing — picks the contract, prices it and sizes the position.
Places NO orders and writes nothing. Verifies the mechanics before the gate ever flips."""
import sys
sys.path.insert(0, "/home/arun/quantifyd")
from services import momentum_paper as mp

mp.init_db()
print("=== config ===")
for k in ("hedge_enabled", "hedge_ratio", "hedge_dte_target", "hedge_dte_min", "hedge_dte_max",
          "hedge_resize_drift", "hedge_max_premium_pct"):
    print(f"  {k:22} {mp.CFG[k]}")
print("\n=== contract selection (bi-weekly ATM NIFTY PE) ===")
c = mp._hedge_pick()
if not c:
    print("  NO CONTRACT FOUND"); sys.exit(1)
for k, v in c.items():
    print(f"  {k:12} {v}")
ltp = mp._opt_ltp(c["tsym"])
print(f"  {'ltp':12} {ltp}")
eq = mp._equity_value()
cash = mp._cash()
nav = eq + cash
qty = mp._hedge_target_qty(eq, c["spot"], c["lot"])
cost = ltp * qty
print(f"\n=== sizing against the CURRENT book ===")
print(f"  equity           Rs {eq:,.0f}")
print(f"  idle cash        Rs {cash:,.0f}")
print(f"  NAV              Rs {nav:,.0f}")
print(f"  target notional  Rs {mp.CFG['hedge_ratio']*eq:,.0f}  ({mp.CFG['hedge_ratio']}x equity)")
print(f"  qty              {qty}  ({qty/c['lot']:.0f} lots of {c['lot']})")
print(f"  actual notional  Rs {qty*c['spot']:,.0f}  -> effective ratio {qty*c['spot']/eq:.2f}x")
print(f"  premium cost     Rs {cost:,.0f}  ({100*cost/nav:.2f}% of NAV)")
print(f"  premium cap      {100*mp.CFG['hedge_max_premium_pct']:.0f}% of NAV -> "
      f"{'WITHIN CAP' if cost <= nav*mp.CFG['hedge_max_premium_pct'] else 'EXCEEDS CAP (would skip)'}")
print(f"  cash sufficient  {'YES' if cash >= cost else 'NO — hedge would be skipped'}")
print(f"\n=== current state ===")
st = mp.get_state()
print(f"  mode={st['mode']}  gate={st['gate']}  holdings={st['n_holdings']}  hedge={st['hedge']}")
print("\n(no orders placed, nothing written — mechanics verified only)")
