"""MOCK GO-LIVE REPORT — what a Rs20L deployment would actually look like tomorrow.

Uses the LIVE ranking and REAL current prices. Places no orders, writes nothing.
Shows the exact basket, whole-share quantities, order values, cash left over, the initial Donchian
stop for each name, the gate/hedge state, and what the first day's order list would be.
"""
import sys
from datetime import date
sys.path.insert(0, "/home/arun/quantifyd")
from services import momentum_paper as mp

CAPITAL = 2_000_000
mp.init_db()
close, tv = mp._panel()
asof = close.index[-1]
etf = mp._rs_basket(close, tv, asof)
top8 = etf[:mp.CFG["n_hold"]]
buf = etf[:mp.CFG["buffer"]]
live = mp._live_prices(top8)
risk_off = mp._gate_risk_off(close, asof)
b = close[mp.BENCH].dropna()
gate_last, gate_sma = float(b.iloc[-1]), float(b.tail(mp.CFG["gate_sma"]).mean())

print("=" * 78)
print(f"MOCK GO-LIVE REPORT — Rs{CAPITAL:,} · generated {date.today()} · data as of {asof.date()}")
print("=" * 78)
print(f"\nMACRO GATE : NIFTYBEES {gate_last:.2f} vs 100-SMA {gate_sma:.2f} "
      f"({(gate_last/gate_sma-1)*100:+.2f}%) -> {'RISK-OFF' if risk_off else 'RISK-ON'}")
if risk_off:
    print("  Gate is RISK-OFF: the book would NOT deploy into stocks; it would wait in cash")
    print("  and buy a bi-weekly 2x NIFTY put only once positions exist. Nothing to buy today.")
else:
    print("  Gate is RISK-ON -> deploy fully into the top-8, equal weight. No hedge (hedge only")
    print("  activates at a risk-off gate).")

RES_PCT = mp.CFG["cash_reserve_pct"]
reserve = CAPITAL * RES_PCT
per = (CAPITAL - reserve) / mp.CFG["n_hold"]
print(f"\nTARGET BASKET — top {mp.CFG['n_hold']} of {len(etf)} ranked, equal weight Rs{per:,.0f} each")
print(f"{'#':>2} {'STOCK':<14} {'TIER':<8} {'PRICE':>9} {'QTY':>7} {'VALUE':>12} {'WT':>6} {'DONCHIAN STOP':>14}")
rows, deployed = [], 0.0
for i, s in enumerate(top8, 1):
    px = live.get(s)
    if not px:
        cs = close[s].loc[:asof].dropna()
        px = float(cs.iloc[-1]) if len(cs) else None
    if not px:
        print(f"{i:>2} {s:<14} NO PRICE"); continue
    qty = int(per // px)
    val = qty * px
    stop = mp._donchian_low(close, s, asof)
    tier = mp._mcap_tier(s) or "-"
    deployed += val
    rows.append((s, px, qty, val, stop, tier))
    sd = f"{stop:,.1f}" if stop else "n/a"
    dist = f" ({(px/stop-1)*100:+.1f}%)" if stop else ""
    print(f"{i:>2} {s:<14} {tier:<8} {px:>9,.1f} {qty:>7} {val:>12,.0f} "
          f"{100*val/CAPITAL:>5.1f}% {sd:>10}{dist}")
cash = CAPITAL - deployed
cost = deployed * mp.CFG["cost_rt"] / 2
print(f"\n{'':>2} {'DEPLOYED':<14} {'':<8} {'':>9} {'':>7} {deployed:>12,.0f} {100*deployed/CAPITAL:>5.1f}%")
print(f"{'':>2} {'CASH LEFT':<14} {'':<8} {'':>9} {'':>7} {cash:>12,.0f} {100*cash/CAPITAL:>5.1f}%"
      f"   (Rs{reserve:,.0f} hedge reserve + odd-lot rounding)")
print(f"{'':>2} {'EST. ENTRY COST':<14} {'':<8} {'':>9} {'':>7} {cost:>12,.0f}"
      f"   ({mp.CFG['cost_rt']/2*100:.2f}% one-way: STT + charges)")

print(f"\nDAY-1 ORDER LIST ({len(rows)} orders, all NSE CNC MARKET, tag MOMENTUM)")
for s, px, qty, val, stop, tier in rows:
    print(f"  BUY  {qty:>6} x {s:<14} ~Rs{px:>9,.1f}   = Rs{val:>11,.0f}")
print(f"  (each order is checked against the Rs{mp.CFG['live_max_order_value']:,} per-order cap: "
      f"max here Rs{max(r[3] for r in rows):,.0f} -> "
      f"{'OK' if max(r[3] for r in rows) <= mp.CFG['live_max_order_value'] else 'BLOCKED'})")

print(f"\nONGOING RULES ONCE LIVE")
print(f"  Daily 15:15  — sell any name closing below its 15-day low (stop shown above), to cash")
print(f"  Weekly Fri   — if NIFTYBEES < 100-SMA: HOLD the stocks and BUY a bi-weekly ~2x NIFTY put")
print(f"  Monthly last — re-rank; sell names dropping out of the top {mp.CFG['buffer']}, buy new top-8")
print(f"                 winners are NOT trimmed (no needless STCG)")
print(f"  Daily 09:20  — reconcile book vs broker holdings, email on mismatch")
print(f"  Daily 15:35  — EOD email; month-end 15:40 — monthly report")

print(f"\nIF THE GATE FLIPS RISK-OFF (hedge illustration at today's prices)")
try:
    c = mp._hedge_pick()
    if c:
        ltp = mp._opt_ltp(c["tsym"])
        eq = deployed
        qty = mp._hedge_target_qty(eq, c["spot"], c["lot"])
        prem = ltp * qty
        print(f"  BUY {qty} x {c['tsym']} (~{qty//c['lot']} lots) @ ~{ltp:.1f} = Rs{prem:,.0f}")
        print(f"    spot {c['spot']:,.0f} · strike {c['strike']:,.0f} · expiry {c['expiry']} ({c['dte']}d)")
        print(f"    notional Rs{qty*c['spot']:,.0f} = {qty*c['spot']/eq:.2f}x equity · "
              f"premium {100*prem/CAPITAL:.2f}% of capital")
        print(f"    funded from the Rs{cash:,.0f} idle cash -> "
              f"{'SUFFICIENT' if cash >= prem else 'INSUFFICIENT - hedge would be SKIPPED + email alert'}")
except Exception as e:
    print(f"  (hedge illustration unavailable: {e})")

print(f"\nSAFETY BEFORE ANY REAL ORDER")
print(f"  live_mode  must be ON  (currently {'ON' if mp._is_live() else 'OFF'})")
print(f"  live_armed must be ON  (currently {'ON' if str(mp._get('live_armed','0')).lower() in ('1','true','on','yes') else 'OFF'})")
print(f"  -> both keys required; either one off blocks every order")
print("\nNO ORDERS PLACED. NOTHING WRITTEN. Simulation only.")
