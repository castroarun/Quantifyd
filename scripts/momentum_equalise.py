# -*- coding: utf-8 -*-
"""ONE-OFF: trim the oversized holdings back to target and fill the vacant slots.

Why this is needed once, and should not need to happen again:

Three deposits were spread across whatever was held at the time, so BHEL, LAURUSLABS and
NATIONALUM drifted ~45% above an equal-weight slot while two slots sat vacant. The deposit rule
that caused it is fixed (2026-08-31) — money now goes to the biggest shortfall first — but that
only prevents NEW drift. This corrects the distortion already in the book.

It is cheap precisely because the excess is returned deposits, not gains: measured 2026-08-31 the
whole trim realised ~Rs410 of tax because the three are barely in profit (NATIONALUM is at a loss).
The standing "never trim winners" rule exists to avoid paying 20% on real appreciation; there is
almost none here to pay on.

Self-funding: the trim frees ~Rs1.49L which, with idle cash, covers both vacant slots. No deposit.

SAFETY
  - dry run unless --confirm is passed
  - refuses outside market hours (sells and buys must actually fill)
  - refuses if the gate is risk-off (the book should be going to cash, not rebalancing into it)
  - sells only the book's own quantity, never Arun's personal shares in the same account
  - skips any candidate already below its Donchian stop (research/115)
"""
import sys
from datetime import date

sys.path.insert(0, "/home/arun/quantifyd")
from services import momentum_paper as mp   # noqa: E402

CONFIRM = "--confirm" in sys.argv


def main():
    if not mp._get("seeded"):
        print("book not seeded — abort"); return
    close, tv = mp._panel()
    asof = close.index[-1]
    if mp._gate_risk_off(close, asof):
        print("GATE IS RISK-OFF — the book should be moving to cash, not rebalancing. Abort.")
        return
    if CONFIRM and not mp._market_open_now():
        print("market is shut — orders would be refused. Re-run during 09:15-15:30. Abort.")
        return

    pos = mp._positions()
    n = mp.CFG["n_hold"]
    etf = mp._rs_basket(close, tv, asof) or []
    buf = set(etf[:mp.CFG["buffer"]])
    live = mp._live_prices(sorted(set(list(pos) + etf)))
    kept = [s for s in pos if s in buf]

    # vacancies filled from the ranked pool, skipping anything below its own stop
    adds = []
    for cand in etf:
        if len(kept) + len(adds) >= n:
            break
        if cand in kept or not live.get(cand):
            continue
        low = mp._donchian_low(close, cand, asof)
        if low is not None and live[cand] < low:
            print(f"  skip {cand}: {live[cand]:.1f} is below its 15-day low {low:.1f}")
            continue
        adds.append(cand)

    slots = (kept + adds)[:n]
    nav = mp._equity_value() + mp._cash() + mp._sweep_value()
    target = (nav * (1 - mp.CFG["cash_reserve_pct"])) / n
    print(f"\n  NAV Rs{nav:,.0f}  ->  equal-weight target Rs{target:,.0f} across {n} slots")
    print(f"  slots: {slots}\n")

    trims, buys = [], []
    for s in slots:
        px = live.get(s) or (pos[s]["entry_price"] if s in pos else 0)
        if not px:
            continue
        cur = pos[s]["qty"] * px if s in pos else 0.0
        gap = target - cur
        if gap < -px:                                   # oversized by at least one share
            q = int(min(abs(gap) / px, pos[s]["qty"]))
            if q > 0:
                trims.append((s, q, px, q * px))
        elif gap > px:
            buys.append((s, int(gap / px), px, int(gap / px) * px, s not in pos))

    freed = sum(t[3] for t in trims)
    need = sum(b[3] for b in buys)
    cash_now = mp._cash() + mp._sweep_value() - nav * mp.CFG["cash_reserve_pct"]

    print("  TRIM (sell part of the oversized holdings):")
    for s, q, px, v in trims:
        print(f"    SELL {q:>6} {s:<12} @ {px:>9.2f} = Rs{v:>10,.0f}")
    print(f"    frees Rs{freed:,.0f}\n")
    print("  FILL:")
    for s, q, px, v, isnew in buys:
        print(f"    BUY  {q:>6} {s:<12} @ {px:>9.2f} = Rs{v:>10,.0f}" + ("   <-- vacant slot" if isnew else ""))
    print(f"    needs Rs{need:,.0f}\n")
    print(f"  deployable cash now Rs{cash_now:,.0f} + freed Rs{freed:,.0f} = Rs{cash_now+freed:,.0f}"
          f"  vs Rs{need:,.0f} needed  ->  {'SUFFICIENT' if cash_now+freed >= need else 'SHORT by Rs%s' % f'{need-cash_now-freed:,.0f}'}")

    if not CONFIRM:
        print("\n  DRY RUN — nothing sent. Re-run with --confirm during market hours to execute.")
        return

    d = date.today().isoformat()
    if mp.CFG["live_cash_sweep"] and mp._sweep_units() > 0:
        print("\n  releasing parked ETF cash first...")
        mp.unsweep(max(0.0, need - mp._cash()) * 1.10 or None)
    for s, q, px, v in trims:
        print(f"  SELL {q} {s}")
        mp._sell(s, px, d, "EQUALISE_TRIM", qty=q)
    for s, q, px, v, isnew in buys:
        print(f"  BUY  {q} {s}")
        mp._buy(s, px, v, d, "EQUALISE_FILL")
    print("\n  done — re-check /app/momentum-paper")


if __name__ == "__main__":
    main()
