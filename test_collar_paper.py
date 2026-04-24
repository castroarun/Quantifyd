"""
Collar Paper-Trading Dry-Run Test
==================================

Instantiates CollarEngine in paper mode, runs one scan cycle against today's
local market_data.db, and prints entries / legs / estimated prices.

Usage:
    python test_collar_paper.py

No Kite auth is needed — everything runs from backtest_data/market_data.db.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
# Keep the scanner quiet-ish
logging.getLogger('services.kc6_scanner').setLevel(logging.WARNING)

from services.collar_engine import (
    CollarEngine, bs_price,
    infer_strike_interval, round_put_strike, round_call_strike,
    last_thursday_of_month, pick_expiry,
)
from services.collar_db import get_collar_db
from config import COLLAR_DEFAULTS


def smoke_strike_rules():
    """Sanity-check strike rounding + expiry logic."""
    print("\n=== Strike rounding smoke test ===")
    for spot in [245.50, 1234.5, 3678.0, 12499.0, 25000.0]:
        i = infer_strike_interval(spot)
        p = round_put_strike(spot, 5.0)
        c = round_call_strike(spot, 5.0)
        print(f"  spot={spot:>9}  interval={i:>4}  put={p:>8}  call={c:>8}")

    print("\n=== Expiry picker smoke test ===")
    from datetime import date
    for d in [date(2026, 4, 24), date(2026, 4, 28), date(2026, 4, 29), date(2026, 12, 26)]:
        exp = pick_expiry(d, min_days=7)
        print(f"  today={d}  expiry={exp}  (days={(exp-d).days})")

    print("\n=== BS price smoke test ===")
    # ITM, ATM, OTM samples for both put and call
    spot, r, iv = 1000.0, 0.065, 0.25
    for strike in [950, 1000, 1050]:
        for t in [30/365, 7/365]:
            c = bs_price(spot, strike, t, r, iv, is_call=True)
            p = bs_price(spot, strike, t, r, iv, is_call=False)
            print(f"  spot=1000 K={strike} dte={t*365:.0f}d  call={c:>6}  put={p:>6}")


def run_one_scan():
    print("\n=== CollarEngine dry-run scan ===")
    engine = CollarEngine(config=COLLAR_DEFAULTS)
    db = get_collar_db()

    before_open = db.get_open_count()
    before_closed = db.get_stats().get('total_trades', 0)
    print(f"  DB state BEFORE: open={before_open}, closed_total={before_closed}")

    result = engine.run_full_scan()
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        return result

    print(f"  Universe loaded:      {result.get('symbols_loaded')}")
    print(f"  Universe ATR ratio:   {result.get('universe_atr_ratio')}")
    print(f"  Crash filter active:  {result.get('crash_filter_active')}")
    print(f"  Open collars after:   {result.get('open_positions')}")
    entries = result.get('entries_taken', [])
    exits = result.get('exits_taken', [])
    print(f"  Entries taken:        {len(entries)}")
    print(f"  Exits taken:          {len(exits)}")

    if entries:
        print("\n  --- New Collars ---")
        for e in entries:
            print(
                f"    {e['symbol']}  spot={e['spot']:.2f}  qty={e['lot_size']}  "
                f"FUT@{e['future_price']}  "
                f"+PUT {e['put_strike']}@{e['put_price']}  "
                f"-CALL {e['call_strike']}@{e['call_price']}  "
                f"exp={e['expiry']}"
            )

    if exits:
        print("\n  --- Closed Collars ---")
        for x in exits:
            print(
                f"    {x['symbol']}  reason={x['exit_reason']}  "
                f"net_pnl={x['net_pnl']:+.0f}  pnl%={x.get('pnl_pct')}  "
                f"hold={x.get('hold_days')}d"
            )

    # Show MTM on any open positions
    opens = db.get_open_positions_with_legs()
    if opens:
        print("\n  --- Open Collars MTM ---")
        for pos in opens:
            try:
                mtm = engine.mark_to_market(pos)
            except Exception as ex:
                print(f"    {pos['symbol']}: MTM failed: {ex}")
                continue
            print(
                f"    {pos['symbol']}  spot={mtm['spot_now']}  "
                f"net_mtm={mtm['net_mtm']:+.0f}  "
                f"(fut={mtm['future_mtm']:+.0f} "
                f"put={mtm['put_mtm']:+.0f} "
                f"call={mtm['call_mtm']:+.0f})"
            )

    return result


def main():
    try:
        smoke_strike_rules()
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return 1

    try:
        run_one_scan()
    except Exception as e:
        import traceback
        print(f"Scan failed: {e}")
        traceback.print_exc()
        return 2

    print("\nOK - collar paper flow ran end-to-end without errors.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
