"""Post-market smoke test for ORB live order plumbing.

Exercises the exact code paths that run during market hours:
  - kite.margins() + kite.positions() (auth)
  - kite.order_margins() preflight (validates order params without placing)
  - engine.place_entry_order(...) (real place_order call — will be rejected
    by Kite since market is closed)
  - engine.place_sl_m_order(...) on a synthetic position dict
  - engine.cancel_sl_m_order(...) no-op when nothing is live

Uses IRCTC (cheapest stock in the ORB universe) with qty=1 so even if
something unexpected happens, exposure is trivial.
"""
import logging
from services.orb_live_engine import ORBLiveEngine
from services.kite_service import get_kite
from config import ORB_DEFAULTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

TEST_SYM = 'IRCTC'
TEST_QTY = 1

kite = get_kite()
engine = ORBLiveEngine(ORB_DEFAULTS)

print('=' * 60)
print(f'ORB LIVE SMOKE TEST — {TEST_SYM} qty={TEST_QTY}')
print('=' * 60)

# ---------- 1. Auth sanity ----------
print('\n[1] Auth + account probe')
try:
    margins = kite.margins().get('equity', {})
    print(f'  equity cash avail : {margins.get("available", {}).get("live_balance")}')
    print(f'  equity used       : {margins.get("utilised", {}).get("debits")}')
    pos = kite.positions().get('net', [])
    print(f'  open MIS positions: {sum(1 for p in pos if p["product"]=="MIS" and p["quantity"])}')
    print('  ✓ auth OK')
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    raise SystemExit(1)

# ---------- 2. Get LTP so we have a realistic price ----------
print(f'\n[2] Fetch LTP for {TEST_SYM}')
try:
    ltp = kite.ltp([f'NSE:{TEST_SYM}']).get(f'NSE:{TEST_SYM}', {}).get('last_price')
    print(f'  ltp = {ltp}')
    if not ltp:
        raise RuntimeError('no LTP returned')
except Exception as e:
    print(f'  ✗ FAILED: {e}')
    raise SystemExit(1)

# Use a price far below LTP so even if the order somehow fills it's a
# genuinely-below-market fill (unrealistic after hours but belt-and-braces).
test_entry_price = round(ltp * 0.85, 1)
test_sl_price = round(ltp * 0.80, 1)

# ---------- 3. order_margins preflight ----------
print(f'\n[3] order_margins() preflight (no order placed)')
try:
    margin_resp = kite.order_margins([{
        'exchange': 'NSE',
        'tradingsymbol': TEST_SYM,
        'transaction_type': 'BUY',
        'variety': 'regular',
        'product': 'MIS',
        'order_type': 'LIMIT',
        'quantity': TEST_QTY,
        'price': test_entry_price,
    }])
    print(f'  response: {margin_resp}')
    print('  ✓ Kite accepts the order params')
except Exception as e:
    print(f'  ✗ FAILED: {e}')

# ---------- 4. Actual entry order via engine (expected to be rejected) ----------
print(f'\n[4] engine.place_entry_order({TEST_SYM!r}, LONG, qty=1, price={test_entry_price})')
order_id = None
try:
    order_id = engine.place_entry_order(TEST_SYM, 'LONG', TEST_QTY, test_entry_price)
    print(f'  order_id returned: {order_id}')
    if order_id:
        print('  (order reached Kite — will be REJECTED since market closed, but the code path works)')
        # Show what Kite said about it
        try:
            hist = kite.order_history(order_id)
            last = hist[-1] if hist else {}
            print(f'  Kite status: {last.get("status")}  reason: {last.get("status_message")}')
        except Exception as e2:
            print(f'  (could not fetch order history: {e2})')
    else:
        print('  place_entry_order returned None — check logs above for the Kite rejection reason')
except Exception as e:
    print(f'  ✗ unexpected exception: {e}')

# ---------- 5. SL-order test on a synthetic position ----------
print(f'\n[5] engine.place_sl_m_order(synthetic position)')
synthetic_pos = {
    'id': -1,  # fake id — DB updates will fail silently, that is fine
    'instrument': TEST_SYM,
    'direction': 'LONG',
    'qty': TEST_QTY,
    'sl_price': test_sl_price,
    'entry_price': test_entry_price,
}
try:
    sl_id = engine.place_sl_m_order(synthetic_pos)
    print(f'  sl_order_id: {sl_id}')
    if sl_id:
        try:
            hist = kite.order_history(sl_id)
            last = hist[-1] if hist else {}
            print(f'  Kite status: {last.get("status")}  reason: {last.get("status_message")}')
        except Exception as e2:
            print(f'  (could not fetch: {e2})')
except Exception as e:
    print(f'  ✗ unexpected exception: {e}')

# ---------- 6. Cancel test ----------
print(f'\n[6] engine.cancel_sl_m_order(synthetic position) — graceful no-op expected')
try:
    ok = engine.cancel_sl_m_order({**synthetic_pos, 'kite_sl_order_id': None})
    print(f'  ok={ok} (no order id → True = skip)')
except Exception as e:
    print(f'  ✗ unexpected: {e}')

# ---------- 7. Summary ----------
print('\n' + '=' * 60)
print('SMOKE TEST DONE')
print('Expected pattern after market close:')
print('  - auth OK')
print('  - order_margins() returns successfully')
print('  - place_entry_order RETURNS None OR an order id in REJECTED state')
print('    with message like "Market not open" / "Order not allowed"')
print('  - Any rejections are a PASS (not a code bug)')
print('=' * 60)
