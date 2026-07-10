"""Smart-limit top-up execution for the Holdings Chart Wall.

A manual, user-initiated CNC (delivery) BUY of an existing holding, placed as an
"intelligent limit": start below LTP to try to shave the price, then chase up in
bounded steps to a hard cap that guarantees a fill in normal conditions; cancel
if the price runs away past the cap. Runs in a background thread; the frontend
polls status by exec_id.

Guardrails: holdings-only, server-side qty recompute, per-order rupee cap,
market-hours gate, one in-flight order per symbol, bounded price ladder, every
step logged. Paper mode simulates the whole chase without touching the market.
"""
import logging
import threading
import time
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

# --- tunables (all constants; safe to edit) ---
MAX_AMOUNT = 50_000       # hard rupee cap per order
SHAVE_START = 0.0020      # first limit 0.20% BELOW ltp (try to buy cheaper)
PRICE_CAP = 0.0030        # never pay more than 0.30% ABOVE ltp
BUDGET_S = 20             # total chase budget
STEP_S = 4               # seconds between reprice steps
TICK = 0.05               # NSE price tick

_execs = {}               # exec_id -> state dict (single gunicorn worker → shared)
_lock = threading.Lock()


def _now():
    return datetime.now().strftime('%H:%M:%S')


def _round_tick(p):
    return round(round(p / TICK) * TICK, 2)


def _get_ltp(kite, symbol):
    q = kite.ltp([f"NSE:{symbol}"]) or {}
    row = q.get(f"NSE:{symbol}") or {}
    return float(row.get('last_price') or 0)


def _order_state(kite, oid):
    """Latest (status, filled_qty, avg_price, reason) for an order id."""
    try:
        hist = kite.order_history(oid) or []
        last = hist[-1] if hist else {}
        return {
            'status': last.get('status', 'UNKNOWN'),
            'filled': int(last.get('filled_quantity') or 0),
            'avg': float(last.get('average_price') or 0),
            'reason': last.get('status_message') or '',
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"[topup] order_history({oid}) failed: {e}")
        return {'status': 'UNKNOWN', 'filled': 0, 'avg': 0.0, 'reason': str(e)}


def _update(exec_id, **kw):
    with _lock:
        s = _execs.get(exec_id)
        if not s:
            return
        steps_append = kw.pop('steps_append', None)
        s.update(kw)
        if steps_append:
            s['steps'].append(steps_append)


def start_topup(symbol, amount, holdings_syms, paper=False):
    """Validate + kick off a smart-limit top-up. Returns dict (error or exec info)."""
    from services.kite_service import get_kite
    from services.market_data_refresh import _within_market_hours

    symbol = (symbol or '').strip().upper()
    if not symbol or symbol not in holdings_syms:
        return {'error': 'symbol is not one of your holdings'}
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return {'error': 'invalid amount'}
    if amount <= 0 or amount > MAX_AMOUNT:
        return {'error': f'amount must be between ₹1 and ₹{MAX_AMOUNT:,}'}

    # one in-flight order per symbol
    with _lock:
        for s in _execs.values():
            if s['symbol'] == symbol and s['status'] == 'working':
                return {'error': f'a {symbol} top-up is already in progress'}

    if not paper and not _within_market_hours():
        return {'error': 'market is closed — orders run 09:15–15:30 IST'}

    kite = get_kite()
    ltp = _get_ltp(kite, symbol)
    if ltp <= 0:
        return {'error': 'could not fetch live price'}
    qty = int(amount // ltp)
    if qty < 1:
        return {'error': f'₹{amount:,.0f} is below one share (LTP ₹{ltp:,.2f})'}

    exec_id = uuid.uuid4().hex[:12]
    cap = _round_tick(ltp * (1 + PRICE_CAP))
    state = {
        'exec_id': exec_id, 'symbol': symbol, 'qty': qty, 'ref_ltp': ltp,
        'amount': amount, 'used': round(qty * ltp, 2), 'cap': cap, 'paper': paper,
        'status': 'working', 'order_id': None, 'filled_qty': 0, 'avg_price': None,
        'message': 'placing order…', 'steps': [], 'started': datetime.now().isoformat(),
    }
    with _lock:
        _execs[exec_id] = state
    threading.Thread(target=_run, args=(exec_id,), daemon=True).start()
    logger.info(f"[topup] start {symbol} qty={qty} ltp={ltp} cap={cap} paper={paper} exec={exec_id}")
    return {'exec_id': exec_id, 'symbol': symbol, 'qty': qty, 'ref_ltp': ltp,
            'cap': cap, 'used': state['used'], 'paper': paper}


def get_status(exec_id):
    with _lock:
        s = _execs.get(exec_id)
        return dict(s) if s else None


def _ladder(ltp, cap):
    raw = [ltp * (1 - SHAVE_START), ltp * (1 - SHAVE_START / 2), ltp,
           ltp * (1 + PRICE_CAP / 2), cap]
    return [min(_round_tick(p), cap) for p in raw]


def _run(exec_id):
    with _lock:
        s = dict(_execs[exec_id])
    sym, qty, ltp, cap, paper = s['symbol'], s['qty'], s['ref_ltp'], s['cap'], s['paper']
    ladder = _ladder(ltp, cap)

    if paper:
        return _run_paper(exec_id, ladder, qty)

    from services.kite_service import get_kite
    kite = get_kite()
    try:
        oid = kite.place_order(
            variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE,
            tradingsymbol=sym, transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=qty, product=kite.PRODUCT_CNC,
            order_type=kite.ORDER_TYPE_LIMIT, price=ladder[0],
        )
        _update(exec_id, order_id=str(oid), message=f'limit @ ₹{ladder[0]}',
                steps_append={'t': _now(), 'price': ladder[0], 'action': 'place'})
        logger.info(f"[topup] {sym} placed {oid} @ {ladder[0]}")

        deadline = time.time() + BUDGET_S
        step_i = 0
        while time.time() < deadline:
            time.sleep(STEP_S)
            st = _order_state(kite, oid)
            if st['status'] == 'COMPLETE':
                _update(exec_id, status='filled', filled_qty=st['filled'], avg_price=st['avg'],
                        message=f"filled {st['filled']} @ ₹{st['avg']}",
                        steps_append={'t': _now(), 'price': st['avg'], 'action': 'filled'})
                logger.info(f"[topup] {sym} FILLED {st['filled']} @ {st['avg']}")
                return
            if st['status'] in ('REJECTED', 'CANCELLED'):
                _update(exec_id, status='error', message=f"{st['status']}: {st['reason']}")
                logger.warning(f"[topup] {sym} {st['status']}: {st['reason']}")
                return
            step_i += 1
            if step_i < len(ladder):
                newp = ladder[step_i]
                try:
                    kite.modify_order(variety=kite.VARIETY_REGULAR, order_id=oid,
                                      quantity=qty, price=newp, order_type=kite.ORDER_TYPE_LIMIT)
                    _update(exec_id, message=f'chasing @ ₹{newp}',
                            steps_append={'t': _now(), 'price': newp, 'action': 'chase'})
                    logger.info(f"[topup] {sym} chase → {newp}")
                except Exception as me:  # noqa: BLE001
                    logger.error(f"[topup] {sym} modify failed: {me}")

        # budget exhausted — final status, then cancel any remainder
        st = _order_state(kite, oid)
        if st['status'] == 'COMPLETE':
            _update(exec_id, status='filled', filled_qty=st['filled'], avg_price=st['avg'],
                    message=f"filled {st['filled']} @ ₹{st['avg']}")
            return
        try:
            kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=oid)
        except Exception as ce:  # noqa: BLE001
            logger.error(f"[topup] {sym} cancel failed: {ce}")
        if st['filled'] > 0:
            _update(exec_id, status='partial', filled_qty=st['filled'], avg_price=st['avg'],
                    message=f"partial {st['filled']}/{qty} @ ₹{st['avg']}, rest cancelled (price ran away)")
        else:
            _update(exec_id, status='cancelled',
                    message='unfilled — price ran past the cap. Retry?')
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[topup] {sym} exec failed")
        _update(exec_id, status='error', message=str(e))


def _run_paper(exec_id, ladder, qty):
    """Simulate the chase with no real order — for a Monday dry-run."""
    _update(exec_id, message=f'[paper] limit @ ₹{ladder[0]}',
            steps_append={'t': _now(), 'price': ladder[0], 'action': 'place'})
    # pretend it fills at the mid step after ~2 reprices
    for i in range(1, 3):
        time.sleep(STEP_S)
        _update(exec_id, message=f'[paper] chasing @ ₹{ladder[i]}',
                steps_append={'t': _now(), 'price': ladder[i], 'action': 'chase'})
    time.sleep(1)
    fill = ladder[2]
    _update(exec_id, status='filled', filled_qty=qty, avg_price=fill,
            message=f'[paper] filled {qty} @ ₹{fill}',
            steps_append={'t': _now(), 'price': fill, 'action': 'filled'})
