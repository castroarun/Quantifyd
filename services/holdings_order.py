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
TICK = 0.05               # default NSE price tick (per-instrument tick resolved at order time)

_execs = {}               # exec_id -> state dict (single gunicorn worker → shared)
_lock = threading.Lock()

# per-instrument tick size, refreshed once a day from kite.instruments('NSE').
# Many scripts tick at 0.05, but some (e.g. MANORAMA) tick at 0.10 and reject a
# limit price that isn't a multiple — so round every limit to the real tick.
_tick_cache = {}
_tick_day = None
_tick_lock = threading.Lock()


def _tick_for(kite, symbol):
    """Real NSE tick size for a symbol (default 0.05). Cached per day."""
    global _tick_cache, _tick_day
    from datetime import date
    today = date.today()
    with _tick_lock:
        if _tick_day != today or not _tick_cache:
            try:
                rows = kite.instruments('NSE') or []
                _tick_cache = {r['tradingsymbol']: float(r.get('tick_size') or 0.05)
                               for r in rows if r.get('tick_size')}
                _tick_day = today
                logger.info(f"[topup] tick cache built: {len(_tick_cache)} NSE instruments")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[topup] tick cache build failed ({e}); using {TICK}")
        return _tick_cache.get(symbol, TICK) or TICK


def _parse_tick(reason):
    """Pull a tick size out of a broker rejection like 'Tick size for this script is 0.10'."""
    import re
    m = re.search(r'tick size[^0-9]*([0-9]*\.?[0-9]+)', reason or '', re.I)
    try:
        return float(m.group(1)) if m else None
    except (TypeError, ValueError):
        return None


def _kite_for(account):
    """Order-capable Kite client for the given account ('me' | 'dad')."""
    if account == 'dad':
        from services.dad_kite import get_dad_kite_trading
        return get_dad_kite_trading()
    from services.kite_service import get_kite
    return get_kite()


def _now():
    return datetime.now().strftime('%H:%M:%S')


def _round_tick(p, tick=TICK):
    tick = tick or TICK
    return round(round(p / tick) * tick, 2)


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


def start_topup(symbol, amount, holdings_syms, paper=False, account='me', qty=None):
    """Validate + kick off a smart-limit top-up. Returns dict (error or exec info).

    Sizing: pass `qty` for an exact share count (the user scaled it by hand), else
    the qty is derived from `amount` (floor(amount/ltp)). Either way the rupee value
    is capped at MAX_AMOUNT."""
    from services.market_data_refresh import _within_market_hours

    account = 'dad' if account == 'dad' else 'me'
    symbol = (symbol or '').strip().upper()
    if not symbol or symbol not in holdings_syms:
        return {'error': 'symbol is not one of your holdings'}

    explicit_qty = None
    if qty is not None:
        try:
            explicit_qty = int(qty)
        except (TypeError, ValueError):
            return {'error': 'invalid quantity'}
        if explicit_qty < 1:
            return {'error': 'quantity must be at least 1'}
    else:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {'error': 'invalid amount'}
        if amount <= 0:
            return {'error': 'invalid amount'}

    # one in-flight order per symbol per account
    with _lock:
        for s in _execs.values():
            if s['symbol'] == symbol and s.get('account') == account and s['status'] == 'working':
                return {'error': f'a {symbol} order is already in progress'}

    if not paper and not _within_market_hours():
        return {'error': 'market is closed — orders run 09:15–15:30 IST'}

    kite = _kite_for(account)
    ltp = _get_ltp(kite, symbol)
    if ltp <= 0:
        return {'error': 'could not fetch live price'}
    qty_final = explicit_qty if explicit_qty is not None else int(amount // ltp)
    if qty_final < 1:
        return {'error': f'₹{amount:,.0f} is below one share (LTP ₹{ltp:,.2f})'}
    used = round(qty_final * ltp, 2)
    if used > MAX_AMOUNT:
        return {'error': f'₹{used:,.0f} exceeds the ₹{MAX_AMOUNT:,} per-order cap'}

    tick = _tick_for(kite, symbol)
    exec_id = uuid.uuid4().hex[:12]
    cap = _round_tick(ltp * (1 + PRICE_CAP), tick)
    state = {
        'exec_id': exec_id, 'symbol': symbol, 'qty': qty_final, 'ref_ltp': ltp, 'side': 'BUY',
        'account': account, 'amount': used, 'used': used, 'cap': cap, 'tick': tick, 'paper': paper,
        'status': 'working', 'order_id': None, 'filled_qty': 0, 'avg_price': None,
        'message': 'placing order…', 'steps': [], 'started': datetime.now().isoformat(),
    }
    with _lock:
        _execs[exec_id] = state
    threading.Thread(target=_run, args=(exec_id,), daemon=True).start()
    logger.info(f"[topup] start {account} {symbol} qty={qty_final} ltp={ltp} tick={tick} cap={cap} paper={paper} exec={exec_id}")
    return {'exec_id': exec_id, 'symbol': symbol, 'qty': qty_final, 'ref_ltp': ltp,
            'cap': cap, 'used': used, 'paper': paper, 'account': account}


def start_exit(symbol, qty, holdings_qty, account='me', paper=False):
    """Market-SELL an existing holding (full or partial exit). Returns dict (error or exec info).

    Exits use MARKET orders for certainty of fill (the user's stated choice); a smart-limit
    is fine for adding on your own schedule, but an exit you want done should just clear."""
    from services.market_data_refresh import _within_market_hours

    account = 'dad' if account == 'dad' else 'me'
    symbol = (symbol or '').strip().upper()
    try:
        qty = int(qty)
        holdings_qty = int(holdings_qty)
    except (TypeError, ValueError):
        return {'error': 'invalid quantity'}
    if qty < 1:
        return {'error': 'quantity must be at least 1'}
    if holdings_qty < 1:
        return {'error': f'{symbol} is not in this account'}
    if qty > holdings_qty:
        return {'error': f'you hold {holdings_qty} {symbol} — cannot sell {qty}'}

    with _lock:
        for s in _execs.values():
            if s['symbol'] == symbol and s.get('account') == account and s['status'] == 'working':
                return {'error': f'a {symbol} order is already in progress'}

    if not paper and not _within_market_hours():
        return {'error': 'market is closed — orders run 09:15–15:30 IST'}

    kite = _kite_for(account)
    ltp = _get_ltp(kite, symbol)  # for the value estimate only; the order is MARKET

    exec_id = uuid.uuid4().hex[:12]
    state = {
        'exec_id': exec_id, 'symbol': symbol, 'qty': qty, 'ref_ltp': ltp, 'side': 'SELL',
        'account': account, 'held': holdings_qty, 'est_value': round(qty * ltp, 2), 'paper': paper,
        'status': 'working', 'order_id': None, 'filled_qty': 0, 'avg_price': None,
        'message': 'placing sell…', 'steps': [], 'started': datetime.now().isoformat(),
    }
    with _lock:
        _execs[exec_id] = state
    threading.Thread(target=_run_exit, args=(exec_id,), daemon=True).start()
    logger.info(f"[exit] start {account} SELL {symbol} qty={qty}/{holdings_qty} ltp={ltp} paper={paper} exec={exec_id}")
    return {'exec_id': exec_id, 'symbol': symbol, 'qty': qty, 'held': holdings_qty,
            'ref_ltp': ltp, 'est_value': state['est_value'], 'paper': paper, 'account': account}


def get_status(exec_id):
    with _lock:
        s = _execs.get(exec_id)
        return dict(s) if s else None


def _ladder(ltp, cap, tick=TICK):
    raw = [ltp * (1 - SHAVE_START), ltp * (1 - SHAVE_START / 2), ltp,
           ltp * (1 + PRICE_CAP / 2), cap]
    return [min(_round_tick(p, tick), cap) for p in raw]


def _run(exec_id):
    with _lock:
        s = dict(_execs[exec_id])
    sym, qty, ltp, cap, paper = s['symbol'], s['qty'], s['ref_ltp'], s['cap'], s['paper']
    account = s.get('account', 'me')
    tick = s.get('tick', TICK)

    if paper:
        return _run_paper(exec_id, _ladder(ltp, cap, tick), qty)

    kite = _kite_for(account)
    # up to two attempts: attempt 2 fires only if the broker rejected the price for a
    # tick-size mismatch (e.g. a 0.10-tick script) — we re-round to the real tick and retry.
    for attempt in (1, 2):
        ladder = _ladder(ltp, cap, tick)
        try:
            oid = kite.place_order(
                variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE,
                tradingsymbol=sym, transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty, product=kite.PRODUCT_CNC,
                order_type=kite.ORDER_TYPE_LIMIT, price=ladder[0],
            )
        except Exception as pe:  # noqa: BLE001  (tick rejection can arrive synchronously)
            nt = _parse_tick(str(pe))
            if attempt == 1 and nt and nt != tick:
                tick = nt; cap = _round_tick(ltp * (1 + PRICE_CAP), tick)
                _update(exec_id, tick=tick, message=f'adjusting to tick ₹{tick}')
                logger.info(f"[topup] {sym} tick fix {nt} (place); retrying")
                continue
            logger.exception(f"[topup] {sym} place failed")
            _update(exec_id, status='error', message=str(pe))
            return
        _update(exec_id, order_id=str(oid), message=f'limit @ ₹{ladder[0]}',
                steps_append={'t': _now(), 'price': ladder[0], 'action': 'place'})
        logger.info(f"[topup] {sym} placed {oid} @ {ladder[0]} tick={tick}")

        deadline = time.time() + BUDGET_S
        step_i = 0
        retry_tick = None
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
                nt = _parse_tick(st['reason'])
                if attempt == 1 and st['status'] == 'REJECTED' and nt and nt != tick:
                    retry_tick = nt  # re-round + re-place on the next attempt
                    logger.info(f"[topup] {sym} tick fix {nt} (rejected); retrying")
                    break
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

        if retry_tick:
            tick = retry_tick; cap = _round_tick(ltp * (1 + PRICE_CAP), tick)
            _update(exec_id, tick=tick, message=f'adjusting to tick ₹{tick}')
            continue  # attempt 2

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
        return


def _run_exit(exec_id):
    """Place a MARKET SELL and poll to a terminal state. Exits favour certainty over price."""
    with _lock:
        s = dict(_execs[exec_id])
    sym, qty, ltp, paper = s['symbol'], s['qty'], s['ref_ltp'], s['paper']
    account = s.get('account', 'me')

    if paper:
        time.sleep(2)
        _update(exec_id, status='filled', filled_qty=qty, avg_price=ltp,
                message=f'[paper] sold {qty} @ ~₹{ltp}',
                steps_append={'t': _now(), 'price': ltp, 'action': 'filled'})
        return

    kite = _kite_for(account)
    try:
        oid = kite.place_order(
            variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE,
            tradingsymbol=sym, transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=qty, product=kite.PRODUCT_CNC,
            order_type=kite.ORDER_TYPE_MARKET,
        )
        _update(exec_id, order_id=str(oid), message='market sell placed…',
                steps_append={'t': _now(), 'action': 'place'})
        logger.info(f"[exit] {account} {sym} SELL {qty} placed {oid}")

        deadline = time.time() + BUDGET_S
        while time.time() < deadline:
            time.sleep(1.5)
            st = _order_state(kite, oid)
            if st['status'] == 'COMPLETE':
                _update(exec_id, status='filled', filled_qty=st['filled'], avg_price=st['avg'],
                        message=f"sold {st['filled']} @ ₹{st['avg']}",
                        steps_append={'t': _now(), 'price': st['avg'], 'action': 'filled'})
                logger.info(f"[exit] {account} {sym} SOLD {st['filled']} @ {st['avg']}")
                return
            if st['status'] in ('REJECTED', 'CANCELLED'):
                _update(exec_id, status='error', message=f"{st['status']}: {st['reason']}")
                logger.warning(f"[exit] {account} {sym} {st['status']}: {st['reason']}")
                return
        # market order still not terminal after budget — report last known state
        st = _order_state(kite, oid)
        if st['filled'] > 0:
            _update(exec_id, status=('filled' if st['filled'] >= qty else 'partial'),
                    filled_qty=st['filled'], avg_price=st['avg'],
                    message=f"sold {st['filled']}/{qty} @ ₹{st['avg']}")
        else:
            _update(exec_id, status='working',
                    message='sell placed — still pending at exchange, check orderbook')
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[exit] {account} {sym} SELL failed")
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
