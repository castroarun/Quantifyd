#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idle-cash park: IPO Base and Open Alpha idle cash into a liquid ETF, automatically.

WHY. Both books hold uninvested money as plain cash earning nothing, while every backtest
credits 5.2% post-tax on idle cash. Arun asked for an arbitrage fund; Kite Connect cannot place
mutual-fund orders, so the automatable choice is a liquid ETF — CASHIETF, which True North
already sweeps into. Chosen by Arun 13-Sep-2026. Status doc:
docs/CASH_PARK_LIQUID_ETF_IPO_OA_DAILY_DEPLOY_STATUS.md

THE ONE RULE: never park money a buy could need before the next release. Parking changes a
book's yield, never its trades.
  ipo-base    buys are placed at 09:20 by services/equity_executor.py, which calls release()
              first, so the book keeps only a small buffer and parks the rest.
  open-alpha  buys go out at 18:50 as AMOs that execute at the open, before any sale could
              settle, so it keeps cash for every free slot plus one swap, with a cushion, and
              parks only what is left.

LEDGER. A book's `cash` keeps meaning ALL uninvested money, parked or not, at cost. The
`park` block records units, cost and last price.
  free cash = cash - parked cost          (anything that sends an order uses this)
  NAV       = positions + cash + gain     (gain = units x price - parked cost)
  sizing    = unchanged, because cash still includes parked money

Usage:
  cash_park.py status
  cash_park.py plan    --book ipo-base|open-alpha
  cash_park.py park    --book ipo-base|open-alpha [--arm]
  cash_park.py release --book ipo-base|open-alpha --amount N [--arm]
  cash_park.py reconcile
"""
import json
import math
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYMBOL = 'CASHIETF'
CFG_FILE = ROOT / 'backtest_data' / 'cash_park.json'
LEDGER = ROOT / 'backtest_data' / 'cash_park_orders.json'
KILL = ROOT / 'backtest_data' / 'executor_kill.flag'
DB = ROOT / 'backtest_data' / 'market_data.db'
MP_DB = ROOT / 'backtest_data' / 'momentum_paper.db'
FEED = ROOT / 'backtest_data' / 'book_alerts.jsonl'
OA_UI = ROOT / 'static' / 'app' / 'oa_real.json'

DEFAULTS = dict(
    enabled={'ipo-base': False, 'open-alpha': False},
    min_order=25_000,          # the same floor True North's sweep uses
    max_order=None,            # a rupee cap per order, for the live test
    ipo_buffer=10_000,         # IPO releases on demand at 09:20, so it only needs rounding room
    oa_slot_pct=0.0625,
    oa_slots=16,
    oa_cushion=1.10,           # slot size is estimated at 15:10; entries size at 18:50
    oa_swap_slot=True,         # OA-ROT-1 can buy one extra name in an evening
    fill_timeout=90,
)
BOOKS = {
    'ipo-base': dict(module='services.ipo_paper', tag='IPO-PARK'),
    'open-alpha': dict(module='services.oa_real', tag='OA-PARK'),
}


# ─────────────────────────────── config, alerts, ledger ───────────────────────────────
def cfg():
    c = dict(DEFAULTS)
    try:
        c.update(json.load(open(CFG_FILE)))
    except Exception:
        pass
    return c


def ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def alert(title, body, critical=True):
    try:
        with open(FEED, 'a') as f:
            f.write(json.dumps(dict(ts=str(datetime.now()), book='CASH-PARK',
                                    urgency='critical' if critical else 'low',
                                    title=title, body=body)) + '\n')
    except Exception:
        pass
    if critical:
        try:
            from services.dividend_notify import send_email, send_push
            send_email(title, '<pre>%s</pre>' % body)
            send_push(title, body)
        except Exception as e:
            print('alert delivery failed:', e)


def _ledger():
    try:
        return json.load(open(LEDGER))
    except Exception:
        return {}


def _ledger_add(key, payload):
    d = _ledger()
    d[key] = payload
    tmp = LEDGER.with_suffix('.json.tmp')
    json.dump(d, open(tmp, 'w'), indent=1, default=str)
    tmp.replace(LEDGER)


# ─────────────────────────────── the park block (pure) ───────────────────────────────
def block(st):
    """The book's park block, created empty if absent. Pure: no I/O."""
    p = st.get('park')
    if not isinstance(p, dict):
        p = dict(symbol=SYMBOL, units=0, cost=0.0, last_px=None, since=None, orders=[])
        st['park'] = p
    return p


def units(st):
    p = st.get('park') or {}
    return int(p.get('units') or 0)


def parked_cost(st):
    p = st.get('park') or {}
    return float(p.get('cost') or 0.0)


def free_cash(st):
    """Money the broker will actually let this book spend."""
    return float(st.get('cash', 0.0)) - parked_cost(st)


def gain(st, px=None):
    """Unrealised gain on parked units. Zero when nothing is parked or no price is known."""
    u = units(st)
    if u <= 0:
        return 0.0
    price = px if px else (st.get('park') or {}).get('last_px')
    if not price:
        return 0.0
    return u * float(price) - parked_cost(st)


def ui_block(st, px=None):
    """What a book page shows."""
    u = units(st)
    price = px if px else (st.get('park') or {}).get('last_px')
    return dict(symbol=SYMBOL, units=u, cost=round(parked_cost(st)),
                value=round(u * float(price)) if (u and price) else round(parked_cost(st)),
                gain=round(gain(st, px)), free_cash=round(free_cash(st)),
                last_px=price)


# ─────────────────────────────── prices ───────────────────────────────
def db_close(asof=None):
    """Latest CASHIETF daily close on or before `asof`, for the nightly valuation."""
    try:
        con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
        q = ("select close from market_data_unified where symbol=? and timeframe='day' "
             "and close > 0 %s order by date desc limit 1"
             % ("and substr(date,1,10) <= ?" if asof else ''))
        args = (SYMBOL, str(asof)[:10]) if asof else (SYMBOL,)
        r = con.execute(q, args).fetchone()
        con.close()
        return float(r[0]) if r else None
    except Exception:
        return None


def live_px(k):
    try:
        return float(k.ltp(['NSE:' + SYMBOL])['NSE:' + SYMBOL]['last_price'])
    except Exception:
        return None


# ─────────────────────────────── reserve and plan ───────────────────────────────
def reserve(book, st, c=None):
    """Cash that must stay unparked. See the module header for why the books differ."""
    c = c or cfg()
    if book == 'ipo-base':
        return float(c['ipo_buffer'])
    held = len(st.get('positions', []))
    free_slots = max(0, int(c['oa_slots']) - held)
    need_slots = free_slots + (1 if c['oa_swap_slot'] else 0)
    nav = None
    try:
        ui = json.load(open(OA_UI))
        nav = float(ui.get('nav') or 0) or None
    except Exception:
        pass
    if nav is None:
        nav = float(st.get('cash', 0.0)) + sum(p['qty'] * p['buy'] for p in st.get('positions', []))
    return need_slots * float(c['oa_slot_pct']) * nav * float(c['oa_cushion'])


def plan(book, st, c=None):
    """-> dict(action='buy'|'sell'|None, amount, free, reserve, reason). Pure given state."""
    c = c or cfg()
    fr = free_cash(st)
    res = reserve(book, st, c)
    out = dict(book=book, free=round(fr), reserve=round(res), parked_cost=round(parked_cost(st)),
               units=units(st), action=None, amount=0, reason='')
    excess = fr - res
    if excess >= float(c['min_order']):
        amt = excess if not c.get('max_order') else min(excess, float(c['max_order']))
        out.update(action='buy', amount=round(amt),
                   reason='free cash Rs %s is above the reserve Rs %s' % (format(round(fr), ','),
                                                                          format(round(res), ',')))
    elif excess < 0 and units(st) > 0 and book == 'open-alpha':
        amt = min(-excess, parked_cost(st))
        if c.get('max_order'):
            amt = min(amt, float(c['max_order']))
        out.update(action='sell', amount=round(amt),
                   reason='free cash Rs %s is below the reserve Rs %s' % (format(round(fr), ','),
                                                                          format(round(res), ',')))
    else:
        out['reason'] = ('nothing to do: free Rs %s, reserve Rs %s, min order Rs %s'
                         % (format(round(fr), ','), format(round(res), ','),
                            format(int(c['min_order']), ',')))
    return out


# ─────────────────────────────── resting buy orders ───────────────────────────────
OPEN_STATUSES = ('OPEN', 'TRIGGER PENDING', 'AMO REQ RECEIVED', 'OPEN PENDING',
                 'VALIDATION PENDING', 'PUT ORDER REQ RECEIVED', 'MODIFY PENDING')


def book_buy_tags(book):
    """Tags this book puts on orders that SPEND cash."""
    if book == 'ipo-base':
        return ('IPO-ENTRY',)
    try:
        from services import oa_real
        return tuple(t for t in oa_real._book_tags() if 'EXIT' not in t.upper())
    except Exception:
        return ('OA-ENTRY', 'OA-TOPUP')


def open_buy_commitment(orders, tags):
    """Rupees tied up in this book's BUY orders still resting at the broker. Pure.

    Why it exists: IPO's 09:20 buy-stops can still be resting at 15:10, and the broker holds
    cash against them. The ledger does not know that, so without this the park run would try
    to park money that is already committed, and the broker would reject the ETF order."""
    tot = 0.0
    for o in orders or []:
        if (o.get('transaction_type') != 'BUY' or (o.get('tag') or '') not in tags
                or (o.get('status') or '').upper() not in OPEN_STATUSES):
            continue
        left = o.get('pending_quantity')
        if left is None:
            left = int(o.get('quantity') or 0) - int(o.get('filled_quantity') or 0)
        px = float(o.get('price') or 0) or float(o.get('trigger_price') or 0)
        tot += int(left) * px
    return tot


# ─────────────────────────────── orders ───────────────────────────────
def _book(book):
    import importlib
    return importlib.import_module(BOOKS[book]['module'])


def _armable(book, c):
    if KILL.exists():
        return 'kill switch present (%s)' % KILL
    if not c['enabled'].get(book):
        return 'switched off in %s' % CFG_FILE.name
    now = ist()
    if now.weekday() >= 5:
        return 'weekend'
    try:
        from services.trading_calendar import get_default_calendar
        if not get_default_calendar().is_trading_day(now.date()):
            return 'NSE holiday'
    except Exception as e:
        print('trading calendar unreadable (%s), using the weekday rule' % e)
    if not ((9, 20) <= (now.hour, now.minute) <= (15, 20)):
        return 'outside 09:20-15:20 IST'
    return None


def _order(k, side, qty, px, tag, timeout):
    """Marketable LIMIT, paced by the executor's single order gate. Waits for COMPLETE.
    -> (filled_qty, avg_price, order_id, error)."""
    from services import equity_executor as ex
    ex.load_ticks(k)
    limit = (ex.tick_round(px * 1.003, up=True, symbol=SYMBOL) if side == 'BUY'
             else ex.tick_round(px * 0.997, up=False, symbol=SYMBOL))
    oid, err = ex.send_order(k, variety='regular', exchange='NSE', tradingsymbol=SYMBOL,
                             transaction_type=side, quantity=int(qty), product='CNC',
                             order_type='LIMIT', price=limit, validity='DAY', tag=tag)
    if not oid:
        return 0, None, None, err
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        time.sleep(1.5)
        try:
            hist = k.order_history(oid)
            last = hist[-1] if hist else {}
        except Exception:
            continue
        status = (last.get('status') or '').upper()
        if status == 'COMPLETE':
            return int(last.get('filled_quantity') or qty), float(last.get('average_price') or limit), oid, None
        if status in ('REJECTED', 'CANCELLED'):
            fq = int(last.get('filled_quantity') or 0)
            return fq, float(last.get('average_price') or 0) or None, oid, status + ': ' + str(last.get('status_message'))
    try:
        k.cancel_order(variety='regular', order_id=oid)
    except Exception as e:
        print('cancel failed:', e)
    fq = int(last.get('filled_quantity') or 0)
    return fq, float(last.get('average_price') or 0) or None, oid, 'not complete within %ss, cancelled' % timeout


def _apply(book, side, fq, avg, oid, reason):
    """Record a fill on the book's own state, under the book's own lock."""
    m = _book(book)
    if not m.acquire_lock():
        alert('Cash park: FILL NOT RECORDED - %s busy' % book,
              '%s %d %s @ %.2f (order %s) filled but the book lock was held. Record it by hand '
              'in the park block, or run reconcile.' % (side, fq, SYMBOL, avg, oid))
        return False
    try:
        st = m.load_state()
        p = block(st)
        if side == 'BUY':
            p['units'] = int(p['units']) + fq
            p['cost'] = round(float(p['cost']) + fq * avg, 2)
            p['since'] = p.get('since') or str(date.today())
        else:
            avg_cost = float(p['cost']) / int(p['units']) if int(p['units']) else avg
            realised = fq * (avg - avg_cost)
            p['units'] = max(0, int(p['units']) - fq)
            p['cost'] = round(max(0.0, float(p['cost']) - fq * avg_cost), 2) if p['units'] else 0.0
            st['cash'] = round(float(st['cash']) + realised, 2)
            if not p['units']:
                p['since'] = None
        p['last_px'] = avg
        p.setdefault('orders', []).append(dict(ts=str(datetime.now())[:19], side=side, qty=fq,
                                               avg=avg, order_id=oid, reason=reason))
        p['orders'] = p['orders'][-60:]
        m.save_state(st)
    finally:
        m.release_lock()
    return True


def execute(book, side, amount, arm, reason, k=None):
    """Buy or sell about `amount` rupees of CASHIETF for a book. -> rupees actually moved."""
    c = cfg()
    m = _book(book)
    st = m.load_state()
    if side == 'SELL':
        amount = min(amount, parked_cost(st) * 1.05)
    k = k or _kite()
    px = live_px(k) if k else None
    if not px:
        px = (st.get('park') or {}).get('last_px') or db_close()
    if not px:
        print('%s: no %s price' % (book, SYMBOL))
        return 0.0
    if side == 'BUY':
        qty = int(amount // px)
    else:
        qty = min(units(st), int(math.ceil(amount / px)))
    if qty <= 0:
        print('%s: %s qty 0 for Rs %s at %.2f' % (book, side, format(round(amount), ','), px))
        return 0.0
    print('%s: %s %d %s ~Rs %s at ~%.2f  (%s)' % (book, side, qty, SYMBOL,
                                                  format(round(qty * px), ','), px, reason))
    if not arm:
        print('  DRY - nothing sent')
        return qty * px
    why = _armable(book, c)
    if why:
        print('  not arming: %s' % why)
        return 0.0
    fq, avg, oid, err = _order(k, side, qty, px, BOOKS[book]['tag'], int(c['fill_timeout']))
    if fq > 0 and avg:
        _apply(book, side, fq, avg, oid, reason)
        print('  FILLED %d @ %.2f (order %s)' % (fq, avg, oid))
    if err:
        alert('Cash park %s %s: %s' % (book, side, 'partial' if fq else 'failed'),
              '%s %d %s: filled %d. %s' % (side, qty, SYMBOL, fq, err),
              critical=(side == 'SELL'))
        print('  ERROR: %s' % err)
    return fq * (avg or px)


def _kite():
    from services import equity_executor as ex
    return ex.kite()


# ─────────────────────────────── entry points ───────────────────────────────
def park(book, arm=False):
    """The 15:10 run: park the excess, or (Open Alpha) release to restore the reserve."""
    c = cfg()
    m = _book(book)
    st = m.load_state()
    if book == 'ipo-base' and st.get('mode') != 'live':
        print('ipo-base: on paper, nothing to park')
        return
    p = plan(book, st, c)
    print(json.dumps(p))
    if not p['action']:
        return
    key = '%s|%s|park' % (book, date.today())
    if arm and key in _ledger():
        print('already ran today (%s)' % key)
        return
    k = None
    if p['action'] == 'buy':
        try:
            k = _kite()
            held_back = open_buy_commitment(k.orders(), book_buy_tags(book))
        except Exception as e:
            print('open orders unreadable (%s) - not parking' % e)
            return
        if held_back > 0:
            excess_after = (p['free'] - p['reserve']) - held_back
            if excess_after < float(c['min_order']):
                print('Rs %s is held against resting buy orders; what is left (Rs %s) is below '
                      'the minimum order - nothing parked'
                      % (format(round(held_back), ','), format(round(excess_after), ',')))
                return
            p['amount'] = round(min(p['amount'], excess_after))
            p['reason'] += '; Rs %s held back for resting buy orders' % format(round(held_back), ',')
    moved = execute(book, p['action'].upper(), p['amount'], arm, p['reason'], k=k)
    if arm and moved:
        _ledger_add(key, dict(plan=p, moved=round(moved), ts=str(datetime.now())))


def release(book, amount, arm=False, k=None):
    """Free at least `amount` rupees of cash by selling parked units. Used by the executor
    before placing IPO buys. -> rupees released (0 if nothing parked or not armed)."""
    m = _book(book)
    st = m.load_state()
    if units(st) <= 0 or amount <= 0:
        return 0.0
    return execute(book, 'SELL', amount, arm, 'release for buys: Rs %s short' % format(round(amount), ','), k=k)


def status():
    for book in BOOKS:
        try:
            st = _book(book).load_state()
        except Exception as e:
            print(book, 'state unreadable:', e)
            continue
        print(book, json.dumps(dict(ui_block(st, db_close()), plan=plan(book, st))))
    print('config:', json.dumps(cfg()))


def reconcile():
    """The units the books claim must never exceed what the broker holds."""
    claims = {}
    for book in BOOKS:
        try:
            claims[book] = units(_book(book).load_state())
        except Exception:
            claims[book] = 0
    try:
        con = sqlite3.connect('file:%s?mode=ro' % MP_DB, uri=True)
        r = con.execute("select value from mp_state where key='sweep_units'").fetchone()
        claims['truenorth'] = int(float(r[0])) if r else 0
        con.close()
    except Exception as e:
        print('true north sweep units unreadable:', e)
        claims['truenorth'] = 0
    k = _kite()
    held = 0
    for h in k.holdings():
        if h.get('tradingsymbol') == SYMBOL:
            held += int(h.get('quantity') or 0) + int(h.get('t1_quantity') or 0)
    total = sum(claims.values())
    print('claims %s = %d units; broker holds %d' % (claims, total, held))
    if total > held:
        alert('Cash park: books claim more %s than the broker holds' % SYMBOL,
              'claims %s total %d, broker %d' % (claims, total, held))
        return False
    return True


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else 'status'
    book = a[a.index('--book') + 1] if '--book' in a else None
    arm = '--arm' in a
    if cmd == 'status':
        status()
    elif cmd == 'plan':
        st = _book(book).load_state()
        print(json.dumps(plan(book, st)))
    elif cmd == 'park':
        park(book, arm)
    elif cmd == 'release':
        amt = float(a[a.index('--amount') + 1])
        print('released Rs %s' % format(round(release(book, amt, arm)), ','))
    elif cmd == 'reconcile':
        sys.exit(0 if reconcile() else 1)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == '__main__':
    main()
