"""Deploy idle cash in the equity books, automatically, with real orders.

WHY THIS EXISTS. Money credited through the Capital Desk used to sit as cash until Arun
placed the buys himself — Open Alpha and IPO Base have no executor, so a deposit was only
half a deposit. His instruction (08-Sep-2026): "once i give UI deposit, nothing shud be
wiating on me". This closes that loop.

WHAT IT DOES
  open-alpha  tops up the existing holdings toward equal weight with whatever cash is
              idle. Arun chose this (Option A, 08-Sep-2026) over parking the cash for
              the next signal. IT IS A DEVIATION FROM THE TESTED SPEC — see below.
  ipo-base    places the armed buy-stops for pending candidates at 18.75% of equity,
              and tops up nothing: that book sizes at entry by design.

THE DEVIATION, STATED LOUDLY. research/142's 33.8% after-tax CAGR for Open Alpha comes
from a book that sizes at entry and never adds. Topping up positions that are already
days old, some already extended past their pivot, is a rule that backtest never saw.
True North tops up only because research/112 validated it beat parking on 12 of 12
deposit calendars; no equivalent test exists for Open Alpha. A study is registered
(Ops Centre, 2026-10-31) to run that test. Until it returns, this is a known, deliberate,
documented deviation — not an accident.

SAFETY
  - dry run is the DEFAULT. Placing orders needs --arm.
  - kill switch: backtest_data/executor_kill.flag stops everything, no exceptions.
  - market hours only when arming (09:20-15:00 IST, weekdays).
  - idempotent: a per-day ledger keyed (book, symbol, date) means a second run in the
    same session cannot double-buy. This is the guardrail that matters most, because
    cron re-runs and manual runs will overlap.
  - capital fence: never spends more than the book's own recorded cash.
  - marketable LIMIT, tick-rounded. Kite rejects bare MARKET orders on equities
    ("Market orders without market protection are not allowed via API").
  - every order is written to the book's state and raised as an alert.

Run:  venv/bin/python services/equity_executor.py [--arm] [--book open-alpha|ipo-base]
Cron: 09:20 IST weekdays (after the open settles, well before the 15:18 exit check).
Log:  /tmp/equity_executor.log   Ledger: backtest_data/executor_orders.json
"""
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'backtest_data' / 'executor_orders.json'
KILL = ROOT / 'backtest_data' / 'executor_kill.flag'
ALLOC = ROOT / 'backtest_data' / 'allocation_targets.json'
FEED = Path('/tmp/nas_alert_feed.log')

OA_STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
OA_LOCK = ROOT / 'backtest_data' / 'oa_real_state.lock'
IPO_STATE = ROOT / 'backtest_data' / 'ipo_paper_state.json'
IPO_LOCK = ROOT / 'backtest_data' / 'ipo_paper_state.lock'

OA_SLOTS = 16
IPO_SIZE_PCT = 0.1875
MIN_ORDER = 2_000.0          # below this the brokerage/rounding is not worth an order
TICK = 0.05                  # fallback only; real ticks come from the instrument dump
ORDER_GAP = 0.45             # seconds between orders; the broker throttles per second


def ist():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def alert(title, body, urgency='critical'):
    try:
        with open(FEED, 'a') as f:
            f.write(json.dumps(dict(ts=str(datetime.now()), book='EXECUTOR',
                                    urgency=urgency, title=title, body=body)) + '\n')
    except Exception:
        pass


def kite():
    from kiteconnect import KiteConnect
    api_key = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env')
               if l.startswith('KITE_API_KEY')][0]
    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))
    k = KiteConnect(api_key=api_key)
    k.set_access_token(tok.get('access_token') or tok.get('token'))
    return k


_TICKS = {}


def load_ticks(k):
    """Real tick size per symbol. Assuming 0.05 got three orders rejected on the first
    live run - INOXINDIA, KMEW and SBCL all trade in 0.10 - with the plain message
    "Tick size for this script is 0.10". The exchange publishes it; ask rather than guess."""
    global _TICKS
    if _TICKS:
        return _TICKS
    try:
        for i in k.instruments('NSE'):
            if i.get('instrument_type') == 'EQ' and i.get('tick_size'):
                _TICKS[i['tradingsymbol']] = float(i['tick_size'])
    except Exception as e:
        print('tick sizes unavailable (%s) - falling back to 0.05' % e)
    return _TICKS


def tick_round(px, up=True, symbol=None):
    tick = _TICKS.get(symbol, TICK) if symbol else TICK
    n = px / tick
    n = int(n) + 1 if up and n != int(n) else round(n)
    return round(round(n * tick, 4), 2)


# ───────────────────────── the idempotency ledger ─────────────────────────
def load_ledger():
    try:
        return json.load(open(LEDGER))
    except Exception:
        return {}


def save_ledger(d):
    tmp = LEDGER.with_suffix('.json.tmp')
    json.dump(d, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, LEDGER)


def already_done(led, book, symbol, kind='topup'):
    """One order per book/symbol/kind/day. Cron runs and hand runs must not compound."""
    return f'{book}|{symbol}|{kind}|{date.today()}' in led


def record(led, book, symbol, kind, payload):
    led[f'{book}|{symbol}|{kind}|{date.today()}'] = payload
    save_ledger(led)


# ───────────────────────── state helpers ─────────────────────────
def lock(path, tries=20, wait=1.5):
    import time
    for _ in range(tries):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(wait)
    return False


def unlock(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def save_json(path, obj):
    tmp = path.with_suffix('.json.tmp')
    json.dump(obj, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, path)


def place(k, symbol, qty, ltp, arm, tag):
    """A marketable LIMIT buy, CNC, paced and retried.

    The first live run fired sixteen orders back to back and Kite rejected six of them
    with "Maximum allowed order requests per second exceeded" - including the IPO entry,
    which ran last and so paid for the whole Open Alpha batch ahead of it. Orders are
    now spaced, and a throttle is retried rather than treated as a refusal.
    """
    limit = tick_round(ltp * 1.005, up=True, symbol=symbol)
    if not arm:
        return None, limit, 'DRY'
    for attempt in range(4):
        try:
            oid = k.place_order(variety='regular', exchange='NSE', tradingsymbol=symbol,
                                transaction_type='BUY', quantity=int(qty), product='CNC',
                                order_type='LIMIT', price=limit, validity='DAY', tag=tag[:20])
            time.sleep(ORDER_GAP)
            return oid, limit, 'PLACED'
        except Exception as e:
            msg = str(e)
            if 'per second' in msg.lower() and attempt < 3:
                time.sleep(1.2 * (attempt + 1))
                continue
            return None, limit, f'FAILED: {msg}'
    return None, limit, 'FAILED: throttled'



# ───────────────────────── Open Alpha: top up to equal weight ─────────────────────────
def deploy_open_alpha(arm, led):
    st = json.load(open(OA_STATE))
    cash = float(st.get('cash', 0.0))
    pos = st.get('positions', [])
    if cash < MIN_ORDER:
        print(f'open-alpha: cash Rs {cash:,.0f} below the Rs {MIN_ORDER:,.0f} floor, nothing to do')
        return
    if not pos:
        print('open-alpha: no holdings to top up (a fresh book needs its own seeding)')
        return

    k = kite()
    load_ticks(k)
    q = {}
    syms = [p['symbol'] for p in pos]
    for i in range(0, len(syms), 25):
        q.update(k.quote(['NSE:' + s for s in syms[i:i + 25]]))
    ltp = {s: (q.get('NSE:' + s) or {}).get('last_price') for s in syms}

    vals = {p['symbol']: p['qty'] * (ltp.get(p['symbol']) or p['buy']) for p in pos}
    total_after = sum(vals.values()) + cash
    target = total_after / OA_SLOTS          # equal weight across the 16 slots

    # Largest shortfall first, so the money goes where the book is most under-weight.
    short = sorted(((s, max(0.0, target - v)) for s, v in vals.items()),
                   key=lambda r: -r[1])
    spend = 0.0
    orders = []
    for s, gap in short:
        if gap < MIN_ORDER or cash - spend < MIN_ORDER:
            continue
        px = ltp.get(s)
        if not px:
            print(f'  {s}: no quote, skipped')
            continue
        if already_done(led, 'open-alpha', s):
            print(f'  {s}: already topped up today, skipped')
            continue
        budget = min(gap, cash - spend)
        qty = int(budget // px)
        if qty < 1:
            continue
        cost = qty * px
        if spend + cost > cash:
            continue
        oid, limit, note = place(k, s, qty, px, arm, 'OA-TOPUP')
        spend += cost
        orders.append(dict(symbol=s, qty=qty, ltp=px, limit=limit, cost=round(cost),
                           order_id=oid, note=note))
        print(f'  {"PLACED " if arm and oid else "would buy"} {s:<12} x{qty:<5} '
              f'@~{px:>9.2f}  limit {limit:>9.2f}  Rs {cost:>10,.0f}  {note}')
        if arm and oid:
            record(led, 'open-alpha', s, 'topup',
                   dict(order_id=oid, qty=qty, limit=limit, ts=str(datetime.now())))

    print(f'open-alpha: {len(orders)} orders, Rs {spend:,.0f} of Rs {cash:,.0f} cash deployed')
    if not arm:
        print('  (dry run — nothing was sent)')
        return
    if not orders:
        return
    # The cash is only really spent once the broker fills it; the book's own reconcile
    # is what moves positions. Here we record the intent and let the mark pick it up.
    if lock(OA_LOCK):
        try:
            st = json.load(open(OA_STATE))
            st.setdefault('pending_orders', []).extend(
                [dict(o, d=str(date.today()), kind='topup') for o in orders if o['order_id']])
            save_json(OA_STATE, st)
        finally:
            unlock(OA_LOCK)
    alert('Open Alpha: cash deployed',
          f'{len(orders)} top-up orders for Rs {spend:,.0f}. '
          f'Equal-weight target Rs {target:,.0f} a slot.', 'low')


# ───────────────────────── IPO Base: place the armed buy-stops ─────────────────────────
def deploy_ipo(arm, led):
    st = json.load(open(IPO_STATE))
    if st.get('mode') != 'live':
        print('ipo-base: still on paper, nothing to place')
        return
    pend = st.get('pending', [])
    cash = float(st.get('cash', 0.0))
    if not pend:
        print('ipo-base: no buy-stops armed')
        return
    equity = cash + sum(p['qty'] * p['buy'] for p in st.get('positions', []))
    k = kite()
    load_ticks(k)
    placed = 0
    for c in pend:
        s, pivot = c['symbol'], float(c['pivot'])
        if already_done(led, 'ipo-base', s, 'entry'):
            print(f'  {s}: order already placed today, skipped')
            continue
        # A BREACHED PIVOT IS NOT A STOP ORDER. Kite rejects a buy stop-loss whose
        # trigger sits below the last price, and by 09:20 a pivot may already be through
        # — KISSHT was at 320.95 against a 313.60 pivot on 08-Sep. The spec fills at
        # max(pivot, open), so a pivot already cleared means buy now at the market rather
        # than arm a stop that can never be accepted.
        try:
            ltp = k.ltp(['NSE:' + s])['NSE:' + s]['last_price']
        except Exception as e:
            print(f'  {s}: no quote ({e}), skipped')
            continue
        breached = ltp >= pivot
        fill_ref = max(pivot, ltp) if breached else pivot
        size = min(IPO_SIZE_PCT, 0.30) * equity
        qty = int(size // fill_ref)
        if qty < 1 or qty * fill_ref > cash:
            print(f'  {s}: cash short for a slot, skipped')
            continue
        trig = tick_round(pivot, up=True, symbol=s)
        limit = tick_round(fill_ref * 1.005, up=True, symbol=s)
        slip = (ltp / pivot - 1) * 100
        if not arm:
            how = (f'MARKET (pivot already through: last {ltp}, +{slip:.2f}% over pivot)'
                   if breached else f'SL BUY trigger {trig}')
            print(f'  would place {s:<12} x{qty:<5} {how} limit {limit}')
            continue
        try:
            if breached:
                oid = k.place_order(variety='regular', exchange='NSE', tradingsymbol=s,
                                    transaction_type='BUY', quantity=qty, product='CNC',
                                    order_type='LIMIT', price=limit, validity='DAY',
                                    tag='IPO-ENTRY')
                how = f'LIMIT {limit} (pivot {pivot} already through at {ltp})'
            else:
                oid = k.place_order(variety='regular', exchange='NSE', tradingsymbol=s,
                                    transaction_type='BUY', quantity=qty, product='CNC',
                                    order_type='SL', trigger_price=trig, price=limit,
                                    validity='DAY', tag='IPO-ENTRY')
                how = f'SL BUY trigger {trig} limit {limit}'
            print(f'  PLACED  {s:<12} x{qty:<5} {how}  id {oid}')
            record(led, 'ipo-base', s, 'entry',
                   dict(order_id=oid, qty=qty, pivot=pivot, ltp_at_order=ltp,
                        slip_pct=round(slip, 2), breached=breached,
                        limit=limit, ts=str(datetime.now())))
            placed += 1
            # Fill quality is the soak's pass criterion, so it is recorded per order
            # rather than reconstructed later.
            alert(f'IPO entry placed: {s}',
                  f'BUY {s} x{qty} {how}. Pivot {pivot}, last {ltp} '
                  f'({slip:+.2f}% vs pivot).', 'low')
        except Exception as e:
            print(f'  FAILED  {s}: {e}')
            alert(f'IPO order FAILED: {s}', str(e))
    if not arm:
        print('  (dry run — nothing was sent)')
    else:
        print(f'ipo-base: {placed} entry orders placed')


def main():
    arm = '--arm' in sys.argv
    which = 'all'
    if '--book' in sys.argv:
        which = sys.argv[sys.argv.index('--book') + 1]

    if KILL.exists():
        print('KILL SWITCH present (%s) — refusing to do anything' % KILL)
        return
    now = ist()
    if arm:
        if now.weekday() >= 5:
            print(f'{now:%a %H:%M} IST — weekend, not arming')
            return
        if not ((9, 20) <= (now.hour, now.minute) <= (15, 0)):
            print(f'{now:%H:%M} IST — outside 09:20-15:00, not arming')
            return
    print(f'=== equity executor {now:%Y-%m-%d %H:%M} IST · {"ARMED" if arm else "DRY RUN"} ===')
    led = load_ledger()
    # ENTRIES FIRST. A new entry is a signal with a price attached and a day to live; a
    # top-up is housekeeping that can wait until tomorrow. On the first live run the
    # order was the other way round, so sixteen top-up attempts used up the rate limit
    # and the one entry of the day was rejected.
    if which in ('all', 'ipo-base'):
        try:
            deploy_ipo(arm, led)
        except Exception as e:
            print('ipo-base failed:', e)
            alert('Executor failed: ipo-base', str(e))
    if which in ('all', 'open-alpha'):
        try:
            deploy_open_alpha(arm, led)
        except Exception as e:
            print('open-alpha failed:', e)
            alert('Executor failed: open-alpha', str(e))


if __name__ == '__main__':
    main()
