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
ORDER_GAP = 0.60             # minimum gap between ANY two orders
RETRY_BACKOFF = 1.5          # grows per attempt when the broker says slow down


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


FAILURES = ROOT / 'backtest_data' / 'executor_failures.json'
_FAILED = []          # collected across both books, reported once at the end


def note_failure(book, symbol, qty, reason, detail=''):
    """Record a rejected or unplaced order.

    On 08-Sep nine of sixteen orders were rejected and the only trace was a line in a log
    file nobody reads. Three were on the wrong tick and six were throttled, including the
    day's only entry signal. A book that quietly fails to place its trades looks exactly
    like a book with nothing to do.
    """
    _FAILED.append(dict(ts=str(datetime.now()), book=book, symbol=symbol,
                        qty=qty, reason=reason, detail=detail[:300]))


def flush_failures(arm):
    """Persist, alert, and push onto the book feeds so the pages show it."""
    if not _FAILED:
        # A CLEAN RUN MUST CLEAR THE BANNER EVERYWHERE. Clearing only this file left the
        # book pages showing a failure that had since succeeded: KISSHT was reported as
        # not placed while the account already held it, because the page reads
        # `failed_orders` off the book feed and nothing ever reset it.
        if arm:
            save_json(FAILURES, dict(d=str(date.today()), items=[]))
            for sp in (OA_STATE, IPO_STATE):
                try:
                    if not sp.exists():
                        continue
                    st = json.load(open(sp))
                    if st.get('failed_orders'):
                        st['failed_orders'] = []
                        save_json(sp, st)
                except Exception as e:
                    print('  could not clear the banner on %s: %s' % (sp.name, e))
        return
    save_json(FAILURES, dict(d=str(date.today()), items=_FAILED))
    lines = ['%s %s x%s — %s' % (f['book'], f['symbol'], f['qty'], f['reason'])
             for f in _FAILED]
    body = chr(10).join(lines)
    title = '%d order%s did NOT go through' % (len(_FAILED), '' if len(_FAILED) == 1 else 's')
    alert(title, body)
    try:
        sys.path.insert(0, str(ROOT))
        from services.dividend_notify import send_email, send_push
        html = ('<h3>%s</h3><p>Quantifyd executor, %s IST</p><ul>%s</ul>'
                '<p>These trades are NOT in the account. The book will not hold them '
                'unless they are placed.</p>'
                % (title, ist().strftime('%d-%b-%Y %H:%M'),
                   ''.join('<li>%s</li>' % l for l in lines)))
        print('  email:', send_email('Quantifyd: ' + title, html))
        print('  push:', send_push('Quantifyd - ' + title, body))
    except Exception as e:
        print('  notification failed:', e)
    # surface on the book pages: each feed carries what failed for that book
    for state_path, book in ((OA_STATE, 'open-alpha'), (IPO_STATE, 'ipo-base')):
        mine = [f for f in _FAILED if f['book'] == book]
        if not mine or not state_path.exists():
            continue
        try:
            st = json.load(open(state_path))
            st['failed_orders'] = mine
            save_json(state_path, st)
        except Exception as e:
            print('  could not write failures to %s: %s' % (book, e))


_LAST_ORDER = [0.0]


def send_order(k, **kw):
    """EVERY order in this file goes through here. Returns (order_id, error).

    Pacing was first written inside the Open Alpha helper, which left the IPO leg
    calling place_order directly with no spacing and no retry — and the IPO entry was
    exactly what the broker rejected on 08-Sep with "Maximum allowed order requests per
    second exceeded". Scattered pacing is pacing that a new code path will forget, so
    there is now one gate and no way around it.

    Two mechanisms, because they solve different problems: the throttle spaces orders so
    the limit is not hit, and the backoff recovers when something else in the account has
    already consumed the quota.
    """
    for attempt in range(5):
        wait = ORDER_GAP - (time.monotonic() - _LAST_ORDER[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_ORDER[0] = time.monotonic()
        try:
            return k.place_order(**kw), None
        except Exception as e:
            msg = str(e)
            throttled = 'per second' in msg.lower() or 'too many' in msg.lower()
            if throttled and attempt < 4:
                back = RETRY_BACKOFF * (attempt + 1)
                print('    throttled, retrying in %.1fs (attempt %d)' % (back, attempt + 2))
                time.sleep(back)
                continue
            return None, msg
    return None, 'still throttled after retries'


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
    oid, err = send_order(k, variety='regular', exchange='NSE', tradingsymbol=symbol,
                          transaction_type='BUY', quantity=int(qty), product='CNC',
                          order_type='LIMIT', price=limit, validity='DAY', tag=tag[:20])
    return (oid, limit, 'PLACED') if oid else (None, limit, f'FAILED: {err}')



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
        if arm and not oid:
            note_failure('open-alpha', s, qty, 'not placed', note)
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
            note_failure('ipo-base', s, qty, 'skipped: not enough cash for a slot',
                         'needs Rs %.0f, book has Rs %.0f' % (qty * fill_ref, cash))
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
                oid, err = send_order(k, variety='regular', exchange='NSE', tradingsymbol=s,
                                      transaction_type='BUY', quantity=qty, product='CNC',
                                      order_type='LIMIT', price=limit, validity='DAY',
                                      tag='IPO-ENTRY')
                how = f'LIMIT {limit} (pivot {pivot} already through at {ltp})'
            else:
                oid, err = send_order(k, variety='regular', exchange='NSE', tradingsymbol=s,
                                      transaction_type='BUY', quantity=qty, product='CNC',
                                      order_type='SL', trigger_price=trig, price=limit,
                                      validity='DAY', tag='IPO-ENTRY')
                how = f'SL BUY trigger {trig} limit {limit}'
            if not oid:
                raise RuntimeError(err)
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
            note_failure('ipo-base', s, qty, 'entry not placed', str(e))
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
            note_failure('open-alpha', '-', 0, 'the whole leg failed', str(e))
    flush_failures(arm)


if __name__ == '__main__':
    main()
