# -*- coding: utf-8 -*-
"""One daily check over the whole Momentum Portfolio, so a silent failure cannot last a day.

WHY THIS EXISTS. On 09-Sep the entry scan crashed on a regex that ate a full stop. Nothing
was watching, so Open Alpha sat at 14 of 16 slots with about Rs78,000 idle until Arun
happened to look at the page and ask. Every individual part of this system already alerts;
what was missing was something that asks, once a day, whether the parts actually ran.

WHAT IT CHECKS, per book (True North, Open Alpha, IPO Base) and across the portfolio:

    feed freshness      is the page showing today's prices, or yesterday's?
    slots               held + armed == slots, or the book is under-deployed
    exits due           anything past its stop or trail with no sell placed
    order failures      anything rejected or unfilled
    ledger square       does cash agree with the book's own P&L
    scheduled jobs      did each cron job actually run today, and without a traceback
    alert channels      can this system reach Arun at all

It writes static/app/mpf_health.json for the app, prints a text report, and on anything
worse than OK sends the same summary by email and WhatsApp.

Deliberately READ-ONLY. It reads state, feeds, logs and the order book, and changes
nothing. A checker that repairs things is a checker you stop trusting to tell you the truth.

    python3 scripts/mpf_health.py            # check, write, print
    python3 scripts/mpf_health.py --send     # ... and notify if not all-OK
    python3 scripts/mpf_health.py --send-always
"""
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

APP = ROOT / 'static' / 'app'
OUT = APP / 'mpf_health.json'
COST_PCT = 0.0025
UNRECONCILED_FLOOR = 250        # below this it is rounding, same floor the page uses

OK, WARN, FAIL = 'OK', 'WARN', 'FAIL'
RANK = {OK: 0, WARN: 1, FAIL: 2}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, area, name, status, detail):
        self.rows.append(dict(area=area, name=name, status=status, detail=detail))

    @property
    def worst(self):
        return max((r['status'] for r in self.rows), key=lambda s: RANK[s], default=OK)


def _load(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception:
        return default


def _age_min(ts):
    """Minutes since a naive IST timestamp string, or None if unparseable."""
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace(' ', 'T').split('.')[0])
    except Exception:
        return None
    return (datetime.now() - t).total_seconds() / 60.0


def _market_open_now():
    n = datetime.now()
    if n.weekday() > 4:
        return False
    return (9 * 60 + 15) <= (n.hour * 60 + n.minute) <= (15 * 60 + 30)


# ── the books ────────────────────────────────────────────────────────────────
BOOKS = [
    dict(id='tn', label='True North', feed='momentum_live.json',
         state='momentum_state.json', slots_key='slots', held_key='n'),
    dict(id='oa', label='Open Alpha', feed='oa_real.json',
         slots_key='slots', held_key='slots_used'),
    dict(id='ipo', label='IPO Base', feed='ipo_paper.json',
         slots_key='slots', held_key='slots_used'),
]


def check_books(rep, kite_orders):
    """-> the per-book summary the app renders."""
    summary = []
    for b in BOOKS:
        feed = _load(APP / b['feed'])
        st = _load(APP / b['state']) if b.get('state') else feed
        if not feed:
            rep.add(b['label'], 'feed', FAIL, '%s is missing or unreadable' % b['feed'])
            continue

        # ---- freshness ----
        age = _age_min(feed.get('updated'))
        limit = 10 if _market_open_now() else 24 * 60
        if age is None:
            rep.add(b['label'], 'feed fresh', WARN, 'no usable timestamp')
        elif age > limit:
            rep.add(b['label'], 'feed fresh', FAIL,
                    'last marked %.0f min ago (limit %d)' % (age, limit))
        else:
            rep.add(b['label'], 'feed fresh', OK, 'marked %.0f min ago' % age)

        # ---- slots ----
        held = feed.get(b['held_key'])
        slots = feed.get(b['slots_key'])
        armed = [o for o in kite_orders
                 if o.get('tag') == 'OA-ENTRY'
                 and o.get('status') not in ('REJECTED', 'CANCELLED')]
        if held is not None and slots:
            spare = slots - held
            if spare <= 0:
                rep.add(b['label'], 'slots', OK, '%d/%d, full' % (held, slots))
            elif b['id'] == 'tn':
                # Empty slots refill at the MONTH-END rebalance, never before: research/108
                # found refilling early buys names that are already falling.
                d = (st or {}).get('days_to_rebalance')
                rep.add(b['label'], 'slots', OK,
                        '%d/%d held, %d free - refills at the rebalance%s'
                        % (held, slots, spare, ' in %dd' % d if d is not None else ''))
            elif b['id'] == 'ipo':
                # Scarce by construction: r/153 measured 32.7% average deployment across the
                # whole backtest, and r/155 concluded the idle cash should stay idle.
                rep.add(b['label'], 'slots', OK,
                        '%d/%d held, %d free - listings are scarce by design' % (held, slots, spare))
            elif len(armed) >= spare:
                rep.add(b['label'], 'slots', OK,
                        '%d/%d held, %d armed for the next session' % (held, slots, len(armed)))
            elif datetime.now().hour < 19:
                rep.add(b['label'], 'slots', OK,
                        '%d/%d held, %d free - tonight\'s scan arms them' % (held, slots, spare))
            else:
                # The one that matters: Open Alpha refills from the evening scan, and this
                # is precisely the failure that went unnoticed for a day on 09-Sep.
                rep.add(b['label'], 'slots', FAIL,
                        '%d/%d held, %d free and NOTHING armed after the evening scan'
                        % (held, slots, spare))

        # ---- exits due but not placed ----
        due, unplaced = [], []
        for p in feed.get('positions') or []:
            d = [x for x in (p.get('to_stop_pct'), p.get('to_trail_pct')) if x is not None]
            if d and min(d) < 0:
                due.append(p['symbol'])
                if not any(o.get('tradingsymbol') == p['symbol']
                           and o.get('transaction_type') == 'SELL'
                           and o.get('status') not in ('REJECTED', 'CANCELLED')
                           for o in kite_orders):
                    unplaced.append(p['symbol'])
        if unplaced:
            rep.add(b['label'], 'exits due', FAIL,
                    'past the rule with NO sell placed: ' + ', '.join(unplaced))
        elif due:
            rep.add(b['label'], 'exits due', OK, 'placed: ' + ', '.join(due))
        else:
            rep.add(b['label'], 'exits due', OK, 'none')

        # ---- failed orders ----
        bad = feed.get('failed_orders') or []
        rep.add(b['label'], 'order failures', FAIL if bad else OK,
                ('%d unfilled: ' % len(bad)) + ', '.join(str(x.get('symbol', '?')) for x in bad)
                if bad else 'none')

        # ---- ledger square ----
        if b['id'] == 'tn' and st:
            gain = st['nav'] - st['capital']
            parts = (st['unrealized'] + st['realized_net'] + st.get('interest_earned', 0)
                     - COST_PCT * (st['equity'] - st['unrealized']))
        else:
            invested = feed.get('invested', (feed.get('value', 0) - feed.get('pnl', 0)))
            gain = feed.get('gain', 0)
            parts = feed.get('pnl', 0) + feed.get('realized', 0) - COST_PCT * invested
        gap = gain - parts
        rep.add(b['label'], 'ledger square',
                OK if abs(gap) < UNRECONCILED_FLOOR else WARN,
                'unreconciled Rs %+.0f' % gap)

        summary.append(dict(id=b['id'], label=b['label'], held=held, slots=slots,
                            nav=feed.get('nav'), gain=feed.get('gain'),
                            updated=feed.get('updated'), age_min=round(age) if age else None,
                            due=due, unplaced=unplaced, failed=len(bad),
                            unreconciled=round(gap)))
    return summary


# ── the scheduled jobs that keep it all moving ───────────────────────────────
JOBS = [
    ('True North marks',   '/tmp/momentum_live.log',   'market-hours'),
    ('True North state',   '/tmp/momentum_state.log',  'market-hours'),
    ('Open Alpha marks',   '/tmp/oa_real_mark.log',    'market-hours'),
    ('Open Alpha exits',   '/tmp/oa_real.log',         'daily'),
    ('Open Alpha entries', '/tmp/oa_entry.log',        'evening'),
    ('Open Alpha recon',   '/tmp/oa_reconcile.log',    'daily'),
    ('IPO marks',          '/tmp/ipo_mark.log',        'market-hours'),
    ('IPO engine',         '/tmp/ipo_paper.log',       'evening'),
    ('Cash executor',      '/tmp/equity_executor.log', 'daily'),
    ('Universe refresh',   '/tmp/universe_refresh.log', 'evening'),
]
ERR = re.compile(r'Traceback \(most recent call last\)|^\w*Error:|Exception', re.M)


def check_jobs(rep):
    today = date.today()
    for label, path, cadence in JOBS:
        p = Path(path)
        if not p.exists():
            rep.add('Jobs', label, WARN, 'no log at %s - has it ever run?' % path)
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        stale = mtime.date() < today
        # An evening job has not run yet if we are checking before it fires; say so rather
        # than calling it broken.
        if stale and cadence == 'evening' and datetime.now().hour < 19:
            rep.add('Jobs', label, OK, 'due later today (last %s)' % mtime.strftime('%d-%b %H:%M'))
            continue
        tail = ''
        try:
            tail = p.read_text(encoding='utf-8', errors='replace')[-4000:]
        except Exception:
            pass
        if ERR.search(tail):
            first = next((l for l in tail.splitlines()[::-1]
                          if 'Error' in l or 'Exception' in l), 'see the log')
            rep.add('Jobs', label, FAIL, 'ERRORED: %s' % first.strip()[:120])
        elif stale:
            rep.add('Jobs', label, FAIL, 'has not run today (last %s)'
                    % mtime.strftime('%d-%b %H:%M'))
        else:
            rep.add('Jobs', label, OK, 'ran %s' % mtime.strftime('%H:%M'))


def check_channels(rep):
    # Ask the sender what it can do; do not re-derive its rules here and drift from them.
    try:
        from services.dividend_notify import _first_env
        pw = _first_env('EMAIL_SMTP_PASS', 'GMAIL_APP_PASSWORD')
        to = _first_env('EMAIL_TO', 'EMAIL_SMTP_USER')
        have_mail = bool(pw and to)
    except Exception:
        have_mail = False
    rep.add('Alerts', 'email', OK if have_mail else WARN,
            'configured' if have_mail else 'DORMANT - set EMAIL_TO (a Gmail app password is already on file)')
    phones = []
    if os.getenv('NTFY_TOPIC'):
        phones.append('ntfy')
    if os.getenv('TELEGRAM_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
        phones.append('telegram')
    if os.getenv('WHATSAPP_ENABLED') == '1':
        phones.append('whatsapp')
    rep.add('Alerts', 'phone', OK if phones else WARN,
            ', '.join(phones) if phones
            else 'DORMANT - no phone channel. NTFY_TOPIC is the free one, no account needed')


def text_report(rep, books):
    L = ['MOMENTUM PORTFOLIO - daily check  %s' % datetime.now().strftime('%d-%b-%Y %H:%M'),
         'overall: %s' % rep.worst, '']
    for b in books:
        L.append('%-11s %s/%s slots   nav Rs %s   %s'
                 % (b['label'], b['held'], b['slots'],
                    format(b['nav'] or 0, ','),
                    'marked %s min ago' % b['age_min'] if b['age_min'] is not None else ''))
    L.append('')
    area = None
    for r in rep.rows:
        if r['area'] != area:
            area = r['area']
            L.append(area)
        flag = '  ' if r['status'] == OK else ('! ' if r['status'] == WARN else 'X ')
        L.append('%s%-16s %s' % (flag, r['name'], r['detail']))
    return '\n'.join(L)


def main():
    kite_orders = []
    try:
        from services.oa_real import _kite
        kite_orders = _kite().orders()
    except Exception as e:
        print('order book unreachable (checks that need it will say so):', e)

    rep = Report()
    books = check_books(rep, kite_orders)
    check_jobs(rep)
    check_channels(rep)

    payload = dict(generated=str(datetime.now()), overall=rep.worst,
                   books=books, checks=rep.rows)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, OUT)

    body = text_report(rep, books)
    print(body)
    print()
    print('-> %s (overall %s)' % (OUT, rep.worst))

    send = '--send-always' in sys.argv or ('--send' in sys.argv and rep.worst != OK)
    if send:
        try:
            from services.dividend_notify import send_email, send_push
            title = 'Quantifyd mpf check: %s' % rep.worst
            print('email:', send_email(title, '<pre>%s</pre>' % body))
            print('push:', send_push(title, body[:1200]))
        except Exception as e:
            print('notify failed:', e)

    raise SystemExit(0 if rep.worst != FAIL else 1)


if __name__ == '__main__':
    main()
