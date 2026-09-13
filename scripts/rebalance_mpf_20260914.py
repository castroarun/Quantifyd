#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot rebalance of the three Momentum Portfolio books to TN 37.5 / OA 37.5 / IPO 25.

Arun, 13-Sep-2026: "now that entire TN is in cash fund, can v now make the distribution in the
correct ratio?" — approved as "fix IPO, move all Monday". This is a ONE-OFF override of the
05-Sep rule that True North is never sold down to rebalance; the deposit router keeps that
rule for new money.

WHY A SCRIPT, AND WHY IT IS THIS CAREFUL. It moves real money. True North has to redeem about
Rs 2.26L of CASHIETF, which needs market hours, so it cannot run on the Sunday it was approved.
And True North's own withdrawal has a hazard for unattended use: if its cash-fund sale fails,
cash_withdraw() logs the error and still cuts the book's cash and capital by the full amount
while the units stay held. So this job:

  1. values every book from its own freshly baked feed (refreshed every 1-3 minutes in market
     hours) and computes the transfer on CURRENT VALUE, not on money put in;
  2. refuses unless True North holds no stock, the amounts are within 15% of the plan Arun saw,
     the targets are 37.5/37.5/25, and the IPO engine carries the funding-sync fix (without it
     the IPO credit would never reach the cash that sizes its buys);
  3. dry-runs all three legs through the Capital Desk;
  4. REDEEMS THE UNITS AS ITS OWN STEP and confirms the units actually fell and True North's
     cash now covers the transfer — before any ledger is cut;
  5. dry-runs True North's withdrawal AGAIN and requires a plan paid entirely from cash, so the
     app process cannot redeem a second time on a stale view of the cash;
  6. withdraws from True North, then credits Open Alpha and IPO, verifying each;
  7. stops with a critical alert at the first failure, saying exactly what state it left.

It runs once: a marker file blocks a second execution.

Usage:
  rebalance_mpf_20260914.py --dry [--allow-stale]   plan and dry-run only; changes nothing
  rebalance_mpf_20260914.py --execute               the real move; date- and window-locked
"""
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT))
API = 'http://127.0.0.1:5000'
LOGDIR = ROOT / 'logs'
RESULT = LOGDIR / 'rebalance_mpf_20260914.json'
DONE = LOGDIR / 'rebalance_mpf_20260914.done'
FEED = ROOT / 'backtest_data' / 'book_alerts.jsonl'

RUN_DATE = '2026-09-14'
WINDOW = ('09:40', '14:30')          # after the open settles; before TN's 14:45 / 15:05 jobs
TARGETS = {'truenorth': 0.375, 'openalpha': 0.375, 'ipo': 0.25}
PLANNED_TN_OUT = 253616              # the figure Arun approved, on 13-Sep values
TOLERANCE = 0.15
MAX_STALE_MIN = 10
NOT_STOCK = {'CASHIETF', 'Un-swept cash'}

log = []


def say(msg):
    line = '%s  %s' % (datetime.now().strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    log.append(line)


def alert(title, body, critical=True):
    try:
        with open(FEED, 'a') as f:
            f.write(json.dumps(dict(ts=str(datetime.now()), book='CAPITAL-DESK',
                                    urgency='critical' if critical else 'low',
                                    title=title, body=body)) + '\n')
    except Exception as e:
        say('alert feed write failed: %s' % e)
    if not critical:
        return
    try:
        from services.dividend_notify import send_email, send_push
        send_email(title, '<pre>%s</pre>' % body)
        send_push(title, body)
    except Exception as e:
        say('alert delivery failed: %s' % e)


def save(status, **extra):
    LOGDIR.mkdir(exist_ok=True)
    json.dump(dict(status=status, at=str(datetime.now()), log=log, **extra),
              open(RESULT, 'w'), indent=1, default=str)


def stop(title, body, **extra):
    say('STOP: %s — %s' % (title, body))
    alert('REBALANCE STOPPED: ' + title, body)
    save('stopped', reason=title, detail=body, **extra)
    sys.exit(1)


def post(path, body, timeout=300):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or '{}')
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or '{}')
        except Exception:
            return e.code, {}


def minutes_old(ts):
    try:
        return (datetime.now() - datetime.fromisoformat(str(ts)[:19])).total_seconds() / 60
    except Exception:
        return 1e9


def values(allow_stale):
    tn = json.load(open(ROOT / 'static/app/momentum_state.json'))
    oa = json.load(open(ROOT / 'static/app/oa_real.json'))
    ipo = json.load(open(ROOT / 'static/app/ipo_paper.json'))
    fresh = {'truenorth': minutes_old(tn.get('baked')),
             'openalpha': minutes_old(oa.get('updated')),
             'ipo': minutes_old(ipo.get('updated'))}
    for k, m in fresh.items():
        say('  %-9s value feed is %.0f minutes old' % (k, m))
        if m > MAX_STALE_MIN and not allow_stale:
            stop('stale value', '%s feed is %.0f minutes old (limit %d)' % (k, m, MAX_STALE_MIN))
    holdings = tn.get('holdings') or tn.get('positions') or []
    stocks = [h.get('symbol') for h in holdings if h.get('symbol') not in NOT_STOCK]
    if stocks:
        stop('True North holds stock', 'expected only CASHIETF and cash, found %s' % stocks)
    return {'truenorth': float(tn['nav']), 'openalpha': float(oa['nav']),
            'ipo': float(ipo['nav'])}


def plan_amounts(v):
    total = sum(v.values())
    oa_in = round(TARGETS['openalpha'] * total - v['openalpha'])
    ipo_in = round(TARGETS['ipo'] * total - v['ipo'])
    tn_out = oa_in + ipo_in                      # exact, so the three ledgers balance
    say('  values: TN Rs %s  OA Rs %s  IPO Rs %s  total Rs %s' % tuple(
        format(round(x), ',') for x in (v['truenorth'], v['openalpha'], v['ipo'], total)))
    say('  move:   TN -Rs %s   OA +Rs %s   IPO +Rs %s' % tuple(
        format(x, ',') for x in (tn_out, oa_in, ipo_in)))
    if oa_in <= 0 or ipo_in <= 0:
        stop('nothing to move', 'Open Alpha or IPO is already at or above target: OA %+d IPO %+d'
             % (oa_in, ipo_in))
    drift = abs(tn_out - PLANNED_TN_OUT) / PLANNED_TN_OUT
    if drift > TOLERANCE:
        stop('amount moved too far from the approved plan',
             'True North would send Rs %s against the Rs %s Arun approved (%.0f%% apart, limit '
             '%.0f%%). Values changed more than expected; re-approve.'
             % (format(tn_out, ','), format(PLANNED_TN_OUT, ','), drift * 100, TOLERANCE * 100))
    return tn_out, oa_in, ipo_in, round(total)


def preconditions():
    a = json.load(open(ROOT / 'backtest_data/allocation_targets.json'))
    if {k: round(float(x), 4) for k, x in a['targets'].items()} != TARGETS:
        stop('targets changed', 'allocation targets are %s, expected %s' % (a['targets'], TARGETS))
    src = (ROOT / 'services/ipo_paper.py').read_text(encoding='utf-8')
    if 'FUNDING SYNC' not in src:
        stop('IPO funding fix missing',
             'services/ipo_paper.py has no FUNDING SYNC block; an IPO credit would not reach '
             'the cash that sizes its buys')


def dry_legs(tn_out, oa_in, ipo_in):
    c1, r1 = post('/api/sleeves/truenorth/withdraw', dict(amount=tn_out, dry_run=True))
    c2, r2 = post('/api/sleeves/openalpha/deposit', dict(amount=oa_in, dry_run=True))
    c3, r3 = post('/api/sleeves/ipo/deposit', dict(amount=ipo_in, dry_run=True))
    say('  dry run TN  HTTP %s ok=%s shortfall=%s plan=%s'
        % (c1, r1.get('ok'), r1.get('shortfall'), r1.get('plan')))
    say('  dry run OA  HTTP %s ok=%s capital_after=%s' % (c2, r2.get('ok'), r2.get('capital_after')))
    say('  dry run IPO HTTP %s ok=%s capital_after=%s' % (c3, r3.get('ok'), r3.get('capital_after')))
    if c1 != 200 or not r1.get('ok') or float(r1.get('shortfall', 1) or 0) > 1:
        stop('True North dry run failed', json.dumps(r1)[:600])
    if any(p.get('action') == 'SELL' for p in r1.get('plan', [])):
        stop('True North would sell stock', json.dumps(r1.get('plan'))[:600])
    if c2 != 200 or not r2.get('ok'):
        stop('Open Alpha dry run failed', json.dumps(r2)[:600])
    if c3 != 200 or not r3.get('ok'):
        stop('IPO dry run failed', json.dumps(r3)[:600])
    return r1


def execute(tn_out, oa_in, ipo_in, total):
    import services.momentum_paper as mp

    cap0, cash0, units0 = float(mp._get('capital', 0) or 0), mp._cash(), mp._sweep_units()
    say('True North ledger before: capital Rs %s  cash Rs %s  CASHIETF units %s'
        % (format(round(cap0), ','), format(round(cash0), ','), units0))

    # ---- A. redeem the units as its own step, and prove it happened ----
    need = tn_out - cash0
    if need > 1:
        say('A. redeeming about Rs %s of CASHIETF' % format(round(need), ','))
        r = mp.unsweep(need)
        cash1, units1 = mp._cash(), mp._sweep_units()
        say('   result %s; cash now Rs %s, units now %s' % (r, format(round(cash1), ','), units1))
        if not r or units1 >= units0:
            stop('CASHIETF sale did not complete',
                 'Nothing was withdrawn and no ledger was cut. True North still holds %s units.'
                 % units1, before=dict(capital=cap0, cash=cash0, units=units0))
        if cash1 < tn_out - 1:
            stop('sale proceeds short of the transfer',
                 'CASHIETF was sold and the proceeds sit in True North cash (Rs %s), but that is '
                 'below the Rs %s transfer. No withdrawal made; True North is consistent.'
                 % (format(round(cash1), ','), format(tn_out, ',')))
    else:
        say('A. True North cash already covers the transfer; nothing to redeem')

    # ---- B. the app must now see a withdrawal paid entirely from cash ----
    c, r = post('/api/sleeves/truenorth/withdraw', dict(amount=tn_out, dry_run=True))
    sources = [p.get('source') for p in r.get('plan', [])]
    say('B. re-check dry run: %s' % r.get('plan'))
    if c != 200 or not r.get('ok') or sources != ['idle cash']:
        stop('True North would not pay from cash alone',
             'After redeeming, the app still plans %s. Proceeds sit in True North cash; no '
             'withdrawal made, so nothing can be redeemed twice.' % r.get('plan'))

    # ---- C. withdraw from True North ----
    c, r = post('/api/sleeves/truenorth/withdraw', dict(amount=tn_out, dry_run=False))
    cap2 = float(mp._get('capital', 0) or 0)
    say('C. True North withdraw HTTP %s ok=%s raised=%s; capital Rs %s -> Rs %s'
        % (c, r.get('ok'), r.get('raised'), format(round(cap0), ','), format(round(cap2), ',')))
    if c != 200 or not r.get('ok') or abs((cap0 - cap2) - tn_out) > 2:
        stop('True North withdrawal did not record correctly',
             'HTTP %s response %s; capital moved by Rs %s against Rs %s expected. Open Alpha and '
             'IPO were NOT credited.' % (c, json.dumps(r)[:400], round(cap0 - cap2), tn_out))

    failures = []
    # ---- D. credit Open Alpha ----
    c, r = post('/api/sleeves/openalpha/deposit', dict(amount=oa_in, dry_run=False))
    say('D. Open Alpha deposit HTTP %s ok=%s capital_after=%s' % (c, r.get('ok'), r.get('capital_after')))
    if c != 200 or not r.get('ok'):
        failures.append('Open Alpha deposit of Rs %s FAILED (%s) — credit it by hand on the '
                        'Capital Desk' % (format(oa_in, ','), json.dumps(r)[:200]))
    # ---- E. credit IPO ----
    c, r = post('/api/sleeves/ipo/deposit', dict(amount=ipo_in, dry_run=False))
    say('E. IPO deposit HTTP %s ok=%s capital_after=%s' % (c, r.get('ok'), r.get('capital_after')))
    if c != 200 or not r.get('ok'):
        failures.append('IPO deposit of Rs %s FAILED (%s) — credit it by hand on the Capital '
                        'Desk' % (format(ipo_in, ','), json.dumps(r)[:200]))

    DONE.write_text(str(datetime.now()))
    if failures:
        stop('True North withdrew but a credit failed', ' | '.join(failures),
             moved=dict(tn_out=tn_out, oa_in=oa_in, ipo_in=ipo_in))

    # ---- F. record the one-off override on the Capital Desk changelog ----
    note = ('14-Sep-2026 ONE-OFF REBALANCE (Arun, approved 13-Sep): True North withdrew Rs %s, '
            'Open Alpha +Rs %s, IPO +Rs %s, on current values totalling Rs %s, to land on 37.5 / '
            '37.5 / 25. This overrides, for this transfer only, the 05-Sep rule that True North '
            'is never sold down to rebalance; the deposit router keeps that rule for new money.'
            % (format(tn_out, ','), format(oa_in, ','), format(ipo_in, ','), format(total, ',')))
    post('/api/sleeves/allocation/targets', dict(targets=TARGETS, note=note))

    body = ('True North -Rs %s (CASHIETF redeemed, capital now Rs %s). Open Alpha +Rs %s: sits as '
            'cash until you buy. IPO +Rs %s: reaches its buying cash at the 18:45 run. Book is now '
            'on 37.5 / 37.5 / 25 by value.' % (format(tn_out, ','), format(round(cap2), ','),
                                                format(oa_in, ','), format(ipo_in, ',')))
    say('DONE. ' + body)
    alert('Rebalance done: 37.5 / 37.5 / 25', body)
    save('done', moved=dict(tn_out=tn_out, oa_in=oa_in, ipo_in=ipo_in, total=total))


def main():
    dry = '--dry' in sys.argv
    if not dry and '--execute' not in sys.argv:
        print(__doc__)
        sys.exit(2)
    say('rebalance to 37.5 / 37.5 / 25 — %s' % ('DRY RUN' if dry else 'EXECUTE'))
    if not dry:
        now = datetime.now()
        hm = now.strftime('%H:%M')
        if now.strftime('%Y-%m-%d') != RUN_DATE or not (WINDOW[0] <= hm <= WINDOW[1]):
            stop('outside the approved window', 'now %s, allowed %s %s-%s'
                 % (now.strftime('%Y-%m-%d %H:%M'), RUN_DATE, *WINDOW))
        if DONE.exists():
            stop('already ran', 'marker %s exists (%s)' % (DONE, DONE.read_text()))
    preconditions()
    v = values(allow_stale=dry and '--allow-stale' in sys.argv)
    tn_out, oa_in, ipo_in, total = plan_amounts(v)
    dry_legs(tn_out, oa_in, ipo_in)
    if dry:
        say('DRY RUN complete — nothing changed')
        return
    execute(tn_out, oa_in, ipo_in, total)


if __name__ == '__main__':
    main()
