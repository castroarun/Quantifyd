# -*- coding: utf-8 -*-
"""Open Alpha - Base Age: the live entry half.

WHAT THIS REPLACES AND WHY. The legacy scanner (`services/oa_entry.py`, r/142 spec) looked
for names sitting BELOW their all-time high and rested a buy-stop at the pivot, hoping to be
filled on the touch. research/158 killed that on the source site's own 54 published trades:
the published entry is not placeable at all, and every placeable version of it loses
(touch-of-the-pivot: -1.4% CAGR, -81.8% drawdown, against NIFTYBEES' +11.5%). It also found
the scanner's own condition INVERTED - `close < pivot` where the design says
`close > pivot` - so from 08-Sep it was arming names that had not broken out.

The Base Age entry is a different thing and a simpler one. The signal is decided on the
official close: the close is the first above the prior all-time-high close, that prior high
is >= 60 trading bars old, and the stock fell >= 20% below it in between. The FILL IS THE
NEXT DAY'S OPEN - no trigger, no touch, no intraday luck. So there is nothing to re-arm at
09:25 and that cron is retired: an after-market order either participates in tomorrow's
opening trade or it does not, and replacing it mid-session would be a different entry from
the one the study measured.

CONTESTED SLOTS GO TO THE MOST LIQUID NAME. research/161 drew a seeded random number when
more signals fired than the book had slots for. A live book cannot do that. research/164
tested the four obvious ranked rules and only one was consistent: take the largest 20-day
traded value. It beat the random draw at both slot counts tested (+0.78pp at 16 slots,
+2.60pp at 8), on 29/30 and 30/30 paired seeds, in both windows, and it improves capacity
rather than costing it. Declared as deviation D1 in the STATUS doc.

CASH, NOT SLOTS, IS THE BINDING CONSTRAINT. research/164: of 3,619 qualifying events the
16-slot book took 688, refused 977 for want of a slot and 1,955 for want of cash. So a
refusal for cash is a first-class, logged outcome here, not a silent `continue`.

Usage:
    python3 services/oa_baseage_entry.py                 scan and print; place nothing
    python3 services/oa_baseage_entry.py --arm           place the AMO buys for tomorrow
    python3 services/oa_baseage_entry.py --asof 2026-09-11   replay a past evening, no orders
"""
import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import oa_baseage as spec                                  # noqa: E402

STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
ENTRY_TAG = 'OA-ENTRY'
BUY_BAND = 0.02          # AMO LIMIT fallback: last close + 2%


def _variety():
    """'regular' inside market hours, 'amo' outside.

    Kite refuses an AMO between 09:15 and 15:30 ('AMOs can only be placed after trading
    hours') and refuses a regular order outside them. The evening job always wants amo;
    reading the clock is the whole difference.
    """
    n = datetime.now()
    mins = n.hour * 60 + n.minute
    intraday = n.weekday() < 5 and (9 * 60 + 15) <= mins <= (15 * 60 + 30)
    return 'regular' if intraday else 'amo'


def candidates(asof=None, progress=None):
    """Qualifying Base Age signals whose TRIGGER is `asof` (default: the last session)."""
    if asof is None:
        asof = spec.last_session()
    sigs = spec.scan(asof=asof, allow_last_bar=True, progress=progress)
    today = [s for s in sigs if s['trigger_date'] == asof]
    # research/164's tie-break: the most liquid name wins a contested slot. Deterministic,
    # free, and it is also the ordering a human would defend.
    today.sort(key=lambda r: -r['tv20_cr'])
    return today, asof


def _resting_syms(kite):
    try:
        return {o['tradingsymbol'] for o in kite.orders()
                if o.get('status') not in ('REJECTED', 'CANCELLED', 'COMPLETE')}
    except Exception as e:
        print('  order read failed, refusing to place blind:', e)
        return None


def _leaving(kite, held):
    """Held names with a SELL already resting - their slots are going, so they free up."""
    try:
        return {o['tradingsymbol'] for o in kite.orders()
                if o.get('tradingsymbol') in held
                and o.get('transaction_type') == 'SELL'
                and o.get('status') not in ('REJECTED', 'CANCELLED')}
    except Exception as e:
        print('  order read failed, assuming nothing is leaving:', e)
        return set()


def _pending_entries(kite):
    try:
        return {o['tradingsymbol'] for o in kite.orders()
                if o.get('tag') == ENTRY_TAG
                and o.get('status') not in ('REJECTED', 'CANCELLED')}
    except Exception as e:
        print('  order read failed, assuming no pending entries:', e)
        return set()


def plan(st, cand, kite=None, nav=None):
    """Turn the ranked candidates into an order plan. Pure - places nothing.

    Returns (orders, refusals, ctx). `nav` lets the caller pass a marked NAV; without one
    the book's own cost-plus-cash figure is used, which is the conservative reading on a
    day the page has not been marked.
    """
    held = {p['symbol'] for p in st.get('positions', [])}
    leaving = _leaving(kite, held) if kite is not None else set()
    armed = _pending_entries(kite) if kite is not None else set()
    cash = float(st.get('cash', 0.0))
    if nav is None:
        nav = cash + sum(p['qty'] * p['buy'] for p in st.get('positions', []))
    slot_rs = spec.SLOT_PCT * nav
    free = max(0, spec.SLOTS - (len(held) - len(leaving)) - len(armed - held))

    orders, refusals = [], []
    remaining = cash
    for r in cand:
        s = r['symbol']
        if s in held and s not in leaving:
            refusals.append((s, 'already held'))
            continue
        if s in armed:
            refusals.append((s, 'an entry order is already armed'))
            continue
        if len(orders) >= free:
            refusals.append((s, 'no free slot'))
            continue
        px = r['trigger_close']
        qty = int(slot_rs // px)
        if qty < 1:
            refusals.append((s, 'slot Rs %.0f buys no whole share at %.2f' % (slot_rs, px)))
            continue
        cost = qty * px
        if cost > remaining:
            # research/164's first-order constraint. Say it out loud every time.
            refusals.append((s, 'refused for cash: needs Rs %s, book has Rs %s'
                             % (format(round(cost), ','), format(round(remaining), ','))))
            continue
        remaining -= cost
        orders.append(dict(symbol=s, qty=qty, ref_close=px, est_cost=round(cost),
                           tv20_cr=r['tv20_cr'], x_bars=r['x_bars'],
                           depth_pct=r['depth_pct'], prev_ath=r['prev_ath'],
                           trigger_date=r['trigger_date']))
    ctx = dict(nav=round(nav), cash=round(cash), slot_rs=round(slot_rs), free=free,
               held=len(held), leaving=sorted(leaving), armed=sorted(armed),
               cash_left=round(remaining))
    return orders, refusals, ctx


def place_buy(kite, sym, qty, ref_close, tick=0.05, dry=True):
    """One AMO buy for tomorrow's open. Returns (order_id, order_type_used, error).

    MARKET FIRST, LIMIT AS THE FALLBACK. The study fills at the next open with no ceiling,
    and only a market order does that. Zerodha's RMS has historically refused after-market
    MARKET orders on some segments, and `services/oa_real.place_exit` already carries the
    scar of a bare market order being refused via the API - so the refusal is expected,
    caught, and answered with a LIMIT 2% above the last close rather than being allowed to
    skip the entry. Which one went in is logged on the order, because the two are not the
    same trade and nobody should have to guess afterwards.

    NOT TEST-FIRED. Whether this account's RMS takes an AMO MARKET on CNC equity is
    determined by the first real evening run, and either branch is safe: MARKET fills at the
    open (what the study models), LIMIT fills at the open whenever the open is at or below
    close+2% (which it is on all but a violent gap up - and a violent gap up is exactly the
    entry worth skipping).
    """
    if dry:
        limit = round(math.floor((ref_close * (1 + BUY_BAND)) / tick) * tick, 2)
        return None, 'MARKET (fallback LIMIT %.2f)' % limit, None
    err = None
    try:
        oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=sym,
                               transaction_type='BUY', quantity=int(qty), product='CNC',
                               order_type='MARKET', validity='DAY', tag=ENTRY_TAG)
        return oid, 'MARKET', None
    except Exception as e:
        err = str(e)
        print('       AMO MARKET refused (%s); falling back to LIMIT' % err[:90])
    limit = round(math.floor((ref_close * (1 + BUY_BAND)) / tick) * tick, 2)
    try:
        oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=sym,
                               transaction_type='BUY', quantity=int(qty), product='CNC',
                               order_type='LIMIT', price=limit, validity='DAY', tag=ENTRY_TAG)
        return oid, 'LIMIT %.2f' % limit, err
    except Exception as e2:
        return None, None, '%s | LIMIT also refused: %s' % (err, e2)


def run(arm=False, asof=None, progress=None):
    from services.oa_real import OA_RULESET, load_state, _alert, _kite

    if OA_RULESET != 'baseage':
        print('OA_RULESET is %r - the Base Age scanner is not the active ruleset; '
              'nothing scanned, nothing placed.' % OA_RULESET)
        return
    cand, asof = candidates(asof=asof, progress=progress)
    print('Base Age scan as of %s: %d qualifying signal(s) '
          '(ATH close, prior high >= %d bars old, depth >= %.0f%%, TV >= Rs %.0f cr)'
          % (asof, len(cand), spec.X_MIN, spec.DEPTH_MIN, spec.LIQ_MIN_CR))
    for r in cand:
        print('   %-14s close %9.2f  prev ATH %9.2f (%s, %d bars)  depth %5.1f%%  TV %6.2f cr'
              % (r['symbol'], r['trigger_close'], r['prev_ath'], r['prev_ath_date'],
                 r['x_bars'], r['depth_pct'], r['tv20_cr']))

    st = load_state()
    kite = None
    if arm:
        kite = _kite()
    orders, refusals, ctx = plan(st, cand, kite)
    print('\nbook: %d/%d held%s, %d slot(s) free, NAV Rs %s, cash Rs %s, slot Rs %s'
          % (ctx['held'], spec.SLOTS,
             (', leaving: ' + ', '.join(ctx['leaving'])) if ctx['leaving'] else '',
             ctx['free'], format(ctx['nav'], ','), format(ctx['cash'], ','),
             format(ctx['slot_rs'], ',')))
    for s, why in refusals:
        print('   skip %-14s %s' % (s, why))
    if not orders:
        print('nothing to arm')
        return
    placed, failed = [], []
    for o in orders:
        print('  ARM  %-14s BUY %d for tomorrow\'s open (ref close %.2f, Rs %s)'
              % (o['symbol'], o['qty'], o['ref_close'], format(o['est_cost'], ',')))
        if not arm:
            continue
        oid, kind, err = place_buy(kite, o['symbol'], o['qty'], o['ref_close'], dry=False)
        if oid:
            print('       placed %s as %s' % (oid, kind))
            placed.append('%s x%d (%s)' % (o['symbol'], o['qty'], kind))
        else:
            print('       FAILED: %s' % err)
            failed.append('%s x%d: %s' % (o['symbol'], o['qty'], str(err)[:110]))
    if not arm:
        print('\nDRY RUN - nothing placed. Re-run with --arm.')
        return
    if placed:
        _alert('Open Alpha Base Age: %d entry order(s) armed for the open' % len(placed),
               '; '.join(placed), 'low')
    if failed:
        _alert('Open Alpha Base Age: %d entry order(s) REJECTED' % len(failed),
               '\n  '.join(failed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='store_true', help='actually place the AMO buys')
    ap.add_argument('--asof', help='replay a past evening (never places)')
    ap.add_argument('--progress', type=int, default=0)
    a = ap.parse_args()
    if a.asof and a.arm:
        print('--asof is a replay; it never arms')
        return
    run(arm=a.arm, asof=a.asof, progress=a.progress or None)


if __name__ == '__main__':
    main()
