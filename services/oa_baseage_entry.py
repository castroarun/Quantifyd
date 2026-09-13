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

AND THAT IS WHY OA-ROT-1 EXISTS. If refusals are the norm rather than the exception, the
question "which refused signal was the book wrong to turn away, and what should have left to
make room" is worth asking every evening. research/170 Part B answered it: sell the holding
more than 10% under water, buy the refused signal with the highest 12-month relative
strength, one swap a night. 22.58% / -31.78% / Calmar 0.710 against 20.95 / -34.05 / 0.611,
+0.105 paired Calmar on 30/30 fresh seeds. See `rot1_pick` below and
research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md.

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
RS_LOOKBACK = 252        # research/170 `build_aux`: rs252 = close[t] / close[t-252] - 1
# The study's master trading calendar, from research/161 `bt161_sweep.py`. See
# `master_calendar` below for why the anchor symbol is load-bearing rather than incidental.
CAL_ANCHOR, CAL_SINCE = 'NIFTYBEES', '2005-01-03'


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


def plan(st, cand, kite=None, nav=None, marks=None, asof=None):
    """Turn the ranked candidates into an order plan. Places nothing.

    Returns (orders, refusals, ctx). `nav` lets the caller pass a NAV outright; `marks` lets
    it pass the closes it has already read. Given neither, the book is MARKED TO THE LAST
    OFFICIAL CLOSE.

    WHY THE DEFAULT CHANGED (13-Sep-2026). This used to fall back to the book's
    cost-plus-cash figure, described as the conservative reading. It is not conservative, it
    is a different book: the study, research/170's engine and `rot1_pick` below all size a
    slot at 6.25% of the MARKED net asset value, so a cost-based default made the live
    entries drift away from the tested size as the book gained or lost - and drift by a
    different amount than the swap leg of the same evening, which is worse than either basis
    on its own. On the 11-Sep-2026 book it was the difference between PAYTM x20 (cost NAV
    Rs 6,07,923) and PAYTM x21 (marked NAV Rs 6,21,871). A position whose latest bar is stale
    falls back to its buy price and is named in `ctx['unmarked']`.
    """
    held = {p['symbol'] for p in st.get('positions', [])}
    leaving = _leaving(kite, held) if kite is not None else set()
    armed = _pending_entries(kite) if kite is not None else set()
    cash = float(st.get('cash', 0.0))
    unmarked = []
    if nav is None:
        if marks is None:
            asof = asof or (cand[0]['trigger_date'] if cand else spec.last_session())
            marks, stale = official_closes(sorted(held), asof)
            unmarked = [s for s, _ in stale]
        nav = cash + sum(p['qty'] * marks.get(p['symbol'], p['buy'])
                         for p in st.get('positions', []))
    slot_rs = spec.SLOT_PCT * nav
    free = max(0, spec.SLOTS - (len(held) - len(leaving)) - len(armed - held))

    orders, refusals, turned_away = [], [], []
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
            turned_away.append(r)
            continue
        px = r['trigger_close']
        qty = int(slot_rs // px)
        if qty < 1:
            refusals.append((s, 'slot Rs %.0f buys no whole share at %.2f' % (slot_rs, px)))
            turned_away.append(r)
            continue
        cost = qty * px
        if cost > remaining:
            # research/164's first-order constraint. Say it out loud every time.
            refusals.append((s, 'refused for cash: needs Rs %s, book has Rs %s'
                             % (format(round(cost), ','), format(round(remaining), ','))))
            turned_away.append(r)
            continue
        remaining -= cost
        orders.append(dict(symbol=s, qty=qty, ref_close=px, est_cost=round(cost),
                           tv20_cr=r['tv20_cr'], x_bars=r['x_bars'],
                           depth_pct=r['depth_pct'], prev_ath=r['prev_ath'],
                           trigger_date=r['trigger_date']))
    ctx = dict(nav=round(nav), cash=round(cash), slot_rs=round(slot_rs), free=free,
               held=len(held), leaving=sorted(leaving), armed=sorted(armed),
               cash_left=round(remaining),
               # OA-ROT-1's inputs. `turned_away` is the study's `unfilled`: a QUALIFYING
               # signal the book could not take for want of a slot or of cash - never one
               # it already holds or has already armed, which the engine skips too.
               turned_away=turned_away, cash_after_entries=remaining,
               armed_cost=float(sum(o['est_cost'] for o in orders)),
               unmarked=unmarked)
    return orders, refusals, ctx


def place_buy(kite, sym, qty, ref_close, tick=0.05, dry=True, tag=ENTRY_TAG):
    """One AMO buy for tomorrow's open. Returns (order_id, order_type_used, error).

    `tag` is ENTRY_TAG for every ordinary entry. The OA-ROT-1 swap passes ROT1_BUY_TAG so
    that the fill reconciles as the entrant half of a swap rather than as a fresh entry.

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
                               order_type='MARKET', validity='DAY', tag=tag)
        return oid, 'MARKET', None
    except Exception as e:
        err = str(e)
        print('       AMO MARKET refused (%s); falling back to LIMIT' % err[:90])
    limit = round(math.floor((ref_close * (1 + BUY_BAND)) / tick) * tick, 2)
    try:
        oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=sym,
                               transaction_type='BUY', quantity=int(qty), product='CNC',
                               order_type='LIMIT', price=limit, validity='DAY', tag=tag)
        return oid, 'LIMIT %.2f' % limit, err
    except Exception as e2:
        return None, None, '%s | LIMIT also refused: %s' % (err, e2)


# ════════════════════════ OA-ROT-1: the best-entrant swap ════════════════════════
# research/170 Part B, adopted by Arun 13-Sep-2026. The rule in one sentence: on an evening
# when a qualifying signal is refused, sell the holding that is more than 10% below its buy
# price and buy the refused signal with the highest 12-month relative strength, both at the
# next open, at most once a night.
#
# WHY THE TWO HALVES OF THE RULE ARE NOT THE SAME RULE. research/170's cleanest single fact:
# all three entrant priorities it tested (relative strength, traded value, base age) earn
# the SAME +1.6 to +1.8pp of CAGR, and the entire spread between them - 5.5 points of
# drawdown, 0.105 of Calmar - is in which refused breakout gets bought. Who leaves sets the
# return; who enters sets the drawdown. The oldest-base entrant actually LOSES to never
# swapping on Calmar while earning more, which is leverage, not selection. So the -10%
# margin and the rs252 ranking are two separate decisions and both are load-bearing.
#
# WHY 10% AND NOT 7.5% OR 12.5%. All three beat never-swapping, so the plateau clause passes,
# but only 10% wins BOTH pre-registered windows: 7.5% earns its edge in 2005-2015 and loses
# in 2016-2026, and 12.5% does the reverse. The band is thin. This is a constant, not a knob.


def _raw_closes(con, sym, asof=None, limit=None):
    """A symbol's RAW daily closes - dates ascending, no split cut. Returns a numpy array.

    Deliberately NOT `spec.load_bars`. That function truncates the series after the last
    day-over-day fall worse than -35%, which is right for an all-time-high test (a pre-split
    row fakes a high that can never be reached again) and WRONG for a 12-month return: the
    study's rs252 is read off research/164's panel, which is the raw close forward-filled
    onto a master calendar with no split cut at all. Using the cut series here would send
    every recently-split name to the bottom of the entrant ranking on a data artefact.
    """
    import numpy as np
    q = ("SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
         "AND volume>0 AND close>0")
    p = [sym]
    if asof:
        q += " AND substr(date,1,10)<=?"
        p.append(asof)
    q += " ORDER BY date"
    rows, seen = [], {}
    for d, c in con.execute(q, p):
        seen[str(d)] = float(c)          # duplicates: keep the LAST, as the study does
        rows.append(str(d))
    out = [seen[d] for d in sorted(set(rows))]
    if limit:
        out = out[-limit:]
    return np.asarray(out, dtype=float)


_CAL_CACHE = {}


def master_calendar(con, since=CAL_SINCE, anchor=CAL_ANCHOR):
    """The study's own trading calendar: every daily bar `anchor` has, from `since`.

    NOT a cosmetic choice. research/161 built its price panel on exactly this list
    (`bt161_sweep.py`: distinct dates where symbol = NIFTYBEES, from 2005-01-03), research/164
    inherited it, and research/170 computes rs252 on it - so "252 bars ago" in the published
    result means 252 sessions of THIS calendar, not 252 of the symbol's own bars.

    MEASURED, because the difference is not small (`scripts/rot1_rs_diag.py`, over the 3,366
    frozen research/164 events that carry an rs252): counting a symbol's OWN bars reproduces
    the study's number on 73.7% of them and picks a different top-ranked name on 16 of 810
    contested days; counting THIS calendar reproduces it on 99.7% and picks the same name on
    810 of 810. A name that missed six sessions in a year is otherwise reaching a year and
    six days back for its reference price, which is a different number and sometimes a
    different winner.
    """
    key = (id(con), since, anchor)
    if key not in _CAL_CACHE:
        _CAL_CACHE[key] = [d for d in sorted({str(r[0])[:10] for r in con.execute(
            "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
            "AND symbol=?", (anchor,))}) if d >= since]
    return _CAL_CACHE[key]


def rs252(con, sym, asof=None, lookback=RS_LOOKBACK):
    """12-month relative strength in %, read on the trigger close. None if too short.

    `100 * (close[t] / close[t-252] - 1)` - research/170 `sim170.build_aux`, on the study's
    forward-filled panel: the reference bar is 252 sessions of the MASTER CALENDAR ago, and
    the price used is the symbol's last close at or before that session. A name that had not
    listed by then, or that has no close on or before it, has no figure and RANKS LAST -
    which is exactly what the study's NaN (and research/164's `-1e9` sentinel) does.
    """
    import bisect
    cal = master_calendar(con)
    if asof:
        cal = cal[:bisect.bisect_right(cal, asof)]
    if len(cal) <= lookback:
        return None
    today, ref = cal[-1], cal[-(lookback + 1)]
    inside = set(cal)
    rows = [(d, c) for (d, c) in (
        (str(d)[:10], float(c)) for d, c in con.execute(
            "SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
            "AND volume>0 AND close>0 ORDER BY date", (sym,)))
        if d in inside and d <= today]
    if not rows:
        return None
    seen = {}
    for d, c in rows:
        seen[d] = c                       # duplicates: keep the LAST, as the study does
    dates = sorted(seen)
    a = [d for d in dates if d <= ref]
    if not a or dates[-1] > today:
        return None
    p0, p1 = seen[a[-1]], seen[dates[-1]]
    if p0 <= 0:
        return None
    return 100.0 * (p1 / p0 - 1.0)


def official_closes(symbols, asof):
    """{symbol: close} for the names whose LATEST daily bar IS `asof`.

    A name whose last bar is older did not trade, or the 17:45 universe refresh missed it.
    Either way the rule has not been read on today's close for it, so it is left out and the
    caller reports it - the same guard `oa_real.confirm()` applies before selling anything.
    """
    con = spec.connect()
    out, stale = {}, []
    try:
        for s in symbols:
            d, _ = spec.load_bars(con, s, asof)
            if d is None:
                stale.append((s, 'no usable price history'))
                continue
            last = str(d['date'].iloc[-1])[:10]
            if last != asof:
                stale.append((s, 'last DB bar %s, latest session %s' % (last, asof)))
                continue
            out[s] = float(d['close'].iloc[-1])
    finally:
        con.close()
    return out, stale


def rot1_pick(positions, marks, turned_away, cash, armed_cost=0.0, armed=(), leaving=(),
              margin=None, slot_pct=None, cost_pct=0.0025):
    """THE RULE. Pure: reads dicts, returns a decision. Places nothing, reads no database.

    Kept pure on purpose - the replication gate re-decides every rotation decision
    research/170's own engine made by calling THIS function with the engine's own inputs,
    which is only possible if it has no I/O in it.

    positions   : [{symbol, qty, buy}, ...] - the book as it stands tonight
    marks       : {symbol: official close on the trigger day} - a missing name is NOT eligible
    turned_away : the qualifying signals the book refused, each with symbol, trigger_close
                  and rs252 (None -> ranks last)
    cash        : cash left AFTER the ordinary entries were armed
    armed_cost  : rupees committed to those entries (they are book value for the NAV)
    armed       : symbols with an entry order already armed tonight
    leaving     : symbols with a SELL already resting - an ordinary exit, never a swap-out

    Returns (plan | None, why). `plan` carries everything the log line and the two orders
    need, and nothing is decided anywhere else.
    """
    from services.oa_real import ROT1_MARGIN_PCT
    margin = ROT1_MARGIN_PCT if margin is None else float(margin)
    slot_pct = spec.SLOT_PCT if slot_pct is None else float(slot_pct)
    leaving, armed = set(leaving), set(armed)
    held = {p['symbol'] for p in positions}

    # ── 1. who could leave. A position with a SELL already resting is on its way out on the
    #       trail; taking credit for its slot twice would be a double sale.
    elig = []
    for p in positions:
        s = p['symbol']
        if s in leaving or s not in marks or float(p.get('buy', 0)) <= 0:
            continue
        elig.append((100.0 * (marks[s] / float(p['buy']) - 1.0), s, p))
    if not elig:
        return None, 'no holding is eligible to be swapped out'
    elig.sort(key=lambda t: (t[0], t[1]))        # weakest first; ties by symbol, as the study
    weak_ret, weak_sym, weak_pos = elig[0]

    # ── 2. the margin. The entrant's own unrealised return is 0 by construction (it has not
    #       been bought yet), so the engine's `(ent - hold) >= margin` is exactly this.
    if -weak_ret < margin:
        return None, ('deepest loss is %s %+.2f%%, not worse than -%.1f%% - no swap'
                      % (weak_sym, weak_ret, margin))

    # ── 3. who enters: highest rs252 first, no-figure last, ties by symbol.
    ents = [r for r in turned_away
            if r['symbol'] not in held and r['symbol'] not in armed]
    if not ents:
        return None, 'every refused signal is already held or already armed'
    ents.sort(key=lambda r: (-r['rs252'] if r.get('rs252') is not None else 1e18,
                             r['symbol']))

    # ── 4. can the entrant be funded out of the sale? The engine walks down the ranking when
    #       the top name cannot be bought with one whole share, so this does too; only when
    #       NONE of them can are both legs cancelled.
    proceeds = float(weak_pos['qty']) * marks[weak_sym] * (1.0 - cost_pct)
    mv_rest = sum(float(q['qty']) * marks.get(q['symbol'], float(q['buy']))
                  for q in positions if q['symbol'] != weak_sym)
    nav_after = cash + proceeds + mv_rest + armed_cost
    slot_rs = slot_pct * nav_after
    budget = min(slot_rs, cash + proceeds)
    for rank, e in enumerate(ents, 1):
        px = float(e['trigger_close'])
        qty = int(budget // px) if px > 0 else 0
        if qty < 1:
            continue
        return dict(out=weak_pos, out_symbol=weak_sym, out_ret_pct=round(weak_ret, 2),
                    out_close=marks[weak_sym], proceeds=round(proceeds, 2),
                    entrant=e, in_symbol=e['symbol'], in_rank=rank, n_entrants=len(ents),
                    in_rs252=e.get('rs252'), in_ref_close=px, qty=qty,
                    est_cost=round(qty * px, 2), nav_after=round(nav_after, 2),
                    slot_rs=round(slot_rs, 2), margin=margin), 'swap'
    return None, ('both legs cancelled: %s is %+.2f%% but Rs %s buys no whole share of any '
                  'of the %d refused signal(s)'
                  % (weak_sym, weak_ret, format(round(budget), ','), len(ents)))


def rot1_run(st, ctx, asof, kite=None, arm=False):
    """Apply OA-ROT-1 after the ordinary entries have been queued. Returns the plan or None.

    ORDER MATTERS AND IS THE RULE: the SELL goes first. Both legs are after-market orders
    for the same opening trade, so the exchange does not care which was sent first - but if
    the second one is refused, the book would rather be one position light with the cash in
    hand than one position heavy on margin it does not have.
    """
    from services.oa_real import (OA_ROT1, ROT1_SELL_TAG, ROT1_BUY_TAG, place_exit_amo,
                                  _alert)
    if not OA_ROT1:
        print('OA-ROT-1 is switched off (OA_ROT1 = False); exits and entries only.')
        return None
    turned = ctx.get('turned_away') or []
    if not turned:
        print('OA-ROT-1: no qualifying signal was refused tonight; nothing to swap.')
        return None
    positions = st.get('positions', [])
    if not positions:
        print('OA-ROT-1: the book holds nothing to swap out.')
        return None

    marks, stale = official_closes([p['symbol'] for p in positions], asof)
    for s, why in stale:
        print('   OA-ROT-1 skip %-14s %s (not eligible to be swapped out)' % (s, why))
    con = spec.connect()
    try:
        for r in turned:
            r['rs252'] = rs252(con, r['symbol'], asof)
    finally:
        con.close()

    print('OA-ROT-1: %d qualifying signal(s) refused; ranking them by 12-month relative '
          'strength' % len(turned))
    for r in sorted(turned, key=lambda x: (-x['rs252'] if x['rs252'] is not None else 1e18,
                                           x['symbol'])):
        print('   refused %-14s close %9.2f  rs252 %s  TV %6.2f cr'
              % (r['symbol'], r['trigger_close'],
                 ('%+7.1f%%' % r['rs252']) if r['rs252'] is not None else '   n/a ',
                 r['tv20_cr']))

    plan_, why = rot1_pick(positions, marks, turned,
                           cash=float(ctx.get('cash_after_entries', 0.0)),
                           armed_cost=float(ctx.get('armed_cost', 0.0)),
                           armed=set(ctx.get('armed') or ()),
                           leaving=set(ctx.get('leaving') or ()))
    if plan_ is None:
        print('OA-ROT-1: %s' % why)
        return None

    head = ('swap: sold %s (%+.1f%%) for %s (rs252 rank %d of %d)'
            % (plan_['out_symbol'], plan_['out_ret_pct'], plan_['in_symbol'],
               plan_['in_rank'], plan_['n_entrants']))
    detail = ('SELL %s x%d at tomorrow\'s open (close %.2f, buy %.2f) and BUY %s x%d '
              '(ref close %.2f, Rs %s, rs252 %s). Slot 6.25%% of NAV Rs %s = Rs %s.'
              % (plan_['out_symbol'], plan_['out']['qty'], plan_['out_close'],
                 plan_['out']['buy'], plan_['in_symbol'], plan_['qty'],
                 plan_['in_ref_close'], format(round(plan_['est_cost']), ','),
                 ('%+.1f%%' % plan_['in_rs252']) if plan_['in_rs252'] is not None else 'n/a',
                 format(round(plan_['nav_after']), ','), format(round(plan_['slot_rs']), ',')))
    print('OA-ROT-1 %s' % head)
    print('   %s' % detail)
    if not arm:
        print('   DRY RUN - neither leg placed.')
        return plan_

    oid_s, kind_s, err_s = place_exit_amo(kite, plan_['out_symbol'], plan_['out']['qty'],
                                          plan_['out_close'], tag=ROT1_SELL_TAG)
    if not oid_s:
        _alert('Open Alpha OA-ROT-1: the SELL leg did NOT go',
               head + ' The sell was refused (%s), so the BUY was NOT sent either - the '
                      'book is unchanged. %s' % (err_s, detail))
        print('   SELL leg FAILED: %s - buy not sent, swap abandoned' % err_s)
        return None
    print('   SELL placed %s as %s' % (oid_s, kind_s))
    oid_b, kind_b, err_b = place_buy(kite, plan_['in_symbol'], plan_['qty'],
                                     plan_['in_ref_close'], dry=False, tag=ROT1_BUY_TAG)
    if oid_b:
        print('   BUY  placed %s as %s' % (oid_b, kind_b))
        _alert('Open Alpha OA-ROT-1 swap armed for the open',
               head + ' ' + detail + ' SELL %s as %s, BUY %s as %s.'
               % (oid_s, kind_s, oid_b, kind_b), 'low')
    else:
        # The sale is already resting. Say so loudly: the book will be one slot light and
        # the cash goes back into tomorrow evening's ordinary entry queue, which is a safe
        # state but not the state the rule asked for.
        _alert('Open Alpha OA-ROT-1: the SELL is in but the BUY was REJECTED',
               head + ' The sell order %s is resting for the open; the buy was refused '
                      '(%s). Cancel the sell by hand before 09:15 if the swap is not '
                      'wanted, or place the buy by hand. %s' % (oid_s, err_b, detail))
        print('   BUY leg FAILED: %s' % err_b)
    plan_['sell_order'] = oid_s
    plan_['buy_order'] = oid_b
    return plan_


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
    for s in ctx.get('unmarked') or ():
        print('   NOT MARKED %-14s no bar for %s; NAV counts it at its buy price' % (s, asof))
    for s, why in refusals:
        print('   skip %-14s %s' % (s, why))
    placed, failed = [], []
    if not orders:
        print('nothing to arm')
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
    if arm and placed:
        _alert('Open Alpha Base Age: %d entry order(s) armed for the open' % len(placed),
               '; '.join(placed), 'low')
    if arm and failed:
        _alert('Open Alpha Base Age: %d entry order(s) REJECTED' % len(failed),
               '\n  '.join(failed))

    # ── OA-ROT-1, last and only if a qualifying signal is still refused ──────────────
    # It runs AFTER the ordinary entries for the same reason the exits run before them: a
    # signal the book could take normally is not a swap, and a slot freed by tonight's trail
    # exit belongs to the entry queue first. Every name armed above is off the entrant list.
    print('')
    ctx['armed'] = sorted(set(ctx.get('armed') or ()) | {o['symbol'] for o in orders})
    rot1_run(st, ctx, asof, kite=kite, arm=arm)

    if not arm:
        print('\nDRY RUN - nothing placed. Re-run with --arm.')


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
