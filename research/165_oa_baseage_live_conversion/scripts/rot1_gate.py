# -*- coding: utf-8 -*-
"""research/165 - the OA-ROT-1 replication gate. Read-only: no Kite, no state, no orders.

TWO GATES, AND THEY ASK DIFFERENT QUESTIONS.

  R1  DECISION AGREEMENT.  research/170's own engine is regenerated with a decision probe
      (`patch_probe170.py` -> `sim170_probe.py`) and run over the full 2005-2026 history. At
      every rotation decision point it records what it saw and what it did. The LIVE
      `services.oa_baseage_entry.rot1_pick()` is then handed the same inputs and asked to
      decide again. The gate is the set of swap days and swap PAIRS: fire / no-fire, which
      name leaves, which name enters. Anything less than 100% has to be named.

      This is deliberately not "re-implement the rule and compare the two implementations".
      There is only one implementation of the rule in this repo that can place an order, and
      it is the one being interrogated.

  R2  THE LIVE CODE, WALKED.  The staged evening job replayed day by day over the last 400
      sessions on research/164's frozen event list - SuperTrend exits, the traded-value
      tie-break, the cash check, then OA-ROT-1 - from TWO starting books: the live book's
      actual eleven positions, and an empty book. Then the same walk over the last 60
      sessions at the live book's capital, which is the comparison research/165 §8.2 made
      for the un-rotated spec.

Usage:
    python3 research/165_oa_baseage_live_conversion/scripts/rot1_gate.py
    python3 ... rot1_gate.py --seeds=3          # fewer engine paths (default 5)
    python3 ... rot1_gate.py --skip-r1          # only the live-code walk
"""
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / 'results'
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research/170_qs_leeway_and_baseage_best_entrant/scripts'))
sys.path.insert(0, str(HERE))

from services import oa_baseage as spec                                    # noqa: E402
from services import oa_baseage_entry as live                              # noqa: E402

PANEL_PKL = ROOT / 'research/164_baseage_slots_sizing/results/panel164.pkl'
ST_PKL = ROOT / 'research/166_baseage_rotation_and_drift/results/st166.pkl'
EV166 = ROOT / 'research/166_baseage_rotation_and_drift/results/events166.csv'
EV164 = ROOT / 'research/164_baseage_slots_sizing/results/events164.csv'

LOOKBACK = 400
WALK60 = 60
BOOK_CAPITAL = 617637.68
LIVE_CASH = 188697.86
LIVE_POSITIONS = [
    dict(symbol='INDSWFTLAB', qty=99, buy=362.33), dict(symbol='SETL', qty=99, buy=402.69),
    dict(symbol='WELCORP', qty=14, buy=2596.36), dict(symbol='SHILPAMED', qty=40, buy=962.36),
    dict(symbol='SBCL', qty=34, buy=1115.48), dict(symbol='IRISDOREME', qty=634, buy=62.39),
    dict(symbol='INOXINDIA', qty=17, buy=2236.50), dict(symbol='MANINDS', qty=47, buy=800.19),
    dict(symbol='SSWL', qty=106, buy=358.30), dict(symbol='ENTERO', qty=21, buy=1843.72),
    dict(symbol='NITINSPIN', qty=61, buy=636.65),
]

# The adopted cell, research/170 Part B `X_entrs_unre_m010`.
CELL = dict(exit='ST_14_4', hard_stop=False, hard_stop_pct=0.92, rot_sell_only=False,
            time_stop=0, cost_bps=25.0, gate_ok=None, idle_yield=0.052, slots=16,
            slot_pct=0.0625, select='random', rot_score='unreal', rot_margin=10.0,
            rot_max_per_day=1, rot_entrant='rs', trim_mult=0.0, trim_when='month',
            min_fill_frac=0.0)


# ══════════════════════════ R1 - decision agreement ══════════════════════════
def r1(seeds):
    import sim170_probe as S

    print('\n' + '=' * 78)
    print('R1  DECISION AGREEMENT - the live rot1_pick() vs research/170\'s own engine')
    print('=' * 78, flush=True)
    t0 = time.time()
    panel = pickle.load(open(PANEL_PKL, 'rb'))
    stl = pickle.load(open(ST_PKL, 'rb'))
    aux = S.build_aux(panel, stl)
    ev = pd.read_csv(EV166)
    events = ev.to_dict('records')
    for e in events:
        e['entry_i'] = int(e['entry_i'])
    print('panel %d syms / %d days (%s .. %s); %d events; loaded in %.0fs'
          % (len(panel.close), panel.n, panel.cal[0], panel.cal[-1], len(events),
             time.time() - t0), flush=True)

    runs = [('random', sd) for sd in seeds] + [('tv', 1001)]
    recs, rows = [], []
    for sel, sd in runs:
        cfg = dict(CELL, select=sel)
        S.PROBE = []
        t1 = time.time()
        nav, tr, inv, book = S.simulate(events, panel, aux, cfg, sd)
        m = S.metrics(nav, panel.cal, tr)
        print('  engine run select=%-6s seed=%-5d -> %d decision points, %d swaps, '
              'CAGR %.2f%% DD %.2f%% Calmar %.3f  (%.0fs)'
              % (sel, sd, len(S.PROBE), book['swaps'], m['cagr'], m['maxdd'], m['calmar'],
                 time.time() - t1), flush=True)
        for r in S.PROBE:
            r['run'] = '%s/%d' % (sel, sd)
        recs.extend(S.PROBE)
        S.PROBE = None

    # the live function, on the engine's own inputs
    agree = dict(total=0, fire=0, out=0, pair=0)
    buckets = {}
    for r in recs:
        pos = [dict(symbol=h['symbol'], qty=h['shares'], buy=h['entry_px'])
               for h in r['holdings'] if h['entry_i'] < r['i']]
        marks = {h['symbol']: h['close_j'] for h in r['holdings']
                 if h['entry_i'] < r['i'] and h['close_j']}
        turned = [dict(symbol=e['symbol'], trigger_close=e['open_i'], rs252=e['rs252'],
                       tv20_cr=e['tv20_cr']) for e in r['entrants'] if e['open_i']]
        plan_, why = live.rot1_pick(pos, marks, turned, cash=r['cash'],
                                    armed_cost=0.0, armed=(), leaving=())
        fired = plan_ is not None
        out_s = plan_['out_symbol'] if fired else None
        in_s = plan_['in_symbol'] if fired else None
        agree['total'] += 1
        ok_fire = (fired == r['fired'])
        ok_out = ok_fire and (out_s == r['out_symbol'])
        ok_pair = ok_out and (in_s == r['in_symbol'])
        agree['fire'] += ok_fire
        agree['out'] += ok_out
        agree['pair'] += ok_pair
        if not ok_pair:
            b = _bucket(r, plan_, why, fired, out_s, in_s)
            buckets[b] = buckets.get(b, 0) + 1
            if len(rows) < 4000:
                rows.append(dict(run=r['run'], day=r['day'],
                                 engine_fired=r['fired'], engine_out=r['out_symbol'],
                                 engine_in=r['in_symbol'], live_fired=fired,
                                 live_out=out_s, live_in=in_s, live_why=why, bucket=b))

    n = agree['total']
    print('\n  decision points        : %d  (%s .. %s)'
          % (n, min(r['day'] for r in recs), max(r['day'] for r in recs)))
    print('  engine swaps fired     : %d' % sum(1 for r in recs if r['fired']))
    for k, lab in (('fire', 'fire / no-fire agrees'), ('out', '+ the name SOLD agrees'),
                   ('pair', '+ the name BOUGHT agrees  <-- THE GATE')):
        print('  %-38s : %6d / %d = %.2f%%' % (lab, agree[k], n, 100.0 * agree[k] / n))
    if buckets:
        print('\n  mismatch buckets:')
        for b, c in sorted(buckets.items(), key=lambda x: -x[1]):
            print('    %5d  %s' % (c, b))
    else:
        print('\n  no mismatches of any kind.')

    # the last-400-session subset, stated separately because that is the brief's window
    days = sorted({r['day'] for r in recs})
    lo = days[-LOOKBACK] if len(days) > LOOKBACK else days[0]
    sub = [r for r in recs if r['day'] >= lo]
    sub_ok = sum(1 for r in sub
                 if not any(x['run'] == r['run'] and x['day'] == r['day'] for x in rows))
    print('\n  restricted to decision points on or after %s: %d points, %d exact (%.2f%%)'
          % (lo, len(sub), sub_ok, 100.0 * sub_ok / max(len(sub), 1)))

    with open(RES / 'rot1_decisions.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'day', 'engine_fired', 'engine_out',
                                          'engine_in', 'live_fired', 'live_out', 'live_in',
                                          'live_why', 'bucket'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    r3 = _r3_calendar(recs, panel.cal)
    return dict(points=n, engine_swaps=sum(1 for r in recs if r['fired']),
                fire_pct=round(100.0 * agree['fire'] / n, 4),
                out_pct=round(100.0 * agree['out'] / n, 4),
                pair_pct=round(100.0 * agree['pair'] / n, 4),
                window_points=len(sub), window_pct=round(100.0 * sub_ok / max(len(sub), 1), 4),
                window_start=lo, buckets=buckets, runs=['%s/%d' % r for r in runs], rs252=r3)


def _bucket(r, plan_, why, fired, out_s, in_s):
    if fired and not r['fired']:
        return 'live fires where the engine did not'
    if r['fired'] and not fired:
        if 'no whole share' in why:
            return 'engine funded the buy off the FILL-DAY open/NAV, live off the trigger close (R2)'
        return 'live declines where the engine fired: ' + why[:60]
    if out_s != r['out_symbol']:
        eng = [h for h in r['holdings'] if h['symbol'] == r['out_symbol']]
        if eng and eng[0]['entry_i'] >= r['i']:
            return 'engine sold a position bought at the same open (cannot exist live)'
        if eng and not eng[0]['close_j']:
            return 'engine scored a holding with no close on the signal bar'
        return 'different name sold'
    return 'same name sold, different entrant bought (R2 budget boundary)'


def _r3_calendar(recs, cal_panel):
    """R3 measured: does rs252 read off the live DB rank the entrants the same way?

    The engine reads rs252 off research/164's panel - the raw close forward-filled onto a
    master calendar - so "252 bars ago" is 252 CALENDAR sessions. The live scanner has no
    panel and counts 252 of the symbol's OWN bars. For a name clearing the Rs 2 cr liquidity
    floor those coincide except across a halt. This measures how often they disagree about
    which refused signal ranks first, which is the only thing the rule reads.
    """
    print('\n  R3 - rs252 from the live database vs the study panel (entrant ranking)')
    need = set()
    for r in recs:
        if len(r['entrants']) < 2:
            continue
        need |= {e['symbol'] for e in r['entrants']}
    con = spec.connect()
    try:
        cal, arrs = rs_index(con, sorted(need))
        probe = [(s, d) for s in sorted(need)[:25] for d in (cal[-1],)]
        bad = [(s, d) for s, d in probe
               if not _close(_rs(cal, arrs, s, d), live.rs252(con, s, d))]
    finally:
        con.close()
    print('    the vectorised reader agrees with services.oa_baseage_entry.rs252 on '
          '%d of %d spot checks' % (len(probe) - len(bad), len(probe)))

    def db_rs(sym, day):
        return _rs(cal, arrs, sym, day)

    n = same = 0
    for r in recs:
        ents = [e for e in r['entrants'] if e['open_i']]
        if len(ents) < 2:
            continue
        n += 1
        # The engine reads rs252 at bar j - the TRIGGER close, the evening before the fill.
        # `r['day']` is the FILL day, so the live reader must be asked for the day before it
        # or the two are being compared one session apart.
        day_j = cal_panel[r['j']]
        a = sorted(ents, key=lambda e: (-e['rs252'] if e['rs252'] is not None else 1e18,
                                        e['symbol']))[0]['symbol']
        scored = [dict(symbol=e['symbol'], rs=db_rs(e['symbol'], day_j)) for e in ents]
        b = sorted(scored, key=lambda e: (-e['rs'] if e['rs'] is not None else 1e18,
                                          e['symbol']))[0]['symbol']
        same += (a == b)
    print('    contested decision points (2+ entrants): %d' % n)
    print('    top-ranked entrant identical            : %d (%.2f%%)'
          % (same, 100.0 * same / max(n, 1)))
    return dict(contested=n, same_top=same, pct=round(100.0 * same / max(n, 1), 4))


def rs_index(con, symbols):
    """(calendar, {symbol: forward-filled close array}) - the vectorised twin of live.rs252.

    `services.oa_baseage_entry.rs252` re-reads one symbol from SQLite per call, which is the
    right shape for an evening job scoring a handful of names and far too slow for a 400-day
    walk. This is the same arithmetic on arrays: the SAME master calendar, the same
    forward-fill, the same "no close at or before the reference session -> no figure". The
    caller spot-checks it against the deployed function rather than trusting the twin.
    """
    cal = live.master_calendar(con)
    pos = {d: i for i, d in enumerate(cal)}
    arrs = {}
    for s in symbols:
        a = np.full(len(cal), np.nan)
        for d, c in con.execute(
                "SELECT date, close FROM market_data_unified WHERE symbol=? "
                "AND timeframe='day' AND volume>0 AND close>0 ORDER BY date", (s,)):
            k = pos.get(str(d)[:10])
            if k is not None:
                a[k] = float(c)
        arrs[s] = pd.Series(a).ffill().to_numpy()
    return cal, arrs


def _rs(cal, arrs, sym, day):
    a = arrs.get(sym)
    if a is None:
        return None
    k = int(np.searchsorted(cal, day, side='right')) - 1
    if k < live.RS_LOOKBACK:
        return None
    p0, p1 = a[k - live.RS_LOOKBACK], a[k]
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return None
    return 100.0 * (p1 / p0 - 1.0)


def _close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(b))


# ══════════════════════════ R2 - the live code, walked ══════════════════════════
def calendar():
    con = spec.connect()
    try:
        return [r[0][:10] for r in con.execute(
            "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
            "AND symbol='RELIANCE' ORDER BY date")]
    finally:
        con.close()


def load_series(symbols):
    """Per symbol: {date: (open, close)}, the ST(14,4) direction by date, the rs252 index."""
    con = spec.connect()
    px, stdir = {}, {}
    try:
        for s in symbols:
            d, _ = spec.load_bars(con, s)
            if d is None:
                continue
            dates = d['date'].astype(str).str[:10].to_numpy()
            o = d['open'].to_numpy(float)
            c = d['close'].to_numpy(float)
            h = d['high'].to_numpy(float)
            lo = d['low'].to_numpy(float)
            px[s] = dict(zip(dates, zip(o, c)))
            # the SAME SuperTrend(14,4) on the SAME split-cut series the live confirm() reads
            dirs = spec.supertrend_dir(h, lo, c)
            stdir[s] = dict(zip(dates, dirs.tolist()))
        raw = rs_index(con, symbols)
    finally:
        con.close()
    return px, stdir, raw


def walk(events, cal, px, stdir, raw, start_positions, start_cash, label, rot=True):
    """Replay the staged evening job. Calls the LIVE rot1_pick for every swap decision."""
    cal_rs, arrs = raw

    def rs_at(sym, day):
        return _rs(cal_rs, arrs, sym, day)

    by_day = {}
    for e in events:
        by_day.setdefault(e['trigger_date'], []).append(e)

    cash = float(start_cash)
    held = {p['symbol']: dict(p) for p in start_positions}
    pend_sell, pend_buy, log = [], [], []
    n_swap = n_ref_cash = n_ref_slot = n_exit = n_buy = 0
    for i, day in enumerate(cal):
        nxt = cal[i + 1] if i + 1 < len(cal) else None
        # ---- 1. yesterday's after-market orders take today's open
        for p, why in pend_sell:
            s = p['symbol']
            if s not in held or day not in px.get(s, {}):
                continue
            o = px[s][day][0]
            cash += p['qty'] * o * (1 - 0.0025)
            held.pop(s, None)
            n_exit += 1
            log.append(dict(book=label, day=day, action='SELL', symbol=s, qty=p['qty'],
                            price=round(o, 2),
                            ret_pct=round(100 * (o / p['buy'] - 1), 2), why=why))
        pend_sell = []
        for b in pend_buy:
            s = b['symbol']
            if s in held or day not in px.get(s, {}):
                continue
            o = px[s][day][0]
            cost = b['qty'] * o * (1 + 0.0025)
            cash -= cost
            held[s] = dict(symbol=s, qty=b['qty'], buy=round(o, 2), entry_date=day)
            n_buy += 1
            log.append(dict(book=label, day=day, action=b['action'], symbol=s, qty=b['qty'],
                            price=round(o, 2), ret_pct=0, why=b['why']))
        pend_buy = []
        if nxt is None:
            break

        marks = {s: px[s][day][1] for s in held if day in px.get(s, {})}
        # ---- 2. confirm(): SuperTrend(14,4) on today's close -> AMO SELL for the open
        leaving = set()
        for s, p in list(held.items()):
            if stdir.get(s, {}).get(day) == -1:
                pend_sell.append((p, 'ST(14,4) flip on %s close' % day))
                leaving.add(s)

        # ---- 3. the ordinary Base Age entries
        cand = sorted(by_day.get(day, []), key=lambda r: -r['tv20_cr'])
        orders, refusals, ctx = _replan(cand, held, leaving, cash, marks)
        for o in orders:
            pend_buy.append(dict(symbol=o['symbol'], qty=o['qty'], action='BUY',
                                 why='ATH close %s, base %d bars, depth %.0f%%, TV %.2f cr'
                                     % (o['trigger_date'], o['x_bars'], o['depth_pct'],
                                        o['tv20_cr'])))
        for s, why in refusals:
            if why.startswith('refused for cash'):
                n_ref_cash += 1
            elif why == 'no free slot':
                n_ref_slot += 1

        # ---- 4. OA-ROT-1
        if not rot or not ctx['turned_away']:
            continue
        for r in ctx['turned_away']:
            r['rs252'] = rs_at(r['symbol'], day)
        plan_, why = live.rot1_pick(list(held.values()), marks, ctx['turned_away'],
                                    cash=ctx['cash_after_entries'],
                                    armed_cost=ctx['armed_cost'],
                                    armed={o['symbol'] for o in orders}, leaving=leaving)
        if plan_ is None:
            continue
        n_swap += 1
        pend_sell.append((plan_['out'], 'OA-ROT-1 swap out (%+.1f%%) for %s'
                          % (plan_['out_ret_pct'], plan_['in_symbol'])))
        leaving.add(plan_['out_symbol'])
        pend_buy.append(dict(symbol=plan_['in_symbol'], qty=plan_['qty'], action='ROT1_BUY',
                             why='OA-ROT-1 entrant, rs252 %s, rank %d of %d, replacing %s'
                                 % (('%+.1f%%' % plan_['in_rs252'])
                                    if plan_['in_rs252'] is not None else 'n/a',
                                    plan_['in_rank'], plan_['n_entrants'],
                                    plan_['out_symbol'])))
        log.append(dict(book=label, day=day, action='SWAP', symbol=plan_['out_symbol'],
                        qty=plan_['out']['qty'], price=round(plan_['out_close'], 2),
                        ret_pct=plan_['out_ret_pct'],
                        why='swap: sold %s (%+.1f%%) for %s (rs252 rank %d of %d)'
                            % (plan_['out_symbol'], plan_['out_ret_pct'], plan_['in_symbol'],
                               plan_['in_rank'], plan_['n_entrants'])))

    last = cal[-1]
    mv = sum(p['qty'] * px[p['symbol']][last][1]
             for p in held.values() if last in px.get(p['symbol'], {}))
    return log, dict(book=label, start=cal[0], end=cal[-1], positions=len(held),
                     cash=round(cash), nav=round(cash + mv), swaps=n_swap, buys=n_buy,
                     exits=n_exit, refused_cash=n_ref_cash, refused_slot=n_ref_slot)


def _replan(cand, held, leaving, cash, marks):
    """plan() with the broker facts filled in, using the SAME arithmetic as the live code.

    `services.oa_baseage_entry.plan` reads `leaving` and `armed` off the order book through
    a live Kite handle. In a replay there is no order book, so the two sets are handed in and
    everything else - the NAV basis, the slot, the whole-share floor, the cash ladder and the
    order of the three refusal reasons - is the live function's, not a paraphrase.
    """
    st = dict(positions=[dict(p) for p in held.values()], cash=cash)

    class _FakeKite:
        def __init__(self, leaving):
            self._l = leaving

        def orders(self):
            return [dict(tradingsymbol=s, transaction_type='SELL', status='OPEN', tag=None)
                    for s in self._l]

    return live.plan(st, cand, kite=_FakeKite(set(leaving)), marks=marks)


def r2():
    print('\n' + '=' * 78)
    print('R2  THE LIVE CODE, WALKED - last %d sessions, two starting books' % LOOKBACK)
    print('=' * 78, flush=True)
    cal = calendar()
    win = cal[-LOOKBACK:]
    ev = pd.read_csv(EV164, dtype={'trigger_date': str})
    ev = ev[ev.trigger_date >= win[0]]
    events = ev.to_dict('records')
    syms = sorted(set(ev.symbol) | {p['symbol'] for p in LIVE_POSITIONS})
    print('window %s .. %s; %d events; %d symbols to load' % (win[0], win[-1], len(events),
                                                              len(syms)), flush=True)
    t0 = time.time()
    px, stdir, raw = load_series(syms)
    print('series loaded in %.0fs (%d with bars)' % (time.time() - t0, len(px)), flush=True)
    # research/164's frozen list stores the ENTRY open, not the trigger close, because its
    # engine sizes at the fill. The live `plan()` sizes off the trigger close - that is
    # deviation R2 - so the close is read back out of the same split-cut series the live
    # scanner reads, and events with no bar on their own trigger day are dropped loudly.
    keep, lost = [], 0
    for e in events:
        c = px.get(e['symbol'], {}).get(e['trigger_date'])
        if c is None:
            lost += 1
            continue
        e['trigger_close'] = float(c[1])
        keep.append(e)
    events = keep
    if lost:
        print('  %d event(s) dropped: no bar on the trigger date in the split-cut series'
              % lost)

    logs, summ = [], []
    for label, pos, csh in (('live-book', LIVE_POSITIONS, LIVE_CASH),
                            ('empty-book', [], BOOK_CAPITAL)):
        for rot, tag in ((True, ''), (False, '-norot')):
            lg, sm = walk(events, win, px, stdir, raw, pos, csh, label + tag, rot=rot)
            logs.extend(lg)
            summ.append(sm)
            print('  %-16s %3d swaps, %3d buys, %3d exits, %3d refused-cash, '
                  '%3d refused-slot -> %2d positions, NAV Rs %s'
                  % (sm['book'], sm['swaps'], sm['buys'], sm['exits'], sm['refused_cash'],
                     sm['refused_slot'], sm['positions'], format(sm['nav'], ',')), flush=True)

    print('\n  60-session walk at the live book\'s capital (research/165 §8.2 comparison)')
    win60 = cal[-WALK60:]
    ev60 = [e for e in events if e['trigger_date'] >= win60[0]]
    for rot, tag in ((True, 'with OA-ROT-1'), (False, 'un-rotated')):
        lg, sm = walk(ev60, win60, px, stdir, raw, [], BOOK_CAPITAL,
                      '60d-%s' % ('rot' if rot else 'norot'), rot=rot)
        logs.extend(lg)
        summ.append(sm)
        print('    %-14s %2d swaps, %2d buys, %2d exits, %2d refused-cash, %2d refused-slot'
              ' -> %2d positions, cash Rs %s, NAV Rs %s'
              % (tag, sm['swaps'], sm['buys'], sm['exits'], sm['refused_cash'],
                 sm['refused_slot'], sm['positions'], format(sm['cash'], ','),
                 format(sm['nav'], ',')))

    with open(RES / 'rot1_walk.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['book', 'day', 'action', 'symbol', 'qty', 'price',
                                          'ret_pct', 'why'])
        w.writeheader()
        for r in logs:
            w.writerow(r)
    swaps = [r for r in logs if r['action'] == 'SWAP']
    if swaps:
        print('\n  every swap the live code decided:')
        for r in swaps:
            print('    %-16s %s  %s' % (r['book'], r['day'], r['why']))
    return summ


def main():
    seeds_n = 5
    skip_r1 = False
    for a in sys.argv[1:]:
        if a.startswith('--seeds='):
            seeds_n = int(a.split('=', 1)[1])
        elif a == '--skip-r1':
            skip_r1 = True
    out = {}
    if not skip_r1:
        out['R1'] = r1([1000 + k for k in range(1, seeds_n + 1)])
    out['R2'] = r2()
    json.dump(out, open(RES / 'rot1_gate.json', 'w'), indent=1, default=str)
    if 'R1' in out:
        g = out['R1']['pair_pct']
        print('\nGATE R1: swap-pair agreement %.2f%% (bar is 100%%) -> %s'
              % (g, 'PASS' if g >= 100.0 else 'FAIL'))


if __name__ == '__main__':
    main()
