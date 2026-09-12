# -*- coding: utf-8 -*-
"""research/165 - the replication gate: does the LIVE scanner produce the STUDY's signals?

This is the gate that has to pass before the live book is allowed to change ruleset. It is
deliberately run against research/164's frozen `events164.csv` (3,619 events, the adopted
spec after filtering AND the 60-bar re-arm), not against a re-derivation, so what is being
proved is that `services/oa_baseage.py` - the module the live book will actually call -
reproduces the file the published numbers were computed from.

Three phases:

  A. STUDY MODE. The live scanner run under research/161's exact universe rule (its
     substring fund pattern, no last-bar allowance). Nothing here should differ. Bar: >= 95%
     exact symbol-by-date agreement over the last 400 trading days, every mismatch named.

  B. PRODUCTION MODE. The same scanner under the curated `etf_exclusions.json` name list,
     which is deviation D4. The delta from A is the cost and benefit of that deviation, and
     it is reported as its own bucket rather than mixed into the gate.

  C. FORWARD WALK. The last 60 trading days replayed end to end at the live book's own
     capital - entries at the next open, the traded-value tie-break, the cash check, and
     SuperTrend(14,4) exits at the next open - so the 'would-have-been entries with sizes'
     are produced by the same code that will place the orders.

Read-only. No Kite, no state file, no orders.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from services import oa_baseage as spec                                  # noqa: E402

RES = Path(__file__).resolve().parents[1] / 'results'
RES.mkdir(parents=True, exist_ok=True)
STUDY_EVENTS = ROOT / 'research/164_baseage_slots_sizing/results/events164.csv'
LOOKBACK = 400
WALK = 60
BOOK_CAPITAL = 617637.68        # the live book's capital, so the sizes are the real ones


def calendar():
    con = spec.connect()
    try:
        rows = [r[0][:10] for r in con.execute(
            "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
            "AND symbol='RELIANCE' ORDER BY date")]
    finally:
        con.close()
    return rows


def run_scan(study_mode, tag):
    t0 = time.time()
    print('\n=== scanning universe (%s) ===' % tag, flush=True)
    con = spec.connect()
    syms = spec.universe(con, study_mode=study_mode)
    con.close()
    print('universe after exclusions: %d symbols' % len(syms), flush=True)
    ev = spec.scan(study_mode=study_mode, symbols=syms,
                   allow_last_bar=not study_mode, progress=400)
    print('%s: %d qualifying events in %.0fs' % (tag, len(ev), time.time() - t0), flush=True)
    return ev, set(syms)


def compare(mine, study, cal_lo, label):
    """Symbol-by-date set comparison, restricted to the window."""
    m = {(r['symbol'], r['trigger_date']) for r in mine if r['trigger_date'] >= cal_lo}
    s = {(r.symbol, r.trigger_date) for r in study.itertuples() if r.trigger_date >= cal_lo}
    both, miss, extra = m & s, s - m, m - s
    rate = 100.0 * len(both) / len(s) if s else float('nan')
    print('\n--- %s, trigger dates >= %s ---' % (label, cal_lo))
    print('study events   : %d' % len(s))
    print('scanner events : %d' % len(m))
    print('exact matches  : %d  (%.2f%% of the study list)' % (len(both), rate))
    print('MISSES (study has, scanner does not): %d' % len(miss))
    print('EXTRAS (scanner has, study does not): %d' % len(extra))
    return both, miss, extra, rate


def explain(pairs, kind, study_syms, prod_syms, study_df):
    """Bucket every mismatch by cause. Nothing is allowed to stay 'unknown' silently."""
    out = []
    for sym, d in sorted(pairs):
        why = 'UNEXPLAINED'
        if kind == 'miss':
            if sym not in prod_syms and sym in study_syms:
                why = 'excluded by the curated fund list (D4), kept by the study pattern'
            elif sym not in study_syms:
                why = 'not in either universe - check MIN_BARS / DROP'
        else:
            if sym in prod_syms and sym not in study_syms:
                why = 'kept by the curated fund list (D4), excluded by the study pattern'
            elif d >= str(study_df.trigger_date.max()):
                why = 'trigger after the study list ends (live last-bar allowance)'
        out.append(dict(symbol=sym, trigger_date=d, kind=kind, reason=why))
    return out


# ------------------------------------------------------------------ phase C: the walk
def walk(events, cal, n_days=WALK):
    """Replay the last `n_days` sessions: tie-break, cash check, next-open fills, ST exits."""
    win = cal[-n_days:]
    lo = win[0]
    syms = sorted({e['symbol'] for e in events if e['trigger_date'] >= lo})
    con = spec.connect()
    px, stx = {}, {}
    for s in syms:
        d, _ = spec.load_bars(con, s)
        if d is None:
            continue
        d = d.copy()
        d['d'] = d['date'].astype(str).str[:10]
        px[s] = d.set_index('d')[['open', 'close']]
    print('\n=== forward walk: %s .. %s, %d symbols with signals ==='
          % (win[0], win[-1], len(px)), flush=True)

    by_day = {}
    for e in events:
        if e['trigger_date'] >= lo and e['symbol'] in px:
            by_day.setdefault(e['trigger_date'], []).append(e)

    cash, held, log, refused_cash, refused_slot = BOOK_CAPITAL, {}, [], 0, 0
    for i, day in enumerate(win):
        nxt = win[i + 1] if i + 1 < len(win) else None
        # 1. exits confirmed on today's close, filled at the next open
        for s in list(held):
            if s not in px or day not in px[s].index:
                continue
            dd, line, c, _ = spec.st_state(s, asof=day, con=con)
            if dd == -1 and nxt and nxt in px[s].index:
                p = held.pop(s)
                o = float(px[s].loc[nxt, 'open'])
                cash += p['qty'] * o
                log.append(dict(day=nxt, action='SELL', symbol=s, qty=p['qty'],
                                price=round(o, 2), why='ST(14,4) flip on %s close' % day,
                                ret_pct=round(100 * (o / p['buy'] - 1), 2)))
        # 2. entries signalled on today's close, filled at the next open
        cands = sorted(by_day.get(day, []), key=lambda r: -r['tv20_cr'])
        if cands and nxt:
            mv = sum(p['qty'] * float(px[p['sym']].loc[day, 'close'])
                     for p in held.values() if day in px[p['sym']].index)
            nav = cash + mv
            slot = spec.SLOT_PCT * nav
            for r in cands:
                s = r['symbol']
                if s in held:
                    continue
                if len(held) >= spec.SLOTS:
                    refused_slot += 1
                    log.append(dict(day=nxt, action='REFUSED', symbol=s, qty=0, price=0,
                                    why='no free slot (tie-break rank by TV %.2f cr)'
                                        % r['tv20_cr'], ret_pct=0))
                    continue
                if nxt not in px[s].index:
                    continue
                o = float(px[s].loc[nxt, 'open'])
                qty = int(slot // o)
                if qty < 1:
                    continue
                cost = qty * o
                if cost > cash:
                    refused_cash += 1
                    log.append(dict(day=nxt, action='REFUSED', symbol=s, qty=qty,
                                    price=round(o, 2),
                                    why='refused for cash: needs Rs %s, have Rs %s'
                                        % (format(round(cost), ','), format(round(cash), ',')),
                                    ret_pct=0))
                    continue
                cash -= cost
                held[s] = dict(sym=s, qty=qty, buy=o)
                log.append(dict(day=nxt, action='BUY', symbol=s, qty=qty, price=round(o, 2),
                                why='ATH close %s, base %d bars, depth %.0f%%, TV %.2f cr'
                                    % (r['trigger_date'], r['x_bars'], r['depth_pct'],
                                       r['tv20_cr']), ret_pct=0))
    last = win[-1]
    mv = sum(p['qty'] * float(px[p['sym']].loc[last, 'close'])
             for p in held.values() if last in px[p['sym']].index)
    con.close()
    return log, dict(cash=round(cash), positions=len(held), nav=round(cash + mv),
                     refused_cash=refused_cash, refused_slot=refused_slot,
                     start=win[0], end=win[-1])


def main():
    cal = calendar()
    cal_lo = cal[-LOOKBACK]
    study = pd.read_csv(STUDY_EVENTS, dtype={'trigger_date': str, 'entry_date': str})
    print('study list: %d events, %s .. %s'
          % (len(study), study.trigger_date.min(), study.trigger_date.max()))
    print('calendar: %d sessions, gate window starts %s (last %d sessions)'
          % (len(cal), cal_lo, LOOKBACK))

    ev_study, syms_study = run_scan(True, 'PHASE A study mode')
    ev_prod, syms_prod = run_scan(False, 'PHASE B production mode')

    bothA, missA, extraA, rateA = compare(ev_study, study, cal_lo, 'PHASE A - study mode')
    bothB, missB, extraB, rateB = compare(ev_prod, study, cal_lo, 'PHASE B - production mode')

    rows = (explain(missA, 'miss', syms_study, syms_study, study)
            + explain(extraA, 'extra', syms_study, syms_study, study))
    rowsB = (explain(missB, 'miss', syms_study, syms_prod, study)
             + explain(extraB, 'extra', syms_study, syms_prod, study))
    for r in rows:
        r['phase'] = 'A'
    for r in rowsB:
        r['phase'] = 'B'
    allrows = rows + rowsB
    with open(RES / 'replication165.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['phase', 'kind', 'symbol', 'trigger_date', 'reason'])
        w.writeheader()
        for r in allrows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print('\nmismatch buckets:')
    for ph in ('A', 'B'):
        sub = [r for r in allrows if r['phase'] == ph]
        buckets = {}
        for r in sub:
            buckets[(r['kind'], r['reason'])] = buckets.get((r['kind'], r['reason']), 0) + 1
        for (k, why), n in sorted(buckets.items(), key=lambda x: -x[1]):
            print('  phase %s  %-6s %4d  %s' % (ph, k, n, why))

    log, summ = walk(ev_prod, cal)
    with open(RES / 'walk165.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['day', 'action', 'symbol', 'qty', 'price',
                                          'ret_pct', 'why'])
        w.writeheader()
        for r in log:
            w.writerow(r)
    buys = [r for r in log if r['action'] == 'BUY']
    sells = [r for r in log if r['action'] == 'SELL']
    print('\nforward walk %s .. %s on Rs %s of capital:'
          % (summ['start'], summ['end'], format(round(BOOK_CAPITAL), ',')))
    print('  %d buys, %d sells, %d refused for cash, %d refused for slot'
          % (len(buys), len(sells), summ['refused_cash'], summ['refused_slot']))
    print('  end: %d positions, cash Rs %s, NAV Rs %s'
          % (summ['positions'], format(summ['cash'], ','), format(summ['nav'], ',')))
    print('\n  last 25 actions:')
    for r in log[-25:]:
        print('   %s %-8s %-14s %5d @ %9.2f  %s'
              % (r['day'], r['action'], r['symbol'], r['qty'], r['price'], r['why'][:68]))

    json.dump(dict(gate_window_start=cal_lo, lookback=LOOKBACK,
                   phaseA_rate=round(rateA, 3), phaseA_miss=len(missA),
                   phaseA_extra=len(extraA),
                   phaseB_rate=round(rateB, 3), phaseB_miss=len(missB),
                   phaseB_extra=len(extraB),
                   universe_study=len(syms_study), universe_prod=len(syms_prod),
                   walk=summ, buys=len(buys), sells=len(sells)),
              open(RES / 'replication165.json', 'w'), indent=1)
    print('\nGATE: phase A agreement %.2f%% (bar is 95%%) -> %s'
          % (rateA, 'PASS' if rateA >= 95.0 else 'FAIL'))


if __name__ == '__main__':
    main()
