# -*- coding: utf-8 -*-
"""research/165 - the CONVERSION DRY-RUN. The table Arun approves before the switch flips.

For every one of the live book's open positions it answers one question: on the last
official close, what does the NEW rule say, and what does the OLD rule say? Side by side,
with the numbers behind both, so the conversion can be approved on evidence rather than on
a promise that it is equivalent.

Then it shows what the new scanner would arm for the next open - the signals from that same
close - with the position size the book's own NAV implies and the cash check applied.

Read-only by construction: it never imports Kite, never takes the state lock, never writes
`oa_real_state.json`, and runs with whatever `OA_RULESET` happens to be set to. Prices come
from `market_data.db` only, which is also why it works on a weekend with a stale token.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from services import oa_baseage as spec                                  # noqa: E402

RES = Path(__file__).resolve().parents[1] / 'results'
RES.mkdir(parents=True, exist_ok=True)
STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
STOP_PCT = 0.08
TRAIL_N = 15


def sma_n(d, n=TRAIL_N):
    c = d['close'].to_numpy(np.float64)
    return float(c[-n:].mean()) if len(c) >= n else None


def main():
    asof = spec.last_session()
    st = json.load(open(STATE))
    con = spec.connect()
    rows = []
    for p in st['positions']:
        s = p['symbol']
        d, cut = spec.load_bars(con, s, asof)
        if d is None:
            rows.append(dict(symbol=s, qty=p['qty'], buy=p['buy'], close=None,
                             verdict_new='NO DATA', verdict_old='NO DATA'))
            continue
        last_date = str(d['date'].iloc[-1])[:10]
        c = float(d['close'].iloc[-1])
        dirs, line = spec.supertrend_full(d['high'].to_numpy(np.float64),
                                          d['low'].to_numpy(np.float64),
                                          d['close'].to_numpy(np.float64))
        sd, sl = int(dirs[-1]), float(line[-1])
        sma = sma_n(d)
        old = ('SELL (-8% stop, %.2f)' % p['stop'] if c <= p['stop']
               else ('SELL (15-SMA %.2f)' % sma if sma and c < sma else 'HOLD'))
        new = 'SELL at next open' if sd == -1 else 'HOLD'
        rows.append(dict(symbol=s, qty=p['qty'], buy=p['buy'], bar=last_date,
                         close=round(c, 2), value=round(p['qty'] * c),
                         pnl_pct=round(100 * (c / p['buy'] - 1), 2),
                         st_line=round(sl, 2), st_dir=sd,
                         to_st_pct=round(100 * (c / sl - 1), 1) if sl else None,
                         sma15=round(sma, 2) if sma else None,
                         to_sma_pct=round(100 * (c / sma - 1), 1) if sma else None,
                         stop=p['stop'], split_cut=cut,
                         verdict_new=new, verdict_old=old))
    con.close()

    cash = float(st['cash'])
    mv = sum(r.get('value') or 0 for r in rows)
    nav = cash + mv
    slot = spec.SLOT_PCT * nav

    print('OA - Open Alpha - Base Age CONVERSION DRY-RUN')
    print('last official session in market_data.db: %s' % asof)
    print('capital Rs %s | cash Rs %s | positions value Rs %s | NAV Rs %s | slot (6.25%%) Rs %s'
          % (format(round(st['capital']), ','), format(round(cash), ','),
             format(round(mv), ','), format(round(nav), ','), format(round(slot), ',')))
    print()
    hdr = ('%-13s %5s %9s %9s %8s %9s %6s %8s %9s %7s  %-22s %s'
           % ('symbol', 'qty', 'buy', 'close', 'P&L%', 'ST(14,4)', 'dir', 'to ST%',
              '15-SMA', 'to SMA%', 'BASE AGE verdict', 'legacy verdict'))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        if r.get('close') is None:
            print('%-13s %5d  NO DATA' % (r['symbol'], r['qty']))
            continue
        print('%-13s %5d %9.2f %9.2f %+8.2f %9.2f %6d %+8.1f %9.2f %+7.1f  %-22s %s'
              % (r['symbol'], r['qty'], r['buy'], r['close'], r['pnl_pct'], r['st_line'],
                 r['st_dir'], r['to_st_pct'], r['sma15'] or 0, r['to_sma_pct'] or 0,
                 r['verdict_new'], r['verdict_old']))
    n_sell_new = sum(1 for r in rows if r.get('verdict_new', '').startswith('SELL'))
    n_sell_old = sum(1 for r in rows if r.get('verdict_old', '').startswith('SELL'))
    print('\nBase Age would sell %d of %d at the next open; the legacy rule would sell %d.'
          % (n_sell_new, len(rows), n_sell_old))

    # ---- what the new scanner arms for the next open --------------------------------
    print('\n=== entry candidates from the %s close (for the next open) ===' % asof)
    sys.stdout.flush()
    sigs = spec.scan(asof=asof, allow_last_bar=True, progress=400)
    today = sorted([x for x in sigs if x['trigger_date'] == asof],
                   key=lambda r: -r['tv20_cr'])
    held = {p['symbol'] for p in st['positions']}
    free = spec.SLOTS - len(held)
    print('%d qualifying signal(s); %d slot(s) free; cash Rs %s; slot size Rs %s'
          % (len(today), free, format(round(cash), ','), format(round(slot), ',')))
    remaining, taken = cash, 0
    lines = []
    for r in today:
        s = r['symbol']
        px = r['trigger_close']
        qty = int(slot // px)
        cost = qty * px
        if s in held:
            verdict = 'skip - already held'
        elif taken >= free:
            verdict = 'refused - no free slot'
        elif qty < 1:
            verdict = 'refused - slot buys no whole share'
        elif cost > remaining:
            verdict = ('REFUSED FOR CASH - needs Rs %s, Rs %s left'
                       % (format(round(cost), ','), format(round(remaining), ',')))
        else:
            remaining -= cost
            taken += 1
            verdict = 'ARM buy %d for the open (~Rs %s)' % (qty, format(round(cost), ','))
        lines.append((s, r, qty, cost, verdict))
        print('  %-13s close %9.2f  prev ATH %9.2f (%s, %d bars)  depth %5.1f%%  '
              'TV %6.2f cr  ->  %s'
              % (s, px, r['prev_ath'], r['prev_ath_date'], r['x_bars'], r['depth_pct'],
                 r['tv20_cr'], verdict))
    if not today:
        print('  (none - no symbol made a qualifying new all-time-high close that day)')
    print('\ncash after the armed entries: Rs %s' % format(round(remaining), ','))

    json.dump(dict(asof=asof, nav=round(nav), cash=round(cash), slot=round(slot),
                   positions=rows,
                   candidates=[dict(symbol=s, qty=q, cost=round(c), verdict=v,
                                    trigger_close=r['trigger_close'], prev_ath=r['prev_ath'],
                                    prev_ath_date=r['prev_ath_date'], x_bars=r['x_bars'],
                                    depth_pct=r['depth_pct'], tv20_cr=r['tv20_cr'])
                               for s, r, q, c, v in lines]),
              open(RES / 'dryrun165.json', 'w'), indent=1, default=str)
    print('\nwrote %s' % (RES / 'dryrun165.json'))


if __name__ == '__main__':
    main()
