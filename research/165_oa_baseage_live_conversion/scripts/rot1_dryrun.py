# -*- coding: utf-8 -*-
"""research/165 - the OA-ROT-1 dry run on the REAL Open Alpha book. Read-only.

No Kite, no lock, no state write, no order. It reads `backtest_data/oa_real_state.json` and
`backtest_data/market_data.db` and prints what the swap rule would do at the next open, plus
the two hypotheticals that matter for reading the answer honestly:

  (a) THE BOOK AS IT IS. 11 of 16 slots used and Rs 1.88 lakh of cash, so a qualifying signal
      is taken, not refused - and OA-ROT-1 only ever looks at a signal the book REFUSED.
  (b) THE BOOK IF IT WERE FULL. Suppose every slot were taken and the cash gone, so tonight's
      signal were refused. Which holding would be sold? The rule needs the deepest loss to be
      worse than -10%, so the answer has to state the deepest loss, not just the name.

Usage:  python3 research/165_oa_baseage_live_conversion/scripts/rot1_dryrun.py [--asof DATE]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services import oa_baseage as spec                                  # noqa: E402
from services import oa_baseage_entry as live                            # noqa: E402
from services.oa_real import OA_RULESET, OA_ROT1, ROT1_MARGIN_PCT        # noqa: E402

STATE = ROOT / 'backtest_data' / 'oa_real_state.json'


def main():
    asof = sys.argv[sys.argv.index('--asof') + 1] if '--asof' in sys.argv else None
    asof = asof or spec.last_session()
    st = json.load(open(STATE))
    pos = st['positions']
    cash = float(st['cash'])
    print('OA - Open Alpha REAL, dry run of OA-ROT-1 on the %s close' % asof)
    print('switches: OA_RULESET=%r  OA_ROT1=%s  margin=-%.1f%%  slots=%d @ %.2f%% of NAV\n'
          % (OA_RULESET, OA_ROT1, ROT1_MARGIN_PCT, spec.SLOTS, 100 * spec.SLOT_PCT))

    marks, stale = live.official_closes([p['symbol'] for p in pos], asof)
    for s, why in stale:
        print('  STALE %-14s %s' % (s, why))
    mv = sum(p['qty'] * marks.get(p['symbol'], p['buy']) for p in pos)
    nav = mv + cash
    print('book: %d/%d slots used, %d free.  positions Rs %s + cash Rs %s = NAV Rs %s.  '
          'slot Rs %s'
          % (len(pos), spec.SLOTS, spec.SLOTS - len(pos), format(round(mv), ','),
             format(round(cash), ','), format(round(nav), ','),
             format(round(spec.SLOT_PCT * nav), ',')))

    rows = sorted(((100.0 * (marks[p['symbol']] / p['buy'] - 1.0), p) for p in pos
                   if p['symbol'] in marks), key=lambda t: t[0])
    print('\n%-14s %5s %10s %10s %9s   %s'
          % ('symbol', 'qty', 'buy', asof, 'P&L %', 'swap-out rank'))
    for k, (r, p) in enumerate(rows, 1):
        flag = '  <-- deepest loss' if k == 1 else ''
        print('%-14s %5d %10.2f %10.2f %+8.2f%%   %2d%s'
              % (p['symbol'], p['qty'], p['buy'], marks[p['symbol']], r, k, flag))
    worst_r, worst_p = rows[0]

    # (a) the book as it is
    cand, _ = live.candidates(asof=asof)
    print('\nqualifying Base Age signals on the %s close: %d' % (asof, len(cand)))
    con = spec.connect()
    try:
        for r in cand:
            rs = live.rs252(con, r['symbol'], asof)
            print('   %-14s close %9.2f  base %4d bars  depth %5.1f%%  TV %7.2f cr  '
                  'rs252 %s' % (r['symbol'], r['trigger_close'], r['x_bars'], r['depth_pct'],
                                r['tv20_cr'],
                                ('%+.1f%%' % rs) if rs is not None else 'n/a'))
    finally:
        con.close()
    orders, refusals, ctx = live.plan(st, cand, kite=None)
    for o in orders:
        print('   TAKEN    %-14s BUY %d at the next open, Rs %s (cash left Rs %s)'
              % (o['symbol'], o['qty'], format(o['est_cost'], ','),
                 format(ctx['cash_left'], ',')))
    for s, why in refusals:
        print('   refused  %-14s %s' % (s, why))

    print('\n(a) THE BOOK AS IT IS')
    if not ctx['turned_away']:
        print('    No qualifying signal is refused: %d slot(s) free and Rs %s of cash fund '
              'every signal outright.' % (ctx['free'], format(round(cash), ',')))
        print('    OA-ROT-1 CANNOT FIRE at the next open. It only ever looks at a signal the '
              'book had to turn away.')
    else:
        con = spec.connect()
        try:
            for r in ctx['turned_away']:
                r['rs252'] = live.rs252(con, r['symbol'], asof)
        finally:
            con.close()
        plan_, why = live.rot1_pick(pos, marks, ctx['turned_away'],
                                    cash=ctx['cash_after_entries'],
                                    armed_cost=ctx['armed_cost'])
        print('    %s' % (('swap: sold %s (%+.1f%%) for %s (rs252 rank %d of %d)'
                           % (plan_['out_symbol'], plan_['out_ret_pct'], plan_['in_symbol'],
                              plan_['in_rank'], plan_['n_entrants'])) if plan_ else why))

    # (b) the counterfactual full book
    print('\n(b) THE BOOK IF IT WERE FULL - the first hypothetical')
    print('    Suppose all %d slots were taken and the cash gone, so tonight\'s signal were '
          'refused.' % spec.SLOTS)
    print('    The holding with the largest loss versus its buy price is %s at %+.2f%%.'
          % (worst_p['symbol'], worst_r))
    if -worst_r >= ROT1_MARGIN_PCT:
        print('    That IS worse than -%.1f%%, so %s would be sold at the next open.'
              % (ROT1_MARGIN_PCT, worst_p['symbol']))
    else:
        print('    That is NOT worse than -%.1f%%, so NO SWAP would fire even then. The rule '
              'would need %s to fall a further %.1f%% (to Rs %.2f) before it qualified.'
              % (ROT1_MARGIN_PCT, worst_p['symbol'],
                 100.0 * (1 - (worst_p['buy'] * (1 - ROT1_MARGIN_PCT / 100.0))
                          / marks[worst_p['symbol']]),
                 worst_p['buy'] * (1 - ROT1_MARGIN_PCT / 100.0)))
    print('\nNothing was placed, nothing was written.')


if __name__ == '__main__':
    main()
