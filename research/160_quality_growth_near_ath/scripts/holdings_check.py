# -*- coding: utf-8 -*-
"""research/160 step 7 - do Arun's own holdings pass Arun's own screen?

The replication gate for a DISCRETIONARY process. There is no published trade list to match
here, but there is something better: the real Zerodha book. If the mechanical screen cannot
reproduce the names he actually bought, then the backtest is measuring a different strategy
from the one he runs, and the study has to say which of the two it is testing.

READ-ONLY on backtest_data/holdings_snapshots.db. Nothing is written to it.

WHAT THE SNAPSHOT HISTORY CAN AND CANNOT SAY
  * It starts 2026-04-20. The 30 names present in that first snapshot were bought at some
    unknown earlier date, so evaluating the screen on 2026-04-20 for them tests "does it pass
    TODAY", not "would it have been picked ON PURCHASE". They are reported separately, and a
    purchase date is ESTIMATED from avg_price (the last day the close sat within 2% of the
    average cost) as a clearly-labelled sensitivity, never as the headline.
  * Names first appearing AFTER that are genuine observations: the snapshot runs daily, so the
    first appearance is the purchase day to within a day. Those are the honest sample.
  * avg_price is the average cost of every tranche, so for a position built in pieces the
    estimated date is a blur, not a date. Said out loud rather than hidden in a footnote.

The price-side condition (close >= 0.9 x all-time-high close) is evaluated here too, CAUSALLY
- the running maximum of closes up to and including the evaluation day, never the whole
series. It is the one piece of the engine's job this leg borrows, because the question "did
his picks pass HIS screen" is meaningless without it.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
DB = ROOT / 'backtest_data/market_data.db'
HDB = ROOT / 'backtest_data/holdings_snapshots.db'

SEED_DAY = '2026-04-21'          # first appearance on/before this = pre-existing position
NEAR_ATH = 0.90
AVGP_TOL = 0.02


def first_appearances():
    con = sqlite3.connect('file:%s?mode=ro' % HDB, uri=True)
    rows = list(con.execute(
        'select snap_date, holdings_json from holdings_snapshots order by snap_date'))
    con.close()
    first = {}
    for d, j in rows:
        for h in json.loads(j):
            first.setdefault(h['tradingsymbol'],
                             dict(symbol=h['tradingsymbol'], first_seen=d,
                                  avg_price=h.get('avg_price'), qty=h.get('qty'),
                                  invested=h.get('invested')))
    return list(first.values()), len(rows), rows[0][0], rows[-1][0]


def load_closes(symbols):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    q = ("select symbol, date, close from market_data_unified where timeframe='day' "
         "and symbol in (%s) order by symbol, date" % ','.join('?' * len(symbols)))
    df = pd.read_sql_query(q, con, params=list(symbols))
    con.close()
    return {s: (g.date.values.astype('U10'), g.close.values.astype(float))
            for s, g in df.groupby('symbol', sort=False)}


def criteria(row):
    """Arun's screen, on one point-in-time panel row. None where the data cannot say."""
    if row is None:
        return None
    lender = bool(row.is_lender)
    g = lambda v: (float(v) if v == v else None)                      # noqa: E731
    sg, pg = g(row.sales_g3), g(row.profit_g3)
    roe, roce, de, mc = g(row.roe_avg3), g(row.roce_latest), g(row.de_latest), g(row.mcap_pit)
    return dict(
        growth=(sg is not None and pg is not None and sg > 20 and pg > 20),
        roe=(roe is not None and roe > 15),
        roce=(None if lender else (roce is not None and roce > 15)),
        de=(de is not None and de <= 0.2),
        mcap=(mc is not None and mc > 1000),
        no_neg=(not bool(row.neg3)))


def main():
    hold, n_snap, d0, d1 = first_appearances()
    uni = pd.read_csv(RES / 'universe.csv')
    unis = set(uni.symbol)
    funds = set(json.load(open(ROOT / 'backtest_data/etf_exclusions.json'))['symbols'])

    p = pd.read_csv(RES / 'features_pit_monthly.csv.gz')
    p = p.sort_values(['symbol', 'date'])
    bysym = {s: g.reset_index(drop=True) for s, g in p.groupby('symbol')}

    syms = [h['symbol'] for h in hold if h['symbol'] in unis or h['symbol'] in funds]
    px = load_closes(syms) if syms else {}

    def panel_at(sym, day):
        """The panel row the engine would have been looking at on `day` - the latest monthly
        row at or before it, exactly the forward-fill the mask contract specifies."""
        g = bysym.get(sym)
        if g is None:
            return None
        sel = g[g.date <= day]
        return sel.iloc[-1] if len(sel) else None

    def price_at(sym, day):
        pr = px.get(sym)
        if pr is None:
            return None, None, None
        i = np.searchsorted(pr[0], day, 'right') - 1
        if i < 0:
            return None, None, None
        c = pr[1][i]
        ath = float(np.max(pr[1][:i + 1]))          # causal: no bar after `day`
        return float(c), ath, (100.0 * (c / ath - 1.0) if ath else None)

    def est_purchase(sym, avg_price, before):
        """Last day before `before` on which the close sat within 2% of the average cost."""
        pr = px.get(sym)
        if pr is None or not avg_price:
            return None
        i = np.searchsorted(pr[0], before, 'right') - 1
        if i < 0:
            return None
        c, d = pr[1][:i + 1], pr[0][:i + 1]
        hit = np.where(np.abs(c / avg_price - 1.0) <= AVGP_TOL)[0]
        return str(d[hit[-1]]) if len(hit) else None

    out = []
    for h in hold:
        s = h['symbol']
        kind = ('fund' if s in funds else ('equity' if s in unis else 'not-in-universe'))
        seed = h['first_seen'] <= SEED_DAY
        est = est_purchase(s, h['avg_price'], h['first_seen']) if seed else None
        ev = h['first_seen']
        row = panel_at(s, ev)
        c = criteria(row)
        close, ath, gap = price_at(s, ev)
        rec = dict(symbol=s, kind=kind, cohort=('pre-existing' if seed else 'observed buy'),
                   eval_date=ev, est_purchase=est, avg_price=h['avg_price'],
                   invested=h['invested'],
                   has_data=(row is not None and row.n_fy_usable >= 4),
                   n_fy_usable=(int(row.n_fy_usable) if row is not None else 0),
                   close=close, ath_close=ath, pct_from_ath=(round(gap, 1) if gap is not None else None),
                   near_ath=(gap is not None and close >= NEAR_ATH * ath))
        if c:
            rec.update({('c_' + k): v for k, v in c.items()})
            fails = [k for k, v in c.items() if v is False]
            rec['fails'] = ','.join(fails)
            rec['strict'] = (len(fails) == 0)
            for k in ('sales_g3', 'profit_g3', 'roe_avg3', 'roce_latest', 'de_latest',
                      'mcap_pit', 'opm_slope3', 'opm_range3'):
                rec[k] = (round(float(getattr(row, k)), 2)
                          if getattr(row, k) == getattr(row, k) else None)
        else:
            rec['fails'] = 'no-fundamental-data'
            rec['strict'] = False
        # the same evaluation at the estimated purchase date, for the pre-existing cohort
        if est:
            r2 = panel_at(s, est)
            c2 = criteria(r2)
            cl2, at2, gp2 = price_at(s, est)
            rec['strict_at_est'] = (c2 is not None and
                                    not [k for k, v in c2.items() if v is False])
            rec['near_ath_at_est'] = (gp2 is not None and cl2 >= NEAR_ATH * at2)
        out.append(rec)

    d = pd.DataFrame(out).sort_values(['cohort', 'symbol'])
    d.to_csv(RES / 'holdings_check.csv', index=False)

    eq = d[d.kind == 'equity']
    obs = eq[eq.cohort == 'observed buy']
    pre = eq[eq.cohort == 'pre-existing']

    def rate(fr, col='strict'):
        return (100.0 * fr[col].mean()) if len(fr) else float('nan')

    md, w = [], None
    w = md.append
    w('# research/160 - Arun\'s own holdings vs the mechanical screen\n')
    w('Source: `backtest_data/holdings_snapshots.db`, %d daily snapshots, %s to %s, read-only.'
      % (n_snap, d0, d1))
    w('%d distinct symbols ever held: %d equities in the universe, %d funds/ETFs, %d not in'
      % (len(d), int((d.kind == 'equity').sum()), int((d.kind == 'fund').sum()),
         int((d.kind == 'not-in-universe').sum())))
    w('the price database at all.\n')
    w('**The two cohorts are not the same evidence.** %d names were already held when the'
      % len(pre))
    w('snapshot history begins on %s, so for them the screen is being asked "does this pass' % d0)
    w('today", not "would it have been picked on purchase". %d names first appear later; the'
      % len(obs))
    w('snapshot runs daily, so those first appearances ARE purchase days and they are the')
    w('honest sample.\n')

    w('## Hit rate of the mechanical screen on his real picks\n')
    w('| cohort | n | has fundamental data | passes the full screen | passes screen AND near-ATH |')
    w('|---|---:|---:|---:|---:|')
    for lab, fr in (('Observed buys (%s onward)' % SEED_DAY, obs),
                    ('Pre-existing at %s' % d0, pre),
                    ('All equities held', eq)):
        if not len(fr):
            continue
        both = fr.strict & fr.near_ath
        w('| %s | %d | %d (%.0f%%) | **%d (%.0f%%)** | %d (%.0f%%) |'
          % (lab, len(fr), int(fr.has_data.sum()), 100 * fr.has_data.mean(),
             int(fr.strict.sum()), rate(fr), int(both.sum()), 100 * both.mean()))
    w('')

    w('## Where the screen and the man disagree\n')
    cc = [c for c in d.columns if c.startswith('c_')]
    w('| criterion | of %d equities held: pass | fail | n/a (lender or no data) |' % len(eq))
    w('|---|---:|---:|---:|')
    for c in cc:
        col = eq[c]
        w('| %s | %d | %d | %d |' % (c[2:], int((col == True).sum()),  # noqa: E712
                                     int((col == False).sum()), int(col.isna().sum())))
    w('')
    fc = {}
    for f in eq.fails.fillna(''):
        for k in [x for x in f.split(',') if x]:
            fc[k] = fc.get(k, 0) + 1
    w('Most common reason a held name fails: ' +
      ', '.join('%s %d' % (k, v) for k, v in sorted(fc.items(), key=lambda x: -x[1])) + '.\n')

    # ---- which dial is rejecting his picks? ----------------------------------------
    w('## Which dial rejects his picks\n')
    w('A low hit rate is only useful if you know what would fix it. Same %d equities, same'
      % len(eq))
    w('evaluation dates, one criterion relaxed at a time:\n')
    w('| variant | of %d held equities, pass |' % len(eq))
    w('|---|---:|')
    P = eq[[c for c in eq.columns if c.startswith('c_')]].copy()
    base = ['c_growth', 'c_roe', 'c_roce', 'c_de', 'c_mcap', 'c_no_neg']

    def npass(drop=(), extra=None):
        m = pd.Series(True, index=eq.index)
        for c in base:
            if c in drop:
                continue
            m &= P[c].fillna(True).astype(bool)      # n/a (lender ROCE) counts as pass
        if extra is not None:
            m &= extra
        return int(m.sum())

    g15 = (eq.sales_g3 > 15) & (eq.profit_g3 > 15)
    g10 = (eq.sales_g3 > 10) & (eq.profit_g3 > 10)
    sales_only = eq.sales_g3 > 20
    w('| the full screen as written | %d |' % npass())
    w('| growth bar lowered to 15%% | %d |' % npass(drop=('c_growth',), extra=g15.fillna(False)))
    w('| growth bar lowered to 10%% | %d |' % npass(drop=('c_growth',), extra=g10.fillna(False)))
    w('| growth on SALES only (profit growth dropped) | %d |'
      % npass(drop=('c_growth',), extra=sales_only.fillna(False)))
    for c, lab in (('c_growth', 'growth dropped entirely'), ('c_de', 'debt/equity dropped'),
                   ('c_roe', 'ROE dropped'), ('c_roce', 'ROCE dropped'),
                   ('c_mcap', 'market-cap floor dropped')):
        w('| %s | %d |' % (lab, npass(drop=(c,))))
    w('| growth AND debt/equity both dropped | %d |' % npass(drop=('c_growth', 'c_de')))
    w('| near-ATH condition alone (no fundamentals) | %d |' % int(eq.near_ath.sum()))
    w('')
    w('Read it as a description of his process, not a scoring of it: he is buying names that are')
    w('near their highs and profitable, without insisting on 20%+ growth on BOTH the top and')
    w('bottom line, and without the debt-free constraint. Those two dials are what separate the')
    w('mechanical screen from the book.\n')

    w('## Every equity held, at its evaluation date\n')
    w('| symbol | cohort | eval date | n FY | sales g3 | profit g3 | ROE3 | ROCE | D/E | mcap Rs cr | % from ATH | verdict |')
    w('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r_ in eq.sort_values(['cohort', 'strict', 'symbol'],
                             ascending=[True, False, True]).itertuples():
        v = 'PASS' if r_.strict else ('no data' if r_.fails == 'no-fundamental-data'
                                      else 'fails ' + str(r_.fails))
        if r_.strict and not r_.near_ath:
            v += ' (not near ATH)'
        w('| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |'
          % (r_.symbol, r_.cohort, r_.eval_date, r_.n_fy_usable,
             getattr(r_, 'sales_g3', None), getattr(r_, 'profit_g3', None),
             getattr(r_, 'roe_avg3', None), getattr(r_, 'roce_latest', None),
             getattr(r_, 'de_latest', None), getattr(r_, 'mcap_pit', None),
             r_.pct_from_ath, v))
    w('')
    if 'strict_at_est' in d.columns and pre.strict_at_est.notna().any():
        w('**Sensitivity, pre-existing cohort at the avg-cost-implied purchase date** '
          '(estimate, not a record): %d of %d pass the screen there vs %d of %d at %s. '
          'A position built in tranches makes that date a blur; treat it as a direction, '
          'not a number.\n'
          % (int(pre.strict_at_est.fillna(False).sum()), len(pre),
             int(pre.strict.sum()), len(pre), d0))

    (RES / 'holdings_check.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('wrote %s and holdings_check.csv' % (RES / 'holdings_check.md'))


if __name__ == '__main__':
    sys.exit(main())
