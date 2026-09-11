# -*- coding: utf-8 -*-
"""research/160 - reconcile the panel against what Arun sees on screener.in TODAY.

Three questions, all of them "is this thing wired up correctly", none of them research:

1. How many names pass all five fundamental criteria on TODAY's figures - the live-site
   analogue, with no filing lag - versus on the point-in-time row for 2026-09-01? The two
   should be the same order of magnitude. A PIT count far below the today count means the
   lag is eating a year it should not, or the panel is thin.

2. Is `mcap_pit` on the right SCALE? Equity capital is in Rs crore and face value in rupees,
   so equity_capital / face_value is a share count in CRORES and times a rupee price gives
   Rs crore. Any unit slip there - lakh vs crore anywhere in the chain - moves market cap by
   100x and silently switches the `mcap > 1000` criterion either fully on or fully off. It is
   checked against Screener's own market cap on the 20 largest AND the 20 smallest names,
   because a multiplicative error is invisible if you only ever look at large caps.

3. Do the ratios this panel COMPUTES agree with the ones Screener PUBLISHES? ROE is computed
   here as net_profit/(equity capital + reserves); Screener's own ROE uses its own convention.
   Agreement within a few points means the plumbing is right; a systematic gap is a finding.

Nothing here is usable point-in-time - every number in the "today" column is TODAY's. The
whole file exists to prove the PIT panel is not broken, and is quoted as a reconciliation,
never as evidence about the strategy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
CACHE = RES / 'screener_cache'


def cagr(a, b, yrs):
    if a is None or b is None or a <= 0 or b <= 0 or yrs <= 0:
        return np.nan
    return 100.0 * ((b / a) ** (1.0 / yrs) - 1.0)


def main():
    uni = pd.read_csv(RES / 'universe.csv')
    tick2sym = {}
    for r in uni.itertuples():
        tick2sym.setdefault(r.screener_ticker, r.symbol)

    rows = []
    for f in sorted(CACHE.glob('*.json')):
        d = json.load(open(f))
        ann = {pd.Timestamp(k): v for k, v in (d.get('annual') or {}).items()}
        if len(ann) < 4:
            continue
        ys = sorted(ann)
        a, b = ys[-4], ys[-1]
        yrs = (b - a).days / 365.25
        lat = ann[b]
        eq = lambda y: (None if (ann[y].get('equity_capital') is None            # noqa: E731
                                 or ann[y].get('reserves') is None)
                        else ann[y]['equity_capital'] + ann[y]['reserves'])
        roes = [100.0 * ann[y]['net_profit'] / eq(y) for y in ys[-3:]
                if eq(y) and ann[y].get('net_profit') is not None and eq(y) != 0]
        top = d.get('top') or {}
        e = eq(b)
        sh = (lat.get('equity_capital') / top['face_value']
              if (top.get('face_value') and lat.get('equity_capital') is not None
                  and top['face_value'] > 0) else np.nan)
        rows.append(dict(
            ticker=d['ticker'], symbol=tick2sym.get(d['ticker']), n_fy=len(ann),
            fy_latest=str(b.date()),
            sales_g3=cagr(ann[a].get('sales'), lat.get('sales'), yrs),
            profit_g3=cagr(ann[a].get('net_profit'), lat.get('net_profit'), yrs),
            roe_avg3=(float(np.mean(roes)) if roes else np.nan),
            roce_latest=(lat.get('roce_pct') if lat.get('roce_pct') is not None else np.nan),
            is_lender=(not any(ann[y].get('roce_pct') is not None for y in ys)) and bool(roes),
            de_latest=((lat['borrowings'] / e) if (e and lat.get('borrowings') is not None
                                                   and e > 0) else np.nan),
            neg3=any(ann[y].get(k) is not None and ann[y][k] < 0
                     for y in ys[-3:] for k in ('sales', 'net_profit')),
            shares=sh, face_value=top.get('face_value'),
            mcap_today=top.get('market_cap'), price_today=top.get('current_price'),
            roe_screener=top.get('roe'), roce_screener=top.get('roce')))
    t = pd.DataFrame(rows)
    t['mcap_computed'] = t.shares * t.price_today
    t['mcap_err_pct'] = 100.0 * (t.mcap_computed - t.mcap_today) / t.mcap_today

    ok = lambda s: np.nan_to_num(s.values.astype(float), nan=-np.inf)   # noqa: E731
    c = pd.DataFrame(dict(
        growth=(ok(t.sales_g3) > 20) & (ok(t.profit_g3) > 20),
        roe=ok(t.roe_avg3) > 15,
        roce=((ok(t.roce_latest) > 15) & ~t.is_lender.values) | t.is_lender.values,
        de=t.de_latest.fillna(np.inf).values <= 0.2,
        mcap=ok(t.mcap_today) > 1000,
        no_neg=~t.neg3.values))
    c['all5'] = c.all(axis=1)
    t = pd.concat([t, c.add_prefix('p_')], axis=1)

    md, w = [], None
    w = md.append
    w('# research/160 — reconciliation against what screener.in shows today\n')
    w('Everything in the "today" column is TODAY\'s value with no filing lag. It exists only to')
    w('prove the point-in-time panel is wired correctly, and is never evidence about the')
    w('strategy.\n')
    w('## 1. How many names pass, on today\'s figures\n')
    w('%d tickers with >= 4 fiscal years.\n' % len(t))
    w('| criterion | passes | fails |')
    w('|---|---:|---:|')
    for k in ['growth', 'roe', 'roce', 'de', 'mcap', 'no_neg', 'all5']:
        w('| %s | %d | %d |' % (k, int(t['p_' + k].sum()), int((~t['p_' + k]).sum())))
    w('')
    w('**All five criteria on today\'s figures: %d names.**' % int(t.p_all5.sum()))
    w('(Arun\'s live query adds `price >= 0.9 x all-time high`, which this table deliberately')
    w('does not apply — that is the engine leg\'s condition and it cuts the list further.)\n')
    names = t[t.p_all5].sort_values('mcap_today', ascending=False)
    w('Largest 25 of them: ' + ', '.join(names.ticker.head(25)) + '\n')

    pit_path = RES / 'features_pit_monthly.csv.gz'
    if pit_path.exists():
        p = pd.read_csv(pit_path)
        last = p[p.date == p.date.max()].copy()
        lender = last.is_lender.astype(bool).values
        pc = ((np.nan_to_num(last.sales_g3, nan=-np.inf) > 20)
              & (np.nan_to_num(last.profit_g3, nan=-np.inf) > 20)
              & (np.nan_to_num(last.roe_avg3, nan=-np.inf) > 15)
              & (((np.nan_to_num(last.roce_latest, nan=-np.inf) > 15) & ~lender) | lender)
              & (last.de_latest.fillna(np.inf).values <= 0.2)
              & (np.nan_to_num(last.mcap_pit, nan=-np.inf) > 1000)
              & ~last.neg3.astype(bool).values
              & (last.n_fy_usable.values >= 4))
        w('## 2. Point-in-time row for %s vs today\n' % p.date.max())
        w('| | names |')
        w('|---|---:|')
        w('| screenable (n_fy_usable >= 4) at %s | %d |' % (p.date.max(),
                                                            int((last.n_fy_usable >= 4).sum())))
        w('| passing all five, point-in-time | **%d** |' % int(pc.sum()))
        w('| passing all five, today\'s figures | %d |' % int(t.p_all5.sum()))
        w('')
        pitset = set(last[pc].symbol)
        todayset = set(t[t.p_all5].symbol.dropna())
        w('Overlap %d; PIT-only %d; today-only %d.'
          % (len(pitset & todayset), len(pitset - todayset), len(todayset - pitset)))
        w('The two differ for one honest reason: on 2026-09-01 the FY2026 annuals (year-end')
        w('31-Mar-2026) are NOT yet usable under the 4-month lag — they become usable on')
        w('01-Aug-2026, so most names do have them, but any company whose latest page year is')
        w('FY2026 and whose FY2025 was weaker will differ. A large gap in either direction is')
        w('a bug; a small one is the lag doing its job.\n')

    w('## 3. Market-cap scale check — the 20 largest AND the 20 smallest\n')
    w('`equity_capital (Rs cr) / face_value (Rs)` = shares in **crores**; times a rupee price')
    w('= market cap in **Rs crore**. A lakh/crore slip anywhere would show as a ~100x error')
    w('and would switch the `mcap > 1000` criterion fully on or fully off.\n')
    v = t.dropna(subset=['mcap_err_pct', 'mcap_today'])
    for lab, fr in (('20 largest', v.nlargest(20, 'mcap_today')),
                    ('20 smallest', v.nsmallest(20, 'mcap_today'))):
        w('**%s by Screener market cap**\n' % lab)
        w('| ticker | Screener mcap (Rs cr) | computed (Rs cr) | error |')
        w('|---|---:|---:|---:|')
        for r_ in fr.itertuples():
            w('| %s | %.0f | %.0f | %+.1f%% |'
              % (r_.ticker, r_.mcap_today, r_.mcap_computed, r_.mcap_err_pct))
        w('')
    ae = v.mcap_err_pct.abs()
    w('Across all %d reconcilable names: median |error| %.2f%%, %.1f%% within 10%%, '
      '%.1f%% within 1%%. Worst offenders: %s.'
      % (len(v), ae.median(), 100.0 * (ae <= 10).mean(), 100.0 * (ae <= 1).mean(),
         ', '.join('%s %+.0f%%' % (r.ticker, r.mcap_err_pct)
                   for r in v.nlargest(5, 'mcap_err_pct', keep='all')
                   .head(5).itertuples())))
    w('')
    w('A handful of large |errors| is expected and is not a unit bug: companies with a second')
    w('listed class (DVRs), partly-paid shares, or an equity issue after the last balance')
    w('sheet date all move the share count off the equity-capital route. A UNIT error would')
    w('show as ~100x on every name, not on a few.\n')

    w('## 4. Computed ratios vs Screener\'s published ones\n')
    for col, pub, lab in (('roe_avg3', 'roe_screener', 'ROE (3-yr avg here, latest there)'),
                          ('roce_latest', 'roce_screener', 'ROCE (latest FY both)')):
        d_ = t.dropna(subset=[col, pub])
        diff = (d_[col] - d_[pub]).abs()
        w('- **%s**: %d comparable, median |difference| %.1f pp, %.0f%% within 5 pp'
          % (lab, len(d_), diff.median(), 100.0 * (diff <= 5).mean()))
    w('')
    w('ROE is expected to differ: this panel averages three years, Screener publishes the')
    w('latest. ROCE is taken from Screener\'s own row, so it should agree almost exactly —')
    w('any material gap there means the wrong row is being read.\n')

    (RES / 'today_reconcile.md').write_text('\n'.join(md) + '\n')
    t.to_csv(RES / 'today_reconcile.csv', index=False)
    print('\n'.join(md))


if __name__ == '__main__':
    sys.exit(main())
