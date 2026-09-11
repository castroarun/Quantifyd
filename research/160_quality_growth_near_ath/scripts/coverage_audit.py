# -*- coding: utf-8 -*-
"""research/160 step 6 - coverage and survivorship audit of the fundamental leg.

The price database deliberately keeps dead names. Screener does not: a company delisted in
2019 has no page today. So the fundamental leg sees a SURVIVING subset of the universe the
price leg trades, and any result from it is flattered by exactly that much.

Most studies write that as a sentence. This one writes it as a number: how many universe
names Screener cannot see, how many of those are dead, and how much LIQUID dead capital is
therefore invisible - because a delisted micro-cap nobody could trade is not a bias, and a
delisted name that once turned over 5 crore a day is.

It also fixes the honest START of the study. Screener's free depth is about twelve fiscal
years, so the earliest usable year is FY2015 for most names, and four filed years - the
minimum a three-year growth rate needs - do not exist until FY2018 has been filed. The exact
month is computed here rather than assumed, as the first month where >= 80% of the names that
were trading AND had ever been liquid can be screened at all.

Writes results/coverage_audit.md. Reads only.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
RES = ROOT / 'research/160_quality_growth_near_ath/results'
CACHE = RES / 'screener_cache'

LIQ1, LIQ5 = 1.0, 5.0
DEAD_BEFORE = '2026-06-01'


def main():
    uni = pd.read_csv(RES / 'universe.csv')
    recs = {}
    for f in sorted(CACHE.glob('*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        recs[d['ticker']] = d
    uni['has_screener'] = uni.screener_ticker.map(
        lambda t: bool(recs.get(t, {}).get('annual')))
    uni['n_fy'] = uni.screener_ticker.map(lambda t: recs.get(t, {}).get('n_fy', 0))
    uni['n_q'] = uni.screener_ticker.map(lambda t: recs.get(t, {}).get('n_q', 0))
    uni['source'] = uni.screener_ticker.map(lambda t: recs.get(t, {}).get('source'))
    uni['dead'] = uni.last_date < DEAD_BEFORE

    have, miss = uni[uni.has_screener], uni[~uni.has_screener]
    md = []
    w = md.append
    w('# research/160 - Coverage & survivorship audit of the fundamental leg\n')
    w('Universe = every `timeframe=\'day\'` symbol in `market_data.db`, minus the 346 funds in')
    w('`backtest_data/etf_exclusions.json` and the legacy ticker regex, minus symbols with')
    w('< 250 daily rows or nothing after 2015-01-01. Screener tickers are the DB symbol with')
    w('the NSE series suffix stripped; a symbol and its `-BE` twin share one Screener page.\n')

    w('## 1. Coverage\n')
    w('| | count | share |')
    w('|---|---:|---:|')
    w('| Universe symbols | %d | 100%% |' % len(uni))
    w('| With Screener fundamental history | %d | %.1f%% |'
      % (len(have), 100.0 * len(have) / len(uni)))
    w('| Without | %d | %.1f%% |' % (len(miss), 100.0 * len(miss) / len(uni)))
    w('| — with >= 4 fiscal years (screenable at all) | %d | %.1f%% |'
      % (int((uni.n_fy >= 4).sum()), 100.0 * (uni.n_fy >= 4).mean()))
    w('| — with a quarterly table | %d | %.1f%% |'
      % (int((uni.n_q > 0).sum()), 100.0 * (uni.n_q > 0).mean()))
    src = Counter(have.source.fillna('?'))
    w('\nPage used: ' + ', '.join('%s %d' % (k, v) for k, v in src.most_common()) +
      '. Consolidated is preferred and standalone is read only when consolidated is absent')
    w('or thin; which page answered is recorded per symbol, because silently mixing the two')
    w('makes debt and equity inconsistent between names.\n')

    w('## 2. Survivorship exposure — the number, not the sentence\n')
    w('| | universe | Screener has it | Screener does NOT |')
    w('|---|---:|---:|---:|')
    for lab, sel in (('All symbols', slice(None)),
                     ('Dead (last bar < %s)' % DEAD_BEFORE, uni.dead),
                     ('Dead AND ever tv20 >= Rs %.0fcr' % LIQ1, uni.dead & (uni.max_tv20 >= LIQ1)),
                     ('Dead AND ever tv20 >= Rs %.0fcr' % LIQ5, uni.dead & (uni.max_tv20 >= LIQ5)),
                     ('Alive', ~uni.dead)):
        s = uni if isinstance(sel, slice) else uni[sel]
        w('| %s | %d | %d | %d |' % (lab, len(s), int(s.has_screener.sum()),
                                     int((~s.has_screener).sum())))
    dead_liq = uni[uni.dead & (uni.max_tv20 >= LIQ5) & ~uni.has_screener]
    w('')
    w('**The expected bias did not appear, and the reason matters: Screener KEEPS the pages of')
    w('delisted companies.** Of the %d universe names whose price series has stopped, %d still'
      % (int(uni.dead.sum()), int(uni[uni.dead].has_screener.sum())))
    w('carry fundamental history. Exactly **%d** stopped name that once traded >= Rs %.0fcr/day has'
      % (len(dead_liq), LIQ5))
    w('no page — %.2f%% of the universe. On the Screener side this study is close to'
      % (100.0 * len(dead_liq) / len(uni)))
    w('survivorship-clean. That is the opposite of what r/158 assumed from a 638-name sample, and')
    w('it is worth stating plainly rather than repeating an inherited caveat that the data refutes.\n')
    w('**The real exposure has moved upstream, into `market_data.db` itself, where this leg cannot')
    w('measure it.** The universe carries only %d stopped series out of %d (%.1f%%) across eleven'
      % (int(uni.dead.sum()), len(uni), 100.0 * uni.dead.mean()))
    w('years — fewer than the NSE actually delisted or suspended over that period. A company that')
    w('never entered the price database is invisible to both legs AND to this audit; it cannot be')
    w('counted from inside. So the fundamental filter adds almost no survivorship of its own, and')
    w('the open question the study must carry is the price database\'s own coverage.\n')
    w('And a stopped series is not always a dead company. The one liquid name missing here is')
    w('%s — renames, series moves and demergers end a symbol without'
      % (', '.join(dead_liq.symbol) if len(dead_liq) else 'none'))
    w('ending the company (TATAMOTORS\'s series stops at a demerger, not a delisting). The engine')
    w('should treat a series that simply stops as a data event to inspect, not as a bankruptcy.\n')
    if len(dead_liq):
        w('Largest by peak traded value:\n')
        w('| symbol | last bar | peak tv20 (Rs cr) |')
        w('|---|---|---:|')
        for r_ in dead_liq.sort_values('max_tv20', ascending=False).head(15).itertuples():
            w('| %s | %s | %.1f |' % (r_.symbol, r_.last_date, r_.max_tv20))
        w('')

    w('## 3. Fiscal-year depth\n')
    w('| filed years on the page | symbols |')
    w('|---:|---:|')
    for n, c in sorted(Counter(uni.n_fy).items()):
        w('| %d | %d |' % (n, c))
    w('')
    w('Screener serves about twelve years to a signed-out reader, so the earliest fiscal year')
    w('is FY2015 for most names. **This is a hard floor on the study: four filed years — the')
    w('minimum a three-year growth rate needs — do not exist before FY2018 is filed.**\n')

    # ---- honest start ---------------------------------------------------------------
    w('## 4. The honest study start\n')
    p = pd.read_csv(RES / 'features_pit_monthly.csv.gz',
                    usecols=['date', 'symbol', 'n_fy_usable'])
    ok = p[p.n_fy_usable >= 4].groupby('date').symbol.apply(set)
    months = pd.date_range('2015-01-01', '2026-09-01', freq='MS')
    rows = []
    for d in months:
        ds = str(d.date())
        # names that were actually trading that month AND had ever been liquid
        live = uni[(uni.first_date <= ds) & (uni.last_date >= ds) & (uni.max_tv20 >= LIQ5)]
        if not len(live):
            continue
        cov = len(set(live.symbol) & ok.get(ds, set())) / len(live)
        rows.append((ds, len(live), cov))
    cov = pd.DataFrame(rows, columns=['date', 'n_live_liquid', 'screenable'])
    first80 = cov[cov.screenable >= 0.80].date.min()
    first50 = cov[cov.screenable >= 0.50].date.min()
    w('Share of names that were trading in that month and had ever turned over')
    w('>= Rs %.0fcr/day, which had four filed fiscal years and could therefore be screened:\n' % LIQ5)
    w('| month | live & ever-liquid | screenable |')
    w('|---|---:|---:|')
    for r_ in cov[cov.date.str.endswith(('-01-01', '-07-01'))].itertuples():
        w('| %s | %d | %.0f%% |' % (r_.date, r_.n_live_liquid, 100 * r_.screenable))
    w('')
    w('**First month >= 50%% screenable: %s. First month >= 80%%: %s.**' % (first50, first80))
    w('The study should start at the 80% month; anything earlier is measuring Screener\'s')
    w('depth rather than the screen. Coverage before that is not random — it is whichever')
    w('companies happen to have longer pages — so an early start is a selection effect, not')
    w('merely a smaller sample.\n')

    # ---- split-scale defect ---------------------------------------------------------
    w('## 5. Known data defect touching this leg: split scale\n')
    p2 = pd.read_csv(RES / 'features_pit_monthly.csv.gz',
                     usecols=['date', 'symbol', 'mcap_scale_suspect', 'mcap_pit'])
    sus = p2.groupby('symbol').mcap_scale_suspect.any()
    w('`market_data.db` is not retroactively split-adjusted: pre-split rows keep the old, higher')
    w('price scale, so `mcap_pit = shares x close` is INFLATED for any month preceding a split.')
    w('Each panel row carries `mcap_scale_suspect`, set when a one-day close collapse below')
    w('0.55x lies in that row\'s future.\n')
    w('- symbols with at least one suspect month: **%d of %d** (%.1f%%)'
      % (int(sus.sum()), len(sus), 100.0 * sus.mean()))
    w('- panel rows flagged: **%d of %d** (%.1f%%)'
      % (int(p2.mcap_scale_suspect.sum()), len(p2), 100.0 * p2.mcap_scale_suspect.mean()))
    w('')
    w('The flag is a candidate, not a confirmation — a genuine one-day 45%% collapse (a fraud')
    w('or a blow-up) trips it too. It is deliberately generous: the engine should run the')
    w('market-cap floor with and without the flagged rows and report the gap, in the same way')
    w('it reports the missing-data policy both ways. The share count itself is sound —')
    w('reconciliation against Screener\'s own market cap is within 10%% for ~95%% of names.\n')

    w('## 6. What the study must carry\n')
    w('1. **Screener coverage is NOT this study\'s survivorship problem.** %d of %d universe names'
      % (len(have), len(uni)))
    w('   have fundamental history, delisted ones included, and only %d liquid stopped name is'
      % len(dead_liq))
    w('   missing. Still run screened arms against an unscreened arm on the SAME sub-universe so')
    w('   the comparison is like-for-like — but the open survivorship question belongs to')
    w('   `market_data.db`\'s own universe (section 2), and cannot be answered from inside it.')
    w('2. **Restated, not as-reported.** Screener shows figures as they stand today. The')
    w('   filing lag controls the timing; it cannot undo a restatement.')
    w('3. **No pre-%s study window** — before that, coverage is the result, not the screen.' % first80)
    w('4. **Quarterly OPM is a recent-window feature** (~mid-2023 on): Screener carries about')
    w('   thirteen quarters, not a history.')
    w('5. **Market-cap floor is the softest criterion** — it depends on face value, share count')
    w('   and an unadjusted price series, where the others need only the filed statements.')

    (RES / 'coverage_audit.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:8]))
    print('...')
    print('universe %d | screener %d | missing %d | dead+liquid missing %d | first80 %s'
          % (len(uni), len(have), len(miss), len(dead_liq), first80))
    print('wrote %s' % (RES / 'coverage_audit.md'))
    cov.to_csv(RES / 'coverage_by_month.csv', index=False)


if __name__ == '__main__':
    sys.exit(main())
