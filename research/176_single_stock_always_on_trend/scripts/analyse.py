"""research/176 — aggregate a stage's per-name cells into the basket verdict.

Pre-registered primary metric (locked in the STATUS doc before any cell ran):
  beat_rate = fraction of basket names whose NET CAGR at 20 bps round trip exceeds
              that same name's buy-and-hold CAGR over the same window.
Gates: G1a beat_rate >= 0.55 in BOTH halves; G1b neighbours >= 80% of the best;
       G2 still clears at 40 bps.

Usage: analyse.py --glob 'stage1_day_part*.csv' --tag day
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
SPLIT_SUSPECT = ['JKLAKSHMI', 'TIMKEN', 'MFSL', 'ADANIENT', 'ABBOTINDIA', 'JSL',
                 'ABFRL', 'STAR', 'BAJFINANCE', 'RAYMOND', 'WELCORP', 'NMDC',
                 'GNFC', 'OFSS']
NAMED = ['MARUTI', 'RELIANCE', 'HDFCBANK']


def agg(df):
    g = df.groupby(['window', 'signal', 'policy', 'fill'])
    out = g.apply(lambda x: pd.Series({
        'n_names': len(x),
        'beat_rate20': float((x.cagr20 > x.bh_cagr).mean()),
        'beat_rate10': float((x.cagr10 > x.bh_cagr).mean()),
        'beat_rate40': float((x.cagr40 > x.bh_cagr).mean()),
        'med_cagr20': float(x.cagr20.median()),
        'med_bh_cagr': float(x.bh_cagr.median()),
        'med_excess': float((x.cagr20 - x.bh_cagr).median()),
        'med_maxdd': float(x.maxdd.median()),
        'med_calmar': float(pd.to_numeric(x.calmar20, errors='coerce').median()),
        'med_bh_calmar': float(pd.to_numeric(x.bh_calmar, errors='coerce').median()),
        'calmar_beat': float((pd.to_numeric(x.calmar20, errors='coerce') >
                              pd.to_numeric(x.bh_calmar, errors='coerce')).mean()),
        'med_switch_yr': float(x.switches_yr.median()),
        'med_trades': float(x.n_trades.median()),
        'med_wr': float(pd.to_numeric(x.win_rate, errors='coerce').median()),
        'med_exp': float(pd.to_numeric(x.expectancy, errors='coerce').median()),
        'med_tim': float(x.time_in_mkt.median()),
        'pos_cagr_share': float((x.cagr20 > 0).mean()),
    }), include_groups=False).reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keep-suspect', action='store_true')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RES, a.glob)))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f'loaded {len(df):,} rows from {len(files)} shards, '
          f'{df.symbol.nunique()} symbols', flush=True)
    if not a.keep_suspect:
        n0 = df.symbol.nunique()
        df = df[~df.symbol.isin(SPLIT_SUSPECT)]
        print(f'excluded {n0 - df.symbol.nunique()} split-suspect names '
              f'-> {df.symbol.nunique()} symbols', flush=True)

    res = agg(df)
    res.to_csv(os.path.join(RES, f'agg_{a.tag}.csv'), index=False)

    print('\n' + '=' * 110)
    print(f'TOP CELLS BY PRE-REGISTERED METRIC  (window=full, fill=next_open)  [{a.tag}]')
    print('=' * 110)
    for pol in ['long_flat', 'long_short', 'short_flat']:
        s = res[(res.window == 'full') & (res.fill == 'next_open') & (res.policy == pol)]
        s = s.sort_values('beat_rate20', ascending=False).head(8)
        print(f'\n--- policy = {pol} ---')
        print(s[['signal', 'n_names', 'beat_rate20', 'beat_rate40', 'med_cagr20',
                 'med_bh_cagr', 'med_excess', 'med_maxdd', 'med_calmar',
                 'med_bh_calmar', 'calmar_beat', 'med_switch_yr', 'med_wr',
                 'med_exp', 'med_tim']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print('\n' + '=' * 110)
    print('BEST CELL PER WINDOW (both halves must pass G1a beat_rate >= 0.55)')
    print('=' * 110)
    for pol in ['long_flat', 'long_short']:
        print(f'\n--- {pol}, next_open ---')
        for wnd in ['full', 'h1', 'h2']:
            s = res[(res.window == wnd) & (res.fill == 'next_open') & (res.policy == pol)]
            if s.empty:
                continue
            b = s.sort_values('beat_rate20', ascending=False).iloc[0]
            print(f'  {wnd:5s} best={b.signal:18s} beat20={b.beat_rate20:.3f} '
                  f'beat40={b.beat_rate40:.3f} medCAGR={b.med_cagr20:+.3%} '
                  f'BH={b.med_bh_cagr:+.3%} n={int(b.n_names)}')
        # stability: is the SAME cell good in both halves?
        f = res[(res.window == 'full') & (res.fill == 'next_open') & (res.policy == pol)]
        best = f.sort_values('beat_rate20', ascending=False).iloc[0].signal
        for wnd in ['h1', 'h2']:
            r = res[(res.window == wnd) & (res.fill == 'next_open') &
                    (res.policy == pol) & (res.signal == best)]
            if len(r):
                r = r.iloc[0]
                print(f'  full-best {best} in {wnd}: beat20={r.beat_rate20:.3f} '
                      f'medExcess={r.med_excess:+.3%}')

    print('\n' + '=' * 110)
    print('FILL SENSITIVITY — honest next_open vs optimistic signal_close (window=full)')
    print('=' * 110)
    for pol in ['long_flat', 'long_short']:
        s = res[(res.window == 'full') & (res.policy == pol)]
        p = s.pivot_table(index='signal', columns='fill',
                          values=['beat_rate20', 'med_cagr20'])
        p['delta_beat'] = p[('beat_rate20', 'signal_close')] - p[('beat_rate20', 'next_open')]
        p['delta_cagr'] = p[('med_cagr20', 'signal_close')] - p[('med_cagr20', 'next_open')]
        print(f'\n--- {pol}: median across {len(p)} signal cells ---')
        print(f"  beat_rate  next_open {p[('beat_rate20','next_open')].median():.3f}  "
              f"signal_close {p[('beat_rate20','signal_close')].median():.3f}  "
              f"delta {p['delta_beat'].median():+.3f}")
        print(f"  med CAGR   next_open {p[('med_cagr20','next_open')].median():+.3%}  "
              f"signal_close {p[('med_cagr20','signal_close')].median():+.3%}  "
              f"delta {p['delta_cagr'].median():+.3%}")

    print('\n' + '=' * 110)
    print('SUPERTREND PLATEAU MAP — beat_rate20, window=full, next_open')
    print('=' * 110)
    for pol in ['long_flat', 'long_short']:
        s = res[(res.window == 'full') & (res.fill == 'next_open') &
                (res.policy == pol) & (res.signal.str.startswith('ST_'))].copy()
        if s.empty:
            continue
        s['p'] = s.signal.str.split('_').str[1].astype(int)
        s['m'] = s.signal.str.split('_').str[2].astype(float)
        print(f'\n--- {pol} ---')
        print(s.pivot_table(index='p', columns='m', values='beat_rate20').to_string(
            float_format=lambda v: f'{v:.3f}'))

    print('\n' + '=' * 110)
    print("THE THREE NAMES ARUN NAMED — where they sit in the basket (window=full, next_open)")
    print('=' * 110)
    for pol in ['long_flat', 'long_short']:
        print(f'\n--- {pol} ---')
        sub = df[(df.window == 'full') & (df.fill == 'next_open') & (df.policy == pol)]
        for nm in NAMED:
            x = sub[sub.symbol == nm]
            if x.empty:
                continue
            best = x.sort_values('cagr20', ascending=False).iloc[0]
            bh = best.bh_cagr
            n_beat = int((x.cagr20 > x.bh_cagr).sum())
            print(f'  {nm:10s} B&H {bh:+.2%} | best of {len(x)} cells: '
                  f'{best.signal:18s} {best.cagr20:+.2%} (dd {best.maxdd:.1%}, '
                  f'{best.switches_yr:.0f} sw/yr) | cells beating B&H: {n_beat}/{len(x)}')
            # Arun's exact cells
            for want in ['ST_7_3.0', 'MST_7_5.0_7_2.0']:
                y = x[x.signal == want]
                if len(y):
                    y = y.iloc[0]
                    print(f'      {want:18s} {y.cagr20:+.2%} vs B&H {y.bh_cagr:+.2%} '
                          f'(dd {y.maxdd:.1%}, {y.switches_yr:.0f} sw/yr, '
                          f'{y.n_trades} trades)')

    print('\n' + '=' * 110)
    print("ARUN'S OWN CELLS ACROSS THE BASKET (window=full, next_open)")
    print('=' * 110)
    for want in ['ST_7_3.0', 'MST_7_5.0_7_2.0', 'EMA_20_50', 'EMA_50_200']:
        for pol in ['long_flat', 'long_short']:
            r = res[(res.window == 'full') & (res.fill == 'next_open') &
                    (res.policy == pol) & (res.signal == want)]
            if len(r):
                r = r.iloc[0]
                print(f'  {want:18s} {pol:11s} beat20={r.beat_rate20:.3f} '
                      f'beat40={r.beat_rate40:.3f} medCAGR={r.med_cagr20:+.2%} '
                      f'vsBH={r.med_excess:+.2%} calmarBeat={r.calmar_beat:.3f} '
                      f'sw/yr={r.med_switch_yr:.0f} exp/trade={r.med_exp:+.4f}')
    print('\nwrote', os.path.join(RES, f'agg_{a.tag}.csv'))


if __name__ == '__main__':
    main()
