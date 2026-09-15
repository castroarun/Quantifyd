"""research/176 — read the block-permutation null results and answer the one
question the headline sweep left open: the daily long/flat trend book had a BETTER
Calmar than buy-and-hold on most names. Is that timing, or just exposure?
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
SPLIT_SUSPECT = ['JKLAKSHMI', 'TIMKEN', 'MFSL', 'ADANIENT', 'ABBOTINDIA', 'JSL',
                 'ABFRL', 'STAR', 'BAJFINANCE', 'RAYMOND', 'WELCORP', 'NMDC',
                 'GNFC', 'OFSS']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', default='day')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(RES, f'stage3_{a.tf}_part*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[~df.symbol.isin(SPLIT_SUSPECT)]
    for c in ['calmar', 'bh_calmar', 'null_calmar_med', 'pct_calmar']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f'{a.tf}: {len(df):,} (symbol, cell) rows over {df.symbol.nunique()} symbols, '
          f'{df.n_draws.iloc[0]} draws each\n')

    print('=' * 118)
    print('TIME-IN-MARKET-MATCHED BLOCK-PERMUTATION NULL — long/flat, next-open fill, 20 bps')
    print('The null keeps time in market, trade count and run-length distribution; it only moves WHEN the')
    print('long spells happen. pct_* is the fraction of 200 shuffles the real rule beats (0.50 = no timing skill).')
    print('=' * 118)
    g = df.groupby('signal').apply(lambda x: pd.Series({
        'n': len(x),
        'tim': x.time_in_mkt.median(),
        'trades': x.n_trades.median(),
        'CAGR': x.cagr.median(),
        'null_CAGR': x.null_cagr_med.median(),
        'BH_CAGR': x.bh_cagr.median(),
        'pct_CAGR': x.pct_cagr.median(),
        'beat_null_cagr': (x.cagr > x.null_cagr_med).mean(),
        'Calmar': x.calmar.median(),
        'null_Calmar': x.null_calmar_med.median(),
        'BH_Calmar': x.bh_calmar.median(),
        'pct_Calmar': x.pct_calmar.median(),
        'beat_null_calmar': (x.calmar > x.null_calmar_med).mean(),
        'MaxDD': x.maxdd.median(),
        'null_MaxDD': x.null_maxdd_med.median(),
        'BH_MaxDD': x.bh_maxdd.median(),
    }), include_groups=False).sort_values('beat_null_calmar', ascending=False)
    pd.set_option('display.width', 200)
    print(g.to_string(float_format=lambda v: f'{v:.4f}'))

    print('\n' + '=' * 118)
    print('READ-OUT')
    print('=' * 118)
    for s, r in g.iterrows():
        verdict_c = ('TIMING' if r.beat_null_cagr >= 0.60 else
                     'no timing skill' if r.beat_null_cagr >= 0.40 else 'ANTI-timing')
        verdict_k = ('TIMING' if r.beat_null_calmar >= 0.60 else
                     'no timing skill' if r.beat_null_calmar >= 0.40 else 'ANTI-timing')
        print(f'  {s:18s} CAGR beats own null on {r.beat_null_cagr:5.1%} of names ({verdict_c:15s}) | '
              f'Calmar on {r.beat_null_calmar:5.1%} ({verdict_k})')
    print('\n  Reminder of the pre-registered gate G1c: the best cell must beat the matched')
    print('  random null on at least 60% of names, or the effect is exposure, not timing.')


if __name__ == '__main__':
    main()
