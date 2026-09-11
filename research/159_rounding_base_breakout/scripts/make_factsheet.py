"""
research/159 — client factsheet (STATUS 11.6) for the best v3 configuration,
benchmarked against NIFTYBEES buy-and-hold and overlaid with Open Alpha.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/arun/quantifyd/research/_utilities')
from tearsheet import generate_tearsheet

RES = Path(__file__).resolve().parents[1] / 'results'
R154 = Path(__file__).resolve().parents[2] / '154_multi_system_blends' / 'results'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'ST_14_4_h0_s15_k3_a0.90_sl16'

z = np.load(RES / ('final_curve_%s.npz' % TAG), allow_pickle=True)
idx = pd.to_datetime([str(d) for d in z['dates']])
nav = pd.Series(z['nav'], index=idx).dropna()
bh = pd.Series(z['bh'], index=idx).dropna()

extra = None
oa_p = R154 / 'oa_navs30.csv'
if oa_p.exists():
    oa = pd.read_csv(oa_p, parse_dates=['date']).set_index('date').median(axis=1)
    extra = oa.reindex(nav.index).ffill()

meta = {
    'Strategy': 'Rounding base -> shelf breakout near ATH (research/159 v3)',
    'Universe': 'NSE cash, all daily symbols, 20d median traded value >= Rs2 cr',
    'Entry': 'saucer base, then close > 15-bar shelf high (range <=12%), '
             'close >= 0.90x all-time-high close, volume >= 3x 20-bar median, up candle',
    'Fill': 'next-day open, both sides',
    'Exit': 'SuperTrend(14,4) close-based trail',
    'Book': '16 slots @ 6.25% of NAV, Rs10L, idle cash 5.5% p.a.',
    'Costs': '25 bps per side; after-tax 20% STCG / 12.5% LTCG with FY loss netting',
    'Robustness': '30-seed median; window 03-Jan-2005 -> 11-Sep-2026',
    'Verdict': 'SIGNAL, not STRATEGY - fails the 20% CAGR floor and the pre-2016 window',
}
generate_tearsheet(nav, bh, 'Rounding Base Shelf Breakout (research/159)', meta,
                   out_dir=str(RES), extra_nav=extra, extra_label='Open Alpha (r/154)')
print('factsheet written to', RES)
for p in sorted(RES.glob('*.png')):
    print('  ', p.name, p.stat().st_size)
